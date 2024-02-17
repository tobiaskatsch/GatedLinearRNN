import re
import torch
import jax.numpy as jnp
import jax
import soundfile as sf
from tqdm import tqdm
from jax import random
import torchaudio
import numpy as np
import os
from flax_gate_loop.text_2_speech_models import Text2SpeechModel
from flax_gate_loop.language_models import GateLoopLM, TransformerLM


vocab = {
    'AA0': 0, 'AA1': 1, 'AA2': 2, 'AE0': 3, 'AE1': 4, 'AE2': 5, 'AH0': 6, 'AH1': 7, 'AH2': 8,
    'AO0': 9, 'AO1': 10, 'AO2': 11, 'AW0': 12, 'AW1': 13, 'AW2': 14, 'AY0': 15, 'AY1': 16, 'AY2': 17,
    'B': 18, 'CH': 19, 'D': 20, 'DH': 21, 'EH0': 22, 'EH1': 23, 'EH2': 24, 'ER0': 25, 'ER1': 26, 'ER2': 27,
    'EY0': 28, 'EY1': 29, 'EY2': 30, 'F': 31, 'G': 32, 'HH': 33, 'IH0': 34, 'IH1': 35, 'IH2': 36,
    'IY0': 37, 'IY1': 38, 'IY2': 39, 'JH': 40, 'K': 41, 'L': 42, 'M': 43, 'N': 44, 'NG': 45,
    'OW0': 46, 'OW1': 47, 'OW2': 48, 'OY0': 49, 'OY1': 50, 'OY2': 51, 'P': 52, 'R': 53, 'S': 54,
    'SH': 55, 'T': 56, 'TH': 57, 'UH0': 58, 'UH1': 59, 'UH2': 60, 'UW': 61, 'UW0': 62, 'UW1': 63, 'UW2': 64,
    'V': 65, 'W': 66, 'Y': 67, 'Z': 68, 'ZH': 69, 'UNK': 70
}


def tokenize_transcript(cmu_dict, text, max_phonetics):
    # Remove special characters, keeping only letters and whitespaces
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = clean_text.split()
    phonetics = []
    for word in words:
        word = word.lower()
        if word in cmu_dict:
            phonetics.extend(cmu_dict[word][0])
    tokens = []
    for phoenetic in phonetics:
        if phoenetic in vocab.keys():
            tokens.append(vocab[phoenetic])
    tokens = np.array(tokens)[:min(len(tokens), max_phonetics)]
    tokens_padded = np.full(max_phonetics, 70, dtype=int)  # pad with 70 (UNK)
    tokens_padded[:len(tokens)] = tokens
    mask = np.zeros(max_phonetics, dtype=bool)
    mask[:len(tokens)] = 1  # Mark original token positions with 1
    return tokens_padded, mask


def flatten_waveform_tokens(tokens, num_quantizers):
    n_q, B, T = tokens.shape
    transpose_tokens = tokens.transpose(0, 2)
    return transpose_tokens.reshape(B, T * num_quantizers)

def unflatten_tokens(tokens, num_quantizers):
    B, L = tokens.shape
    T = L // num_quantizers
    return tokens.reshape(T, B, num_quantizers).transpose(0, 2)

def normalize_waveform(waveform, sr, speech_tokenizer):
    waveform = waveform.float()
    waveform = waveform.reshape(1, -1)
    waveform = torchaudio.functional.resample(waveform, sr, speech_tokenizer.sample_rate)
    return waveform

def tokenize_waveform(waveform, speech_tokenizer, num_quantizers, device):
    with torch.no_grad():
        codes = speech_tokenizer.encode(waveform.unsqueeze(0).to(device))  # codes: (n_q, B, T)
    semantic_tokens = codes[:num_quantizers, :, :].cpu()
    semantic_tokens = flatten_waveform_tokens(semantic_tokens, num_quantizers)
    return semantic_tokens


def decode_tokens(tokens, speech_tokenizer, num_quantizers):
    unflattened_tokens = unflatten_tokens(tokens, num_quantizers)
    return speech_tokenizer.decode(unflattened_tokens)

def save_to_file(tok, filename, speech_tokenizer, num_quantizers, device):
    tok = jax.device_get(jnp.asarray(tok))
    tok_tensor = torch.from_numpy(tok).to(device)
    outputwav = decode_tokens(tok_tensor.detach(), speech_tokenizer, num_quantizers).cpu().detach().numpy()
    save_waveform(filename, outputwav)

def save_waveform(filename, waveform):
    sf.write(filename, waveform[0, 0], 16000)

def round_up_to_nearest_four(n):
    return ((n + 3) // 4) * 4

def unconditioned_generation(model, params, out_dir, speech_tokenizer, device, audio_length_seconds=5, rng=42, batch_size=10, num_quantizers=4, initial_token=623):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    key = random.PRNGKey(rng)
    tokens = jnp.array([[initial_token]] * batch_size)

    carry = None
    max_audio_tokens = round_up_to_nearest_four(int(200 * audio_length_seconds))  # such that quantization works
    for _ in tqdm(range(max_audio_tokens-1)):
        key, subkey = random.split(key)
        token = tokens[:, -1:]
        carry, logits = model.apply({'params': params}, token, training=False, carry=carry)
        next_token = random.categorical(subkey, logits[:, -1, :], shape=(batch_size,))
        tokens = jnp.concatenate((tokens, next_token[:, None]), axis=1)
    for b, this_tokens in enumerate(tokens):
        this_tokens = this_tokens.reshape(1, -1)
        save_to_file(this_tokens, os.path.join(out_dir, f"generated_{b}.wav"), speech_tokenizer, num_quantizers, device)

def conditioned_generation(text, cmu_dict, model, params, out_dir, speech_tokenizer, device, audio_length_seconds=5, rng=42, batch_size=10, num_quantizers=4, initial_token=623, max_phonetics=100):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    transcript_tokens, this_mask = tokenize_transcript(cmu_dict, text, max_phonetics)
    transcript_tokens = np.tile(transcript_tokens, (batch_size, 1))

    key = random.PRNGKey(rng)
    audio_tokens = jnp.array([[initial_token]] * batch_size)
    carry = None
    encoding = None
    max_audio_tokens = round_up_to_nearest_four(int(200 * audio_length_seconds))  # such that quantization works
    for _ in tqdm(range(max_audio_tokens-1)):
        key, subkey = random.split(key)
        audio_token = audio_tokens[:, -1:]
        encoding, carry, logits = model.apply(
            {'params': params}, audio_token, False, decoder_carry=carry, encoder_input=transcript_tokens, encoding=encoding,
        )
        next_audio_token = random.categorical(subkey, logits[:, -1, :], shape=(batch_size,))
        audio_tokens = jnp.concatenate((audio_tokens, next_audio_token[:, None]), axis=1)

    for b, this_tokens in enumerate(audio_tokens):
        this_tokens = this_tokens.reshape(1, -1)
        save_to_file(this_tokens, os.path.join(out_dir, f"generated_{b}.wav"), speech_tokenizer, num_quantizers, device)

