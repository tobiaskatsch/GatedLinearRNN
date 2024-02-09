import torch
from torch.utils.data import Dataset
import os
import numpy as np
from transformers import GPT2Tokenizer
from tqdm import tqdm

class SpeechDataset(Dataset):
    def __init__(self, data_folder_path):
        file_path = os.path.join(data_folder_path, 'data.npy')
        self.sequences = np.load(os.path.join(file_path), allow_pickle=True)  # (nr_sequences, seq_len)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        inputs = self.sequences[index][:-1]
        targets = self.sequences[index][1:]
        return targets, inputs



def preprocess_speech(data_folder_path, speech_tokenizer_path, playlist_url, snippet_length=10, num_quantizers=4):

    # Source: "2084: MarcRandbot: Speech Synthesis with Mamba" by Lukas Nel.
    # https://2084.substack.com/p/2084-marcrandbot-speech-synthesis

    from speechtokenizer import SpeechTokenizer
    import soundfile as sf
    import torchaudio
    from pytube import Playlist
    from moviepy.editor import AudioFileClip
    from datasets import load_dataset
    from scipy.io import wavfile

    def download_audio_from_playlist(playlist_url, output_path):
        playlist = Playlist(playlist_url)
        for video in playlist.videos:
            audio_stream = video.streams.get_audio_only()
            audio_stream.download(output_path=output_path, filename=video.title + ".mp4")

    def convert_mp4_to_wav_clips(mp4_file, output_dir, snippet_length):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert mp4 to wav and split into 10-second clips
        # Load the audio from the video file
        audio = AudioFileClip(mp4_file)

        # Duration of the audio in seconds
        duration = int(audio.duration)

        # Split the audio into 10-second clips and save
        mp4_output_name = os.path.basename(mp4_file).split('.')[0].replace(' ', '_').replace("#", "num")
        print(mp4_output_name)
        for start in range(0, duration, snippet_length):
            outputpath = os.path.join(output_dir, f'{mp4_output_name}_clip_{start}_{start + snippet_length}.wav')
            if os.path.exists(outputpath):
                continue
            end = min(start + snippet_length, duration)
            clip = audio.subclip(start, end)
            clip.write_audiofile(outputpath, verbose=False, logger=None)

    def get_files(directory):
        return [os.path.join(directory, file) for file in os.listdir(directory)]

    def save_to_file(tok, filename, speech_tokenizer, device):
        outputwav = decode_tokens(tok.detach().to(device), speech_tokenizer)
        save_waveform(filename, outputwav)

    def save_waveform(filename, waveform):
        torchaudio.save(filename, waveform[0].detach().cpu(), 16000)

    def decode_tokens(tokens, speech_tokenizer):
        unflattened_tokens = unflatten_tokens(tokens)
        return speech_tokenizer.decode(unflattened_tokens)

    def flatten_tokens(tokens, num_quantizers):
        n_q, B, T = tokens.shape
        transpose_tokens = tokens.transpose(0, 2)
        return transpose_tokens.reshape(B, T * num_quantizers)

    def unflatten_tokens(tokens, num_quantizers):
        B, L = tokens.shape
        T = L // num_quantizers
        return tokens.reshape(T, B, num_quantizers).transpose(0, 2)

    def normalize_waveform(waveform, sr, speech_tokenizer):
        waveform = waveform.float()
        waveform = torch.mean(waveform, dim=1, keepdim=True)
        waveform = waveform.reshape(1, -1)
        waveform = torchaudio.functional.resample(waveform, sr, speech_tokenizer.sample_rate)
        return waveform

    def tokenize(waveform, speech_tokenizer, num_quantizers, device):
        with torch.no_grad():
            codes = speech_tokenizer.encode(waveform.unsqueeze(0).to(device))  # codes: (n_q, B, T)
        semantic_tokens = codes[:num_quantizers, :, :].cpu()
        semantic_tokens = flatten_tokens(semantic_tokens, num_quantizers)
        return semantic_tokens


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(speech_tokenizer_path, "speechtokenizer_hubert_avg", "config.json")
    ckpt_path = os.path.join(speech_tokenizer_path, "speechtokenizer_hubert_avg", "SpeechTokenizer.pt")

    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
    speech_tokenizer.eval()


    mp4_folder = os.path.join(data_folder_path, "mp4")
    wav_folder = os.path.join(data_folder_path, "wav")
    download_audio_from_playlist(playlist_url, mp4_folder)


    print("Convert mp4 to wav")
    for file in tqdm(get_files(mp4_folder)):
        convert_mp4_to_wav_clips(file, wav_folder, snippet_length)


    dataset = []
    print("Normalize and Tokenize")
    for file in tqdm(get_files(wav_folder)):
        # load wav file as numpy array
        sample_rate, x = wavfile.read(file)
        x = torch.from_numpy(x)
        x = x / 32768.0
        x = normalize_waveform(x, sample_rate, speech_tokenizer)
        # Only take snippets of full length
        if x.shape[1] != speech_tokenizer.sample_rate * snippet_length:
            continue
        x = tokenize(x, speech_tokenizer, num_quantizers, device)
        x = x.cpu().numpy()
        x = x.reshape(-1)
        dataset.append(x)

    dataset = np.array(dataset)

    np.save(os.path.join(data_folder_path, 'data.npy'), dataset)














