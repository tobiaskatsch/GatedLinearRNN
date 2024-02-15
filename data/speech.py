import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm

class UnconditionalSpeechDataset(Dataset):
    def __init__(self, data_folder_path):
        file_path = os.path.join(data_folder_path, 'data.npy')
        self.sequences = np.load(os.path.join(file_path), allow_pickle=True)  # (nr_sequences, seq_len)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        inputs = self.sequences[index][:-1]
        targets = self.sequences[index][1:]
        return targets, inputs

class ConditionedSpeechDataset(Dataset):
    def __init__(self, data_folder_path):
        file_path = os.path.join(data_folder_path, 'data.npy')
        self.sequences = np.load(os.path.join(file_path), allow_pickle=True)  # (nr_sequences, seq_len)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        inputs = self.sequences[index][:-1]
        targets = self.sequences[index][1:]
        return targets, inputs

def get_subdirs(directory):
    return [os.path.join(directory, name) for name in os.listdir(directory)]

def preprocess_speech(data_folder_path, speech_tokenizer_path, playlist_url, conditional, snippet_length=10, num_quantizers=4):

    # Source: "2084: MarcRandbot: Speech Synthesis with Mamba" by Lukas Nel.
    # https://2084.substack.com/p/2084-marcrandbot-speech-synthesis

    from speechtokenizer import SpeechTokenizer
    from utils.speech_util import *
    import nltk
    from nltk.corpus import cmudict
    nltk.download('cmudict')
    cmu_dict = cmudict.dict()

    if conditional is True:
        from openai import OpenAI
        api_key = input("Please enter your OpenAI API key: ")
        client = OpenAI(api_key=api_key)
    else:
        client = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(speech_tokenizer_path, "speechtokenizer_hubert_avg", "config.json")
    ckpt_path = os.path.join(speech_tokenizer_path, "speechtokenizer_hubert_avg", "SpeechTokenizer.pt")
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path).to(device)
    speech_tokenizer.eval()

    mp4_path = os.path.join(data_folder_path, "mp4")

    download_audio_from_playlist(playlist_url, mp4_path)

    # Assumed file formats
    file_size_mb = 25  # Target file size in MB
    file_size_bytes = file_size_mb * 1048576  # Convert MB to bytes
    sample_rate_hz = 44100  # Sample rate in Hz
    number_of_channels = 2  # Number of audio channels (stereo)
    bit_depth_bytes = 2  # Bit depth in bytes (16 bits = 2 bytes)
    segment_length = file_size_bytes / (sample_rate_hz * number_of_channels * bit_depth_bytes)  # 148.6s
    segment_length = (segment_length // 10) * 10  # Round to 140s

    print(f"Convert mp4s to wav segments of filesize={file_size_mb} and segment_length={segment_length} {'and transribe' if conditional else ''}")
    # 140s --> roughly 24MB < 25MB (upper limit of whisper)
    segments_path = os.path.join(data_folder_path, "segments")
    for file in tqdm(get_subdirs(mp4_path)):
        process_segments(file, segments_path, segment_length, client=client)

    print(f"Convert wav segments to snippets")
    snippets_path = os.path.join(data_folder_path, "snippets")
    for dir in tqdm(get_subdirs(segments_path)):
        process_snippets(dir, snippets_path, snippet_length)


    """dataset = []
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

    np.save(os.path.join(data_folder_path, 'data.npy'), dataset)"""














