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
    import torchaudio
    from pytube import Playlist
    import os
    import torch
    from moviepy.editor import AudioFileClip
    import json
    from pydub import AudioSegment
    import nltk
    from nltk.corpus import cmudict
    nltk.download('cmudict')
    cmu_dict = cmudict.dict()

    def download_audio_from_playlist(playlist_url, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        playlist = Playlist(playlist_url)
        for i, video in enumerate(playlist.videos):
            audio_stream = video.streams.get_audio_only()
            audio_stream.download(output_path=output_path, filename=f"video_{i}.mp4")

    def transcibe(client, audio_file, model="whisper-1"):
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        return transcript.words

    def snippify_transcript(transcript, snippet_length):
        snippets = []
        current_snippet_words = []
        snippet_start_time = transcript[0]['start']
        for word in transcript:
            if word['start'] - snippet_start_time <= snippet_length:
                current_snippet_words.append(word['word'])
            else:
                snippets.append(' '.join(current_snippet_words))
                current_snippet_words = [word['word']]
                snippet_start_time = word['start']
        if current_snippet_words:
            snippets.append(' '.join(current_snippet_words))
        return snippets

    def process_segments(mp4_file, output_path, segment_length, client=None):

        audio = AudioFileClip(mp4_file)
        duration = int(audio.duration)

        # Split the audio into 10-second clips and save
        output_name = os.path.basename(mp4_file)
        for start in range(0, duration, segment_length):
            this_output_dir = os.path.join(output_path, f'{output_name}_segment_{start}_{start + segment_length}')
            if not os.path.exists(this_output_dir):
                os.makedirs(this_output_dir)

            end = min(start + segment_length, duration)
            clip = audio.subclip(start, end)

            audio_path = os.path.join(this_output_dir, "audio.wav")
            if not os.path.exists(audio_path):
                clip.write_audiofile(audio_path, verbose=False, logger=None)

            if client is not None:
                transcript_path = os.path.join(this_output_dir, "transcript.json")
                if not os.path.exists(transcript_path):
                    audio_file = open(audio_path, "rb")
                    words = transcibe(client, audio_file)
                    with open(transcript_path, 'w') as json_file:
                        json.dump(words, json_file)

    def process_snippets(segment_dir, snippets_dir, snippet_length):
        segment_name = os.path.basename(segment_dir)
        segment_audio_path = os.path.join(segment_dir, "audio.wav")
        segment_audio = AudioSegment.from_wav(segment_audio_path)
        segment_length_ms = len(segment_audio)
        snippet_length_ms = snippet_length * 1000

        segment_transcript_path = os.path.join(segment_dir, "transcript.json")
        if os.path.exists(segment_transcript_path):
            with open(segment_transcript_path, 'r') as json_file:
                segment_transcript = json.load(json_file)
            transcript_snippets = snippify_transcript(segment_transcript, snippet_length)
        else:
            transcript_snippets = None

        for snippet_id, start_ms in enumerate(range(0, segment_length_ms, snippet_length_ms)):
            end_ms = start_ms + snippet_length_ms
            snippet_path = os.path.join(snippets_dir, f'{segment_name}_snippet_{start_ms}_{end_ms}')
            if not os.path.exists(snippet_path):
                os.makedirs(snippet_path)

            snippet_audio = segment_audio[start_ms:end_ms]
            snippet_audio_path = os.path.join(snippet_path, "audio.wav")
            if not os.path.exists(snippet_audio_path):
                snippet_audio.export(snippet_audio_path, format="wav")

            snippet_transcript_path = os.path.join(snippet_path, "transcript.txt")
            if transcript_snippets is not None and not os.path.exists(snippet_transcript_path):
                transcript_snippet = transcript_snippets[snippet_id]
                with open(snippet_transcript_path, 'w') as file:
                    file.write(transcript_snippet)

    def flatten_waveform_tokens(tokens, num_quantizers):
        n_q, B, T = tokens.shape
        transpose_tokens = tokens.transpose(0, 2)
        return transpose_tokens.reshape(B, T * num_quantizers)

    def normalize_waveform(waveform, sr, speech_tokenizer):
        waveform = waveform.float()
        waveform = torch.mean(waveform, dim=1, keepdim=True)
        waveform = waveform.reshape(1, -1)
        waveform = torchaudio.functional.resample(waveform, sr, speech_tokenizer.sample_rate)
        return waveform

    def tokenize_waveform(waveform, speech_tokenizer, num_quantizers, device):
        with torch.no_grad():
            codes = speech_tokenizer.encode(waveform.unsqueeze(0).to(device))  # codes: (n_q, B, T)
        semantic_tokens = codes[:num_quantizers, :, :].cpu()
        semantic_tokens = flatten_waveform_tokens(semantic_tokens, num_quantizers)
        return semantic_tokens


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
    segment_length = int((segment_length // 10) * 10)  # Round to 140s

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














