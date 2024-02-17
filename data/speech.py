import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
from utils.speech_util import tokenize_transcript, normalize_waveform, tokenize_waveform


class UnconditionedSpeechDataset(Dataset):
    def __init__(self, data_folder_path):
        file_path = os.path.join(data_folder_path, 'speech_tokens.npy')
        self.sequences = np.load(file_path, allow_pickle=True)  # (nr_sequences, seq_len)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        inputs = self.sequences[index][:-1]
        targets = self.sequences[index][1:]
        return targets, inputs

class ConditionedSpeechDataset(Dataset):
    def __init__(self, data_folder_path):
        speech_tokens_path = os.path.join(data_folder_path, 'speech_tokens.npy')
        self.speech_tokens_sequences = np.load(speech_tokens_path, allow_pickle=True)

        text_tokens_path = os.path.join(data_folder_path, 'text_tokens.npy')
        text_targets_path = os.path.join(data_folder_path, 'text_targets.npy')
        self.text_tokens_sequences = np.load(text_tokens_path, allow_pickle=True)
        self.text_targets_sequences = np.load(text_targets_path, allow_pickle=True)

    def __len__(self):
            return len(self.text_tokens_sequences)

    def __getitem__(self, index):
        speech_targets = self.speech_tokens_sequences[index][1:]
        speech_tokens = self.speech_tokens_sequences[index][:-1]

        text_targets = self.text_tokens_sequences[index][:-1]
        text_tokens = self.text_tokens_sequences[index][:-1]

        return speech_targets, speech_tokens, text_targets, text_tokens

def get_subdirs(directory):
    return [os.path.join(directory, name) for name in os.listdir(directory)]

def preprocess_speech(data_folder_path, speech_tokenizer, device, playlist_url, cmu_dict, conditioned=True, snippet_length=10, num_quantizers=4, max_phonetics=100):

    # Inspired by: "2084: MarcRandbot: Speech Synthesis with Mamba" by Lukas Nel.
    # https://2084.substack.com/p/2084-marcrandbot-speech-synthesis

    from pytube import Playlist
    import os
    import torch
    import shutil
    from moviepy.editor import AudioFileClip
    import json
    from scipy.io import wavfile
    from scipy.ndimage import maximum_filter1d

    def download_audio_from_playlist(playlist_url, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        playlist = Playlist(playlist_url)
        for i, video in enumerate(playlist.videos):
            filename = f"video_{i}.mp4"
            full_output_path = os.path.join(output_path, filename)
            if os.path.exists(full_output_path):
                continue
            audio_stream = video.streams.get_audio_only()
            audio_stream.download(output_path=output_path, filename=filename)

    def transcibe(client, audio_file, model="whisper-1"):
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        return transcript.words

    def process_speech_array(x, detection_threshold=0.01, pooling_seconds=0.2, fps=44100, start_offset_seconds=0.5):
        """
        Process a speech audio array to find and extract a segment where the speech is detected.

        The function applies max pooling to the absolute value of the audio signal to detect areas
        where the amplitude exceeds a specified threshold. It then extracts the segment of audio
        starting from the first rise above the threshold to the last fall below it. This segment is
        placed in a zero array with a specified offset from the start.
        """
        x_abs = np.abs(x)
        max_pooled = maximum_filter1d(x_abs, size=int(pooling_seconds * fps), mode='reflect')
        detected_binary = np.where(max_pooled > detection_threshold, 1, 0)

        # Find the first index where detected_binary jumps from 0 to 1
        changes = np.diff(detected_binary)
        rising_edges = np.where(changes == 1)[0]
        if len(rising_edges) == 0:
            raise ValueError
        min_idx = rising_edges[0]

        # Find the last index where detected_binary jumps from 1 to 0
        falling_edges = np.where(changes == -1)[0]
        if len(falling_edges) == 0:
            max_idx = len(x) - 1  # If no falling edge, use the end of the signal
        else:
            max_idx = falling_edges[-1]

        # Initialize y as a zeros-like array of x
        y = np.zeros_like(x)
        start_offset_idx = int(start_offset_seconds * fps)
        end_idx = start_offset_idx + (max_idx - min_idx)

        y[start_offset_idx:end_idx] = x[min_idx:max_idx]
        return y, min_idx, max_idx


    def process_segments(mp4_file, output_path, segment_length, client=None):

        audio = AudioFileClip(mp4_file)
        duration = audio.duration  # seconds
        duration = int(duration) - 1

        # Split the audio into 10-second clips and save
        output_name = os.path.basename(mp4_file).split(".")[0]

        for start in tqdm(range(0, duration, segment_length)):
            this_output_dir = os.path.join(output_path, f'{output_name}_segment_{start}_{start + segment_length}')
            if not os.path.exists(this_output_dir):
                os.makedirs(this_output_dir)

            audio_path = os.path.join(this_output_dir, "audio.wav")
            if not os.path.exists(audio_path):
                end = min(start + segment_length, duration)
                clip = audio.subclip(start, end)
                clip_array = clip.to_soundarray()  # (44100 * segment_legnth, 2)
                clip_array = np.mean(clip_array, axis=1)  # to mono
                try:
                    clip_array, _, _ = process_speech_array(clip_array)  # remove mid-sentence breaks
                except ValueError:
                    shutil.rmtree(this_output_dir)
                    continue
                clip_array_int16 = np.int16(clip_array * 32767)
                wavfile.write(audio_path, 44100, clip_array_int16)

            if client is not None:
                transcript_path = os.path.join(this_output_dir, "transcript.json")
                if not os.path.exists(transcript_path):
                    audio_file = open(audio_path, "rb")
                    words = transcibe(client, audio_file)
                    with open(transcript_path, 'w') as json_file:
                        json.dump(words, json_file)

    def get_within(this_segment_transcript, min, max):
        # return transcript dict containing all words starting within range [min, max]
        return_list = []
        for word in this_segment_transcript:
            if min <= word["start"] and word["start"] <= max:
                return_list.append(word)
        return return_list

    def transcript_to_txt(this_transcript):
        return ' '.join([element["word"] for element in this_transcript])

    def process_snippets(segment_dir, snippets_dir, snippet_length, avg_word_length_seconds=0.5):
        segment_name = os.path.basename(segment_dir)
        segment_audio_path = os.path.join(segment_dir, "audio.wav")

        sr, segment_array = wavfile.read(segment_audio_path)
        segment_array = segment_array / 32768.0
        segment_length = int(len(segment_array) / sr)
        n = segment_length // snippet_length

        segment_transcript_path = os.path.join(segment_dir, "transcript.json")
        if os.path.exists(segment_transcript_path):
            with open(segment_transcript_path, 'r') as json_file:
                segment_transcript = json.load(json_file)

        for snippet_id in range(n):

            start_seconds = snippet_id * snippet_length
            end_seconds = start_seconds + snippet_length

            snippet_path = os.path.join(snippets_dir, f'{segment_name}_snippet_{start_seconds}_{end_seconds}')

            if not os.path.exists(snippet_path):
                os.makedirs(snippet_path)

            snippet_audio_path = os.path.join(snippet_path, "audio.wav")
            snippet_transcript_path = os.path.join(snippet_path, "transcript.json")

            if not os.path.exists(snippet_audio_path) or not os.path.exists(snippet_transcript_path):
                snippet_array = segment_array[start_seconds * sr:end_seconds * sr]
                snippet_transcript = get_within(segment_transcript, start_seconds, end_seconds)
                try:
                    snippet_array, min_idx, max_idx = process_speech_array(snippet_array)
                except ValueError:
                    shutil.rmtree(snippet_path)
                    continue
                min_seconds, max_seconds = min_idx/sr, max_idx/sr
                # substract avg_word_length_seconds because "start" counts from the full detection and not from the start of speech
                snippet_transcript = get_within(snippet_transcript, min_seconds + start_seconds - avg_word_length_seconds,
                                                max_seconds + start_seconds)

                new_snippet_transcript = []
                for word in snippet_transcript:
                    new_word = word.copy()
                    new_word["start"] = new_word["start"] - start_seconds
                    new_word["end"] = new_word["end"] - start_seconds
                    new_snippet_transcript.append(new_word)
                snippet_transcript = new_snippet_transcript

                clip_array_int16 = np.int16(snippet_array * 32767)
                wavfile.write(snippet_audio_path, 44100, clip_array_int16)

                with open(snippet_transcript_path, 'w') as json_file:
                    json.dump(snippet_transcript, json_file)


    if conditioned is True:
        from openai import OpenAI
        api_key = input("Please enter your OpenAI API key: ")
        client = OpenAI(api_key=api_key)
    else:
        client = None

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

    print(f"Convert mp4s to wav segments of filesize={file_size_mb} and segment_length={segment_length} {'and transcribe' if conditioned else ''}")
    # 140s --> roughly 24MB < 25MB (upper limit of whisper)
    segments_path = os.path.join(data_folder_path, "segments")
    for file in tqdm(get_subdirs(mp4_path)):
        process_segments(file, segments_path, segment_length, client=client)

    print(f"Convert wav segments to snippets")
    snippets_path = os.path.join(data_folder_path, "snippets")
    for dir in tqdm(get_subdirs(segments_path)):
        process_snippets(dir, snippets_path, snippet_length)

    speech_tokens_path = os.path.join(data_folder_path, "speech_tokens.npy")
    if not os.path.exists(speech_tokens_path):
        print(f"Tokenize audio (1024 tokens)")
        speech_tokens = []
        for dir in tqdm(get_subdirs(snippets_path)):
            sample_rate, x = wavfile.read(os.path.join(dir, "audio.wav"))
            x = torch.from_numpy(x)
            x = x / 32768.0
            x = normalize_waveform(x, sample_rate, speech_tokenizer)
            # Only take snippets of full length
            if x.shape[1] != speech_tokenizer.sample_rate * snippet_length:
                print("Issue here")
                continue
            x = tokenize_waveform(x, speech_tokenizer, num_quantizers, device)
            x = x.cpu().numpy()
            x = x.reshape(-1)
            speech_tokens.append(x)
        np.save(speech_tokens_path, speech_tokens)

    data_shape = np.load(speech_tokens_path, allow_pickle=True).shape
    seq_length = data_shape[1]

    text_tokens_path = os.path.join(data_folder_path, "text_tokens.npy")
    text_targets_path = os.path.join(data_folder_path, "text_targets.npy")

    text_tokens = np.full(data_shape, 71, dtype=int)
    text_targets = np.full(data_shape, 71, dtype=int)

    def sec_to_idx(time_s, snippet_length, seq_length):
        return int(min(((seq_length / snippet_length) * time_s), seq_length))

    if (not os.path.exists(text_tokens_path) or not os.path.exists(text_targets_path)) and conditioned is True:
        print(f"Tokenize transcript (71 tokens)")
        for data_idx, dir in enumerate(tqdm(get_subdirs(snippets_path))):
            with open(os.path.join(dir, "transcript.json")) as json_file:
                segment_transcript = json.load(json_file)
            for word_idx, this_dict in enumerate(segment_transcript):
                word_start_idx = sec_to_idx(this_dict["start"], snippet_length, seq_length)
                word_end_idx = sec_to_idx(this_dict["end"], snippet_length, seq_length)
                tokens = tokenize_transcript(cmu_dict, this_dict["word"])
                indices = np.linspace(word_start_idx, word_end_idx, len(tokens)+1, dtype=int)
                for token_idx, token in enumerate(tokens):
                    text_tokens[data_idx, word_idx + token_idx] = token
                    text_targets[data_idx, (word_start_idx + indices[token_idx]):(word_start_idx + indices[token_idx+1])] = token

        np.save(text_tokens_path, text_tokens)
        np.save(text_targets_path, text_targets)



















