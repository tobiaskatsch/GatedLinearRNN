import torchaudio
from pytube import Playlist
import os
import torch
from moviepy.editor import AudioFileClip
import json
from pydub import AudioSegment


def text_to_phonetic_tokens(cmu_dict, text):
    words = text.split()
    phonetic_tokens = []
    for word in words:
        word = word.lower()
        if word in cmu_dict:
            phonetic_tokens.extend(cmu_dict[word][0])
        else:
            phonetic_tokens.append('UNK')
    return phonetic_tokens

def download_audio_from_playlist(playlist_url, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    playlist = Playlist(playlist_url)
    for video in playlist.videos:
        audio_stream = video.streams.get_audio_only()
        audio_stream.download(output_path=output_path, filename=video.title + ".mp4")


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
    output_name = os.path.basename(mp4_file).split('.')[0].replace(' ', '_').replace("#", "num")
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


def flatten_tokens(tokens, num_quantizers):
    n_q, B, T = tokens.shape
    transpose_tokens = tokens.transpose(0, 2)
    return transpose_tokens.reshape(B, T * num_quantizers)


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