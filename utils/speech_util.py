import re

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


def tokenize_transcript(cmu_dict, text):
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
    return tokens

