
def tokenize_transcript(cmu_dict, text):
    words = text.split()
    phonetic_tokens = []
    for word in words:
        word = word.lower()
        if word in cmu_dict:
            phonetic_tokens.extend(cmu_dict[word][0])
        else:
            phonetic_tokens.append('UNK')
    return phonetic_tokens

