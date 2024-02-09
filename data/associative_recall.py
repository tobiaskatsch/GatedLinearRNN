import torch
from data.vocab import Vocab
from torch.utils.data import Dataset
import random
import numpy as np

def recall(seq, query):
    for i in range(len(seq)-2, -1, -1):
        if seq[i] == query:
            return seq[i+1]
    return "<unk>"

class AssociativeRecallDataset(Dataset):
    """
    Custom Associative Recall variant where queries and targets are drawn from the same vocab.
    """

    def __init__(self, vocab_size, seq_length, n_samples, seed):
        tokens = [str(i) for i in range(vocab_size)]
        self.vocab = Vocab(idx_to_token=tokens)
        self.tokens = tokens
        self.seq_length = seq_length
        self.n_samples = n_samples
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        random.seed(self.seed + index)  # Ensure different seeds for each sample
        inputs = random.choices(self.tokens, k=self.seq_length)
        targets = []
        for i, _ in enumerate(inputs):
            if i == 0:
                targets.append("<unk>")
            else:
                current_seq = inputs[:i+1]
                query = current_seq[-1]
                targets.append(recall(seq=current_seq, query=query))
        inputs = np.array(self.vocab.token2idx(inputs))
        targets = np.array(self.vocab.token2idx(targets))
        return targets, inputs





