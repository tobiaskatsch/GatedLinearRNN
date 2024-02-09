import pickle
from collections.abc import Iterable
import numpy as np


class Vocab:
    def __init__(self, idx_to_token, add_unk=True, add_pad=True):
        self.idx_to_token = idx_to_token
        self.add_unk = add_unk
        self.add_pad = add_pad
        if add_unk:
            self.idx_to_token.append(self.unk_token)
        if add_pad:
            self.idx_to_token.append(self.pad_token)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def token2idx(self, tokens):
        if np.isscalar(tokens):
            if self.add_unk:
                return self.token_to_idx.get(tokens, self.unk_idx)
            else:
                return self.token_to_idx[tokens]
        elif isinstance(tokens, Iterable):
            if self.add_unk:
                return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
            else:
                return [self.token2idx(token) for token in tokens]
        else:
            raise TypeError("Input should be a scalar or an iterable")

    def idx2token(self, idxs):
        if np.isscalar(idxs):
            return self.idx_to_token[idxs]
        elif isinstance(idxs, Iterable):
            return [self.idx2token(idx) for idx in idxs]
        else:
            raise TypeError("Input should be a scalar or an iterable")

    @property
    def pad_token(self):
        return "<pad>"

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_idx(self):
        return self.token_to_idx[self.pad_token]

    @property
    def unk_idx(self):
        return self.token_to_idx[self.unk_token]

    def save(self, vocab_path):
        vocab_dict = {
            "idx_to_token": self.idx_to_token,
            "token_to_idx": self.token_to_idx,
            "add_unk": self.add_unk,
            "add_pad": self.add_pad
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_dict, f)

    @classmethod
    def load(cls, vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab_dict = pickle.load(f)
        vocab = cls([], add_unk=False, add_pad=False)
        vocab.idx_to_token = vocab_dict["idx_to_token"]
        vocab.token_to_idx = vocab_dict["token_to_idx"]
        vocab.add_unk = vocab_dict["add_unk"]
        vocab.add_pad = vocab_dict["add_pad"]
        return vocab







