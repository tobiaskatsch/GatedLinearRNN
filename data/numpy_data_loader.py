import torch
from torch.utils.data import DataLoader

class NumpyDataLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            yield self._to_numpy(batch)

    @staticmethod
    def _to_numpy(batch):
        if isinstance(batch, torch.Tensor):
            return batch.cpu().numpy()
        elif isinstance(batch, list):
            return [NumpyDataLoader._to_numpy(b) for b in batch]
        elif isinstance(batch, tuple):
            return tuple(NumpyDataLoader._to_numpy(b) for b in batch)
        elif isinstance(batch, dict):
            return {k: NumpyDataLoader._to_numpy(b) for k, b in batch.items()}
        else:
            return batch
