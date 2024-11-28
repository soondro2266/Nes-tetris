import torch

class TetrisDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, device, dtype = torch.float32):
        self._X = torch.tensor(X, device = device, dtype = dtype)
        self._Y = torch.tensor(Y, device = device, dtype = dtype)
    def __len__(self):
        return len(self._X)
    def __getitem__(self, idx):
        return self._X[idx], self._Y[idx]