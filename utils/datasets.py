import scipy.io
import torch
from torch.utils.data import Dataset

'''
Dataset class for PointJet1D data. 

'''


class PointJet1D(Dataset):
    def __init__(self, closure_path, omega_path, num_chop=15):
        super(PointJet1D, self).__init__()
        closure = self.loadmat(closure_path, key='data_closure')
        self.closure = closure[num_chop:-num_chop, :]
        omega = self.loadmat(omega_path, key='data_w')
        self.omega = omega[num_chop:-num_chop, :]

    def __getitem__(self, idx):
        return self.omega[:, idx], self.closure[:, idx]

    def __len__(self):
        return self.omega.shape[1]

    @staticmethod
    def loadmat(path, key):
        raw = scipy.io.loadmat(path)[key]
        data = torch.tensor(raw.squeeze(axis=0), dtype=torch.float32)
        return data
