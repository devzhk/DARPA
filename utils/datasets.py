import scipy.io
import torch
from torch.utils.data import Dataset
from utils.helper import preprocess

'''
Dataset class for PointJet1D data. 

'''


class PointJet1D(Dataset):
    def __init__(self, closure_path, omega_path, num_chop=15):
        super(PointJet1D, self).__init__()
        closure = self.loadmat(closure_path, key='data_closure')
        self.closure = closure[num_chop+1:-num_chop, :]
        omega = self.loadmat(omega_path, key='data_w')
        self.omega = omega[num_chop+1:-num_chop, :]

    def __getitem__(self, idx):
        return self.omega[:, idx], self.closure[:, idx]

    def __len__(self):
        return self.omega.shape[1]

    @staticmethod
    def loadmat(path, key):
        raw = scipy.io.loadmat(path)[key]
        data = torch.tensor(raw.squeeze(axis=0), dtype=torch.float32)
        return data


class PointJet(Dataset):
    def __init__(self, N_y, beta, tau_inv):
        super(PointJet, self).__init__()
        closure_mean, q_mean, yy, dq_dy_mean, mu_f = preprocess(N_y, beta, tau_inv)
        self.X = torch.tensor(q_mean, dtype=torch.float32)
        self.Y = torch.tensor(mu_f, dtype=torch.float32)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.X.shape[0]
