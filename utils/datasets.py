import os

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from .helper import preprocess, load_data
from .data import load_avg

'''
Dataset class for PointJet1D data. 

'''


def _load(path, tau_invs):
    closure_list = []
    q_list = []
    for tau_inv in tau_invs:
        sub_dir = f'beta_1.0_Gamma_1.0_relax_{tau_inv}'
        cls_path = os.path.join(path, sub_dir, 'data_closure_cons.mat')
        q_path = os.path.join(path, sub_dir, 'data_q.mat')

        closure_avg = load_avg(cls_path, key='data_closure_cons')
        q_avg = load_avg(q_path, key='data_q')

        closure_list.append(closure_avg / tau_inv)
        q_list.append(q_avg)
    closure_data = np.concatenate(closure_list, axis=1)
    q_data = np.concatenate(q_list, axis=1)
    return torch.tensor(closure_data.T, dtype=torch.float32), torch.tensor(q_data.T, dtype=torch.float32)


def _interpolate(x):
    x_c = (x[..., 0:-1] + x[..., 1:]) / 2
    return x_c

def _gradient_first(x, dy):
    x_grad = (x[..., 1:] - x[..., 0:-1]) / dy
    return x_grad


class PointJet1D(Dataset):
    def __init__(self, datapath, tau_invs):
        super(PointJet1D, self).__init__()
        self.tau_invs = tau_invs
        self.datapath = datapath
        self.closure, self.q = _load(datapath, tau_invs)

    def __getitem__(self, idx):
        return self.q[idx], self.closure[idx]

    def __len__(self):
        return self.q.shape[0]


class PointJetdy(Dataset):
    def __init__(self, datapath, tau_invs, dy):
        super(PointJetdy, self).__init__()
        self.tau_invs = tau_invs
        self.datapath = datapath
        self.dy = dy
        self.prepare()

    def prepare(self):
        closure, q = _load(self.datapath, self.tau_invs)
        q_c = _interpolate(q)
        cls_dy = _gradient_first(closure, self.dy)
        self.q_c = q_c
        self.closure_dy = cls_dy

    def __getitem__(self, idx):
        return self.q_c[idx], self.closure_dy[idx]

    def __len__(self):
        return self.q_c.shape[0]


class PointJetdq(Dataset):
    def __init__(self, data_dir, tau_invs):
        super(PointJetdq, self).__init__()
        closure_mean, q_mean, yy, dq_dy_mean, mu_f = preprocess(1, tau_invs)
        # closure_new, q_new, dq_dy_new = load_data(data_dir)
        q = torch.tensor(q_mean, dtype=torch.float32)
        dq_dy = torch.tensor(dq_dy_mean, dtype=torch.float32)
        self.feature = torch.stack([q, dq_dy], dim=-1)
        self.mu_f = torch.tensor(mu_f, dtype=torch.float32)

    def __getitem__(self, item):
        return self.feature[item], self.mu_f[item]

    def __len__(self):
        return self.feature.shape[0]
