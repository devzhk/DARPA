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


class PointJet1D(Dataset):
    def __init__(self, datapath, tau_invs):
        super(PointJet1D, self).__init__()
        self.tau_invs = tau_invs
        self.datapath = datapath
        self.closure, self.q = self.load(datapath)

    def __getitem__(self, idx):
        return self.q[idx], self.closure[idx]

    def __len__(self):
        return self.q.shape[0]

    def load(self, path):
        
        closure_list = []
        q_list = []
        for tau_inv in self.tau_invs:
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


class PointJet(Dataset):
    def __init__(self, data_dir):
        super(PointJet, self).__init__()
        # closure_mean, q_mean, yy, dq_dy_mean, mu_f = preprocess(N_y, beta, tau_inv)
        closure_new, q_new, dq_dy_new = load_data(data_dir)
        self.X = torch.tensor(q_new, dtype=torch.float32)
        self.Y = torch.tensor(closure_new, dtype=torch.float32)

    def __getitem__(self, item):
        return self.X[:, item], self.Y[:, item]

    def __len__(self):
        return self.X.shape[0]


