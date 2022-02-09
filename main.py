# %%
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models import FNO1d
from utils.datasets import PointJet1D
# %%
closure_path = 'data/data_closure.mat'
omega_path = 'data/data_w.mat'
dataset = PointJet1D(closure_path=closure_path, omega_path=omega_path)
