# %%
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from utils.cadam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from models import FNO1d
from utils.datasets import PointJet1D
# %%
closure_path = 'data/data_closure.mat'
omega_path = 'data/data_w.mat'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
layers = [32, 32]
modes1 = [12, 12]
fc_dim = 32
batchsize = 128
num_epochs = 1200
num_pad = 5
# %%


dataset = PointJet1D(closure_path=closure_path, omega_path=omega_path)
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
pbar = tqdm(range(num_epochs), dynamic_ncols=True)

model = FNO1d(modes1=modes1, fc_dim=fc_dim, layers=layers, in_dim=1, activation='gelu').to(device)

optimizer = SGD(model.parameters(), lr=5e-3, momentum=0.5)

scheduler = MultiStepLR(optimizer, milestones=[200, 400, 600, 800, 1000], gamma=0.5)
criterion = nn.MSELoss()
# criterion = LpLoss()

Ny = dataset[0][0].shape[0]
beta = 1.0
yy = np.linspace(-2*np.pi, 2*np.pi, Ny)
yy = torch.tensor(yy, dtype=torch.float32, device=device).unsqueeze(dim=0)
# %%
for e in pbar:
    train_loss = 0
    for omega, closure in train_loader:
        omega = omega.to(device)

        closure = closure.to(device)
        optimizer.zero_grad()

        omega_pad = F.pad(omega+yy, (0, num_pad), 'constant', 0)
        pred_pad = model(omega_pad.unsqueeze(-1))
        pred = pred_pad.squeeze(-1)[..., :-num_pad]
        loss = criterion(pred, closure)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    scheduler.step()
    avg_loss = train_loss / len(train_loader)
    pbar.set_description(
        (
            f'Epoch {e}, MSE loss: {avg_loss}'
        )
    )

torch.save(model.state_dict(), f'ckpts/fno1d-pad{num_pad}.pt')
