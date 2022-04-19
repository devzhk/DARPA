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
#%%
def get_init(q, yy):
    '''
    add positional encoding to q
    '''    
    yy_rep = yy.repeat(q.shape[0], 1).unsqueeze(-1)
    return torch.cat([q, yy_rep], dim=-1)



# %%
data_dir = 'data'
taus = [0.01, 0.02, 0.04, 0.08, 0.16, 0.2]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
layers = [8, 8, 8]
modes1 = [8, 8]
fc_dim = 8

batchsize = 20
num_epochs = 2000
# num_pad = 5
# %%


dataset = PointJet1D(datapath=data_dir, tau_invs=taus)
train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
pbar = tqdm(range(num_epochs), dynamic_ncols=True)

model = FNO1d(modes1=modes1, fc_dim=fc_dim, layers=layers, in_dim=2, activation='gelu').to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
criterion = nn.MSELoss()
# criterion = LpLoss()

Ny = dataset[0][0].shape[0]
beta = 1.0
L = 4 * np.pi
yy = np.linspace(-L / 2.0, L / 2.0, Ny)
yy = torch.tensor(yy, dtype=torch.float32, device=device)
# %%
for e in pbar:
    train_loss = 0
    for q, closure in train_loader:
        q = q.to(device).unsqueeze(-1)
        closure = closure.to(device).unsqueeze(-1)
        in_q = get_init(q, yy)

        optimizer.zero_grad()

        # omega_pad = F.pad(, (0, num_pad), 'constant', 0)
        pred = model(in_q)
        # pred = pred_pad.squeeze(-1)[..., :-num_pad]
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

torch.save(model.state_dict(), f'ckpts/fno1d.pt')
