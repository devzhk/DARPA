from tqdm import tqdm
import torch
import torch.nn as nn

from utils.cadam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import FNO1d
from utils.datasets import PointJet


def train(dataloader, model, optimizer, scheduler, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # hyperparameters
    num_pad = 5
    num_epoch = 2000
    # training
    pbar = tqdm(range(num_epoch), dynamic_ncols=True, smoothing=0.1)
    criterion = nn.MSELoss()

    for e in pbar:
        train_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x.unsqueeze(-1))
            loss = criterion(pred.squeeze(-1), y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        scheduler.step()
        avg_loss = train_loss / len(dataloader)
        pbar.set_description(
            (
                f'Epoch {e}: avg loss : {avg_loss}'
            )
        )
            # q_pad = F.pad(x, (0, num_pad), 'constant', 0)
            # pred_pad = model(q_pad)
    torch.save(model.state_dict(), f'ckpts/fno1d-nopad-final.pt')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # training parameter
    batchsize = 1

    N_y = 384
    beta = 1.0
    tau_inv = [0.01, 0.02, 0.04, 0.08, 0.16]
    dataset = PointJet(N_y, beta, tau_inv)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    layers = [32, 32]
    modes1 = [12, 12]
    fc_dim = 32
    model = FNO1d(modes1=modes1, layers=layers,
                  fc_dim=fc_dim, in_dim=1, activation='tanh').to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[300, 600, 900, 1200, 1500], gamma=0.5)

    train(dataloader, model, optimizer, scheduler, device=device)

