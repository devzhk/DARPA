import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils.cadam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import FNO1d
from utils.datasets import PointJet1D, PointJetdy, PointJetdq
from utils.helper import count_params, interpolate_f2c
#%%
def get_init(x, y):
    '''
    x: input tensor with shape (batchszie, y_dim)
    y: position embedding (y_dim, )
    '''
    if len(x.shape) <= 2:
        x_data = x.unsqueeze(-1)
    else:
        x_data = x
    y_data = y.repeat(x.shape[0], 1).unsqueeze(-1)
    return torch.cat([x_data, y_data], dim=-1)



def train(dataloader, model, optimizer, scheduler, config, interpolate=False, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # hyperparameters
    N_y = config['N_y']
    num_epoch = config['num_epoch']
    num_pad = config['num_pad']
    base_dir = os.path.join('exp', config['logdir'])
    save_img_dir = os.path.join(base_dir, 'figs')
    save_ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_ckpt_dir, exist_ok=True)

    # training
    pbar = tqdm(range(num_epoch), dynamic_ncols=True, smoothing=0.1)
    criterion = nn.MSELoss()
    best_model = None
    best_err = 1.0

    L = 4 * np.pi
    yy = torch.linspace(-L / 2.0, L / 2.0, N_y, device=device)
    if interpolate:
        yy = interpolate_f2c(yy)

    model.train()
    for e in pbar:
        train_loss = 0
        for q, closure in dataloader:
            in_q = get_init(q, yy)
            in_q = in_q.to(device)
            in_q_pad = F.pad(in_q, (0, 0, num_pad, num_pad), 'constant', 0)

            optimizer.zero_grad()

            pred_pad = model(in_q_pad).squeeze(-1)
            pred = pred_pad[..., num_pad: -num_pad]
            loss = criterion(pred, closure)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        scheduler.step()
        avg_loss = train_loss / len(dataloader)
        if avg_loss < best_err:
            best_err = avg_loss
            best_model = deepcopy(model)
        pbar.set_description(
            (
                f'Epoch {e}: avg loss : {avg_loss}, best error: {best_err}'
            )
        )
            # q_pad = F.pad(x, (0, num_pad), 'constant', 0)
            # pred_pad = model(q_pad)
        if e % 50 == 0:
            fig = plt.figure()
            line1, = plt.plot(yy.cpu().numpy(), pred[0, :].detach().cpu().numpy(), label='prediction')
            line2, = plt.plot(yy.cpu().numpy(), closure[0], label='Ground truth')
            plt.legend()
            plt.xlabel('yy')
            plt.ylabel('closure * tau')
            plt.savefig(f'{save_img_dir}/{e}.png', bbox_inches='tight')
            plt.close(fig)


    torch.save(best_model.state_dict(), f'{save_ckpt_dir}/fno1d-best.pt')
    return model

#%%
if __name__ == '__main__':
    config = {
        'N_y': 384, 
        'lr': 0.001, 
        'logdir': 'dq-tau004-elu-mode8x3',
        'num_epoch': 8000,
        'batchsize': 10,
        'beta': 1.0,
        'num_pad': 2,
        'dy': False,
        'dq': True
    }
    #%%
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # training parameter
    batchsize = config['batchsize']
    N_y = config['N_y']
    dy = 4 * np.pi / (N_y - 1)
    # tau_invs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.2]
    tau_invs = [0.04]
    # model configuration
    layers = [4, 4, 4, 4]
    modes1 = [8, 8, 8]
    fc_dim = 4
    activation = 'elu'
    if config['dq']:
        in_dim = 3
    else:
        in_dim = 2
    # prepare data
    data_dir = 'data'
    if config['dy']:
        dataset = PointJetdy(data_dir, tau_invs, dy=dy)
        print('closure_dy')
    elif config['dq']:
        dataset = PointJetdq(data_dir, tau_invs)
    else:
        dataset = PointJet1D(data_dir, tau_invs)
    
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # prepare model
    model = FNO1d(modes1=modes1, layers=layers,
                  fc_dim=fc_dim, in_dim=in_dim, out_dim=1, activation=activation).to(device)
    print('number of parameters', count_params(model))

    # prepare optimization algo
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 4000], gamma=0.5)
    # train
    model = train(dataloader, model, optimizer, scheduler, 
                  interpolate=config['dy'], config=config, device=device)
