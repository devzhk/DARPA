import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from models.fno1d import FNO1d
from utils.helper import gradient_first

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_init(x, y):
    '''
    x: input tensor with shape (batchszie, y_dim)
    y: position embedding (y_dim, )
    '''
    x_data = x.unsqueeze(-1)
    y_data = y.repeat(x.shape[0], 1).unsqueeze(-1)
    return torch.cat([x_data, y_data], dim=-1)


def explicit_solve(model,   #
                   tau,
                   q_jet,
                   dy,
                   yy,
                   dt=1.0,
                   Nt=1000,
                   save_every=1,
                   ):
    t = 0.0
    q = np.copy(q_jet)

    q_data = np.zeros((Nt // save_every + 1, q_jet.shape[0]))
    t_data = np.zeros(Nt // save_every + 1)
    q_data[0, :], t_data[0] = q, t

    # dq = gradient_first(q, dy)
    for i in tqdm(range(1, Nt + 1)):
        q = dt * tau / (dt + tau) * (q_jet / tau + q / dt) - dt * model(q, yy, dy) / (dt + tau)

        if i % save_every == 0:
            q_data[i // save_every, :] = q
            t_data[i // save_every] = i * dt

    return t_data, q_data


def closure(fno, q, y, dy):
    # num_pad = 4
    q = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(dim=0)
    in_q = get_init(q, y)
    # q_pad = F.pad(q, (0, num_pad), 'constant', 0)
    pred = fno(in_q)
    pred = pred.squeeze(-1)
    dy_pred = gradient_first(pred.squeeze(0).cpu().numpy(), dy)
    return dy_pred


if __name__ == '__main__':

    beta = 1.0
    N = 384
    layers = [4, 4, 4]
    modes1 = [4, 4]
    fc_dim = 4
    activation = 'tanh'

    tau_inv = 0.04
    tau = 1 / tau_inv
    data_dir = f'data/beta_1.0_Gamma_1.0_relax_{tau_inv}/'
    # data_dir = 'data/'
    omega_path = f'{data_dir}data_w.mat'
    w = scipy.io.loadmat(omega_path)['data_w']

    omega_jet = np.zeros(N)
    omega_jet[0:N // 2] = 1.0
    omega_jet[N - N // 2:N] = -1.0

    L = 4 * np.pi
    yy = torch.linspace(-L / 2.0, L / 2.0, N, device=device)
    np_yy = yy.cpu().numpy()
    dy = L / (N - 1)
    q_jet = np_yy * beta + omega_jet
    # q_jet = np.linspace(- L / 2 + 1, L / 2 -1, N)

    fno_model = FNO1d(modes1=modes1, fc_dim=fc_dim, layers=layers, in_dim=2, out_dim=1, activation=activation).to(device)
    ckpt_path = 'exp/tanh-mode4/ckpts/fno1d-best.pt'
    ckpt = torch.load(ckpt_path)
    fno_model.load_state_dict(ckpt)
    fno_model.eval()
    with torch.no_grad():
        model = partial(closure, fno_model)
        t_data, q_data = explicit_solve(model, tau, q_jet, dy=dy, yy=yy,
                                        dt = 0.001, Nt=100000, save_every = 100)
    plt.plot(np.mean(q_data, axis=0), np_yy, label='fit')
    plt.plot(np.mean(w[0,:,:].T, axis=0) + np_yy, np_yy, label='truth')
    plt.xlabel('y')
    plt.ylabel('q')
    plt.legend()
    plt.savefig(f'figs/test-result-{tau_inv}.png')