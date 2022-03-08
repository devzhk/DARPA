import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

from functools import partial
from models.fno1d import FNO1d
from utils.helper import gradient_first

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def explicit_solve(model,   #
                   tau,
                   q_jet,
                   dy,
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
    for i in range(1, Nt + 1):
        q = dt * tau / (dt + tau) * (q_jet / tau - model(q, dy) + q / dt)

        if i % save_every == 0:
            q_data[i // save_every, :] = q
            t_data[i // save_every] = i * dt

    return t_data, q_data


def closure(fno, q, dy):
    num_pad = 4
    q = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(dim=0)
    q_pad = F.pad(q, (0, num_pad), 'constant', 0)
    pred = fno(q_pad.unsqueeze(-1))
    pred = pred.squeeze(-1)[...,:-num_pad]
    dy_pred = gradient_first(pred.squeeze(0).cpu(), dy)
    return dy_pred


if __name__ == '__main__':
    tau = 10.0 # TBD
    beta = 1.0 # TBD
    N = 384
    layers = [32, 32]
    modes1 = [12, 12]
    fc_dim = 32

    omega_path = 'data/test_data_w.mat'
    w = scipy.io.loadmat(omega_path)

    omega_jet = np.zeros(N)
    omega_jet[0:N // 2] = 1.0
    omega_jet[N - N // 2:N] = -1.0

    L = 4 * np.pi
    yy = np.linspace(-L / 2.0, L / 2.0, N)
    dy = L / (N - 1)
    q_jet = yy * beta + omega_jet

    fno_model = FNO1d(modes1=modes1, fc_dim=fc_dim, layers=layers, in_dim=1, activation='gelu').to(device)
    ckpt_path = 'ckpts/fno1d-pad5.pt'
    ckpt = torch.load(ckpt_path)
    fno_model.load_state_dict(ckpt)
    fno_model.eval()
    with torch.no_grad():
        model = partial(closure, fno_model)
        t_data, q_data = explicit_solve(model, tau, q_jet, dy=dy,
                                        dt = 0.001, Nt = 500000, save_every = 100)
    plt.plot(np.mean(q_data, axis=0), yy, label='fit')
    plt.plot(np.mean(w[0,:,:].T, axis=0) + yy, yy, label='truth')
    plt.legend()
    plt.savefig('figs/result.png')