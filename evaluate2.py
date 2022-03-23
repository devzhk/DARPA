import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

from functools import partial
from models.fno1d import FNO1d
from utils.solver import explicit_solve, nummodel

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
beta = 1.0


def point_jet(tau_inv, fno_model):
    print(f'test {tau_inv}')

    tau = 1 / tau_inv
    Nx = 384
    omega_jet = np.zeros(Nx)
    omega_jet[0:Nx // 2] = 1.0
    omega_jet[Nx // 2:Nx] = -1.0
    L = 4 * np.pi
    yy = np.linspace(-L / 2.0, L / 2.0, Nx)
    q_jet = omega_jet + beta * yy

    data_dir = f'data/beta_1.0_Gamma_1.0_relax_{tau_inv}/'
    dq_dy = scipy.io.loadmat(data_dir + "data_dq_dy.mat")["data_dq_dy"]
    closure = scipy.io.loadmat(data_dir + "data_closure_cons.mat")["data_closure_cons"]
    w = scipy.io.loadmat(data_dir + "data_w.mat")["data_w"]
    q = scipy.io.loadmat(data_dir + "data_q.mat")["data_q"]

    _, Ny, Nt = q.shape
    q_mean_ref = np.mean(q[0, :, Nt // 2:], axis=1)
    w_mean_ref = np.mean(w[0, :, Nt // 2:], axis=1)

    dt, Nt, save_every = 1.0e-4, 200000, 1000
    yy, t_pred, q_pred = explicit_solve(fno_model, q_jet, tau, dt, Nt, save_every, L=L)
    q_mean_pred = np.mean(q_pred[Nt // (2 * save_every):, :], axis=0)

    return yy, q_mean_ref, q_mean_pred, q_pred


if __name__ == '__main__':
    N_y = 384
    tau_invs = [0.005, 0.06, 0.2]
    ckpt_path = 'ckpts/fno1d-nopad-final.pt'
    L = 4 * np.pi
    yy = np.linspace(-L / 2.0, L / 2.0, N_y)

    layers = [32, 32]
    modes1 = [12, 12]
    fc_dim = 32
    model = FNO1d(modes1=modes1, layers=layers,
                  fc_dim=fc_dim, in_dim=1, activation='tanh').to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.train()
    fno_model = partial(nummodel, model)
    with torch.no_grad():
        for i in range(len(tau_invs)):
            _, q_mean_ref, q_mean_pred, _ = point_jet(tau_invs[i], fno_model)
            plt.plot(q_mean_pred, yy, label='fit', alpha=0.5)
            plt.plot(q_mean_ref, yy, label='truth', alpha=0.5)
            plt.xlabel('y')
            plt.ylabel('q')
            plt.legend()
            plt.savefig(f'figs/pre-result-{i}.png')
            plt.clf()