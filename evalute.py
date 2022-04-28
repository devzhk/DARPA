import os
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from models.fno1d import FNO1d
from utils.helper import gradient_first, interpolate_f2c, gradient_first_c2f

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def explicit_solve2(model, 
                    tau, 
                    q_jet,
                    dy, 
                    yy, 
                    dt=1.0, 
                    Nt=1000, 
                    save_every=1):
    Ny = yy.shape[0]

    t = 0.0
    # q has Dirichlet boundary condition
    q = np.copy(q_jet)

    q_data = np.zeros((Nt // save_every + 1, Ny))
    t_data = np.zeros(Nt // save_every + 1)
    q_data[0, :], t_data[0] = q, t

    res = np.zeros(Ny - 1)

    for i in tqdm(range(1, Nt + 1)):
        res = model(q, yy, dy)

        # (q^{n+1} - q^n)/dt = res + (q_jet - q^{n+1})/tau
        q[1:Ny - 1] = dt * tau / (dt + tau) * (q_jet[1:Ny - 1] / tau + res + q[1:Ny - 1] / dt)

        if i % save_every == 0:
            q_data[i // save_every, :] = q
            t_data[i // save_every] = i * dt
    return t_data, q_data


def explicit_solve(model,   #
                   tau,
                   q_jet,
                   dy,
                   yy,
                   dt=1.0,
                   Nt=1000,
                   save_every=1,
                   interpolate=False
                   ):
    t = 0.0
    if interpolate:
        q_jet = interpolate_f2c(q_jet)
        in_yy = interpolate_f2c(yy)
    else:
        in_yy = yy

    q = np.copy(q_jet)

    q_data = np.zeros((Nt // save_every + 1, q_jet.shape[0]))
    t_data = np.zeros(Nt // save_every + 1)
    q_data[0, :], t_data[0] = q, t

    # dq = gradient_first(q, dy)
    for i in tqdm(range(1, Nt + 1)):
        q = dt * tau / (dt + tau) * (q_jet / tau + q / dt) - dt * model(q, in_yy, dy) / (dt + tau)

        if i % save_every == 0:
            q_data[i // save_every, :] = q
            t_data[i // save_every] = i * dt

    return t_data, q_data


def closure(fno, num_pad, q, y, dy):
    # num_pad = 4
    q = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(dim=0)
    in_q = get_init(q, y)
    q_pad = F.pad(in_q, (0, 0, num_pad, num_pad), 'constant', 0)
    pred_pad = fno(q_pad).squeeze(-1)
    pred = pred_pad[..., num_pad:-num_pad]
    dy_pred = gradient_first(pred.squeeze(0).cpu().numpy(), dy)
    return dy_pred


def closure_dy(fno, num_pad, q, y, dy):
    q = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(dim=0)
    in_q = get_init(q, y)
    q_pad = F.pad(in_q, (0, 0, num_pad, num_pad), 'constant', 0)
    pred_pad = fno(q_pad).squeeze(-1)
    pred = pred_pad[..., num_pad:-num_pad]
    cls_dy = pred.squeeze(0).cpu().numpy()
    return cls_dy


def closure_dq(fno, num_pad, q, y, dy):
    dq_c = gradient_first_c2f(q, dy)
    q_c = interpolate_f2c(q)

    dq_c_tensor = torch.tensor(dq_c, dtype=torch.float32)
    q_c_tensor = torch.tensor(q_c, dtype=torch.float32)
    in_feature = torch.stack([dq_c_tensor, q_c_tensor], dim=-1).unsqueeze(0)
    in_y = interpolate_f2c(y)
    in_q = get_init(in_feature, in_y)

    q_pad = F.pad(in_q, (0, 0, num_pad, num_pad), 'constant', 0)
    pred_pad = fno(q_pad).squeeze(-1)
    pred = pred_pad[..., num_pad:-num_pad]
    mu_c = pred.squeeze(0).cpu().numpy()

    mu_c[mu_c<0] = 0.0
    res = gradient_first_c2f(mu_c * dq_c, dy)
    return res


if __name__ == '__main__':
    base_dir = 'exp/dq-tau016-elu-mode8'
    num_pad = 2
    
    closure_type = 'dq' # dq, dy, otherwise
    interpolate = True
    beta = 1.0
    N = 384
    layers = [4, 4, 4]
    modes1 = [8, 8]
    fc_dim = 4
    activation = 'elu'

    tau_inv = 0.04

    if closure_type == 'dq':
        in_dim = 3
    else:
        in_dim = 2

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

    fno_model = FNO1d(modes1=modes1, fc_dim=fc_dim, layers=layers, in_dim=in_dim, out_dim=1, activation=activation).to(device)
    ckpt_path = os.path.join(base_dir, 'ckpts/fno1d-best.pt')
    ckpt = torch.load(ckpt_path)
    fno_model.load_state_dict(ckpt)
    fno_model.eval()
    with torch.no_grad():
        if closure_type == 'dy':
            model = partial(closure_dy, fno_model, num_pad)
        elif closure_type == 'dq':
            model = partial(closure_dq, fno_model, num_pad)
        else:
            model = partial(closure, fno_model, num_pad)
        
        if closure_type == 'dq':
            t_data, q_data = explicit_solve2(model, tau, q_jet, dy=dy, yy=yy,
                                             dt=0.0001, Nt=100000, save_every=100)
        else:
            t_data, q_data = explicit_solve(model, tau, q_jet, dy=dy, yy=yy,
                                            dt=0.0001, Nt=100000, save_every=100, interpolate=interpolate)

    if interpolate:
        np_yy1 = interpolate_f2c(np_yy)
    else:
        np_yy1 = np_yy
    plt.plot(np.mean(q_data[:, :-1], axis=0), np_yy1, label='fit')
    plt.plot(np.mean(w[0,:,:].T, axis=0) + np_yy, np_yy, label='truth')
    plt.xlabel('q')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'{base_dir}/test-result-{tau_inv}.png')