import scipy.io
import numpy as np


def gradient_first(omega, dx, type='copy'):
    nx = len(omega)
    if type == 'copy':
        d_omega = np.copy(omega)
    else:
        d_omega = omega.clone()

    d_omega[1:nx-1] = (omega[2:nx] - omega[0:nx-2]) / (2*dx)
    d_omega[0], d_omega[nx-1] = d_omega[1], d_omega[nx-2]
    return d_omega


def load_data(data_dir):
    closure = -scipy.io.loadmat(data_dir + "data_closure_cons.mat")["data_closure_cons"]
    dq_dy = scipy.io.loadmat(data_dir + "data_dq_dy.mat")["data_dq_dy"]
    q = scipy.io.loadmat(data_dir + "data_q.mat")["data_q"]

    _, Ny, Nt = closure.shape

    q_mean = np.mean(q[0, :, Nt // 2:], axis=1)
    dq_dy_mean = np.mean(dq_dy[0, :, Nt // 2:], axis=1)
    closure_mean = np.mean(closure[0, :, Nt // 2:], axis=1)

    return closure_mean, q_mean, dq_dy_mean