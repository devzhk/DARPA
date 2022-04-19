import scipy.io
import numpy as np

import scipy.ndimage


# compute gradient from face states to cell gradients, no boundary
def gradient_first_f2c(omega, dx, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        d_omega = np.copy(omega)
        d_omega[0:nx - 1] = (omega[1:nx] - omega[0:nx - 1]) / (dx)
        d_omega[nx - 1] = (omega[0] - omega[nx - 1]) / (dx)
    else:
        d_omega = (omega[1:nx] - omega[0:nx - 1]) / (dx)
    return d_omega


# compute gradient from cell states to face gradients, no boundary
def gradient_first_c2f(omega, dx, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        d_omega = np.zeros(nx)
        d_omega[1:nx] = (omega[1:nx] - omega[0:nx - 1]) / (dx)
        d_omega[0] = (omega[0] - omega[nx - 1]) / (dx)
    else:
        d_omega = (omega[1:nx] - omega[0:nx - 1]) / (dx)
    return d_omega


# compute gradient from cell states to face gradients, no boundary
def interpolate_f2c(omega, bc="None"):
    nx = len(omega)

    if bc == "periodic":
        c_omega = np.zeros(nx)
        c_omega[0:nx - 1] = (omega[0:nx - 1] + omega[1:nx]) / 2.0
        c_omega[nx - 1] = (omega[0] + omega[nx - 1]) / 2.0
    else:
        c_omega = (omega[0:nx - 1] + omega[1:nx]) / 2.0
    return c_omega


def gradient_first(omega, dx, type='copy'):
    nx = len(omega)
    if type == 'copy':
        d_omega = np.copy(omega)
    else:
        d_omega = omega.clone()

    d_omega[1:nx-1] = (omega[2:nx] - omega[0:nx-2]) / (2*dx)
    d_omega[0], d_omega[nx-1] = d_omega[1], d_omega[nx-2]
    return d_omega


def load_data(data_dir, start=1000, end=2000):
    closure = scipy.io.loadmat(data_dir + "data_closure_cons.mat")["data_closure_cons"]
    dq_dy = scipy.io.loadmat(data_dir + "data_dq_dy.mat")["data_dq_dy"]
    q = scipy.io.loadmat(data_dir + "data_q.mat")["data_q"]

    q_new = q[0, :, start:end]
    dq_dy_new = dq_dy[0, :, start:end]
    closure_new = closure[0, :, start:end]

    return closure_new, q_new, dq_dy_new


def load_data_mean(data_dir):
    closure = -scipy.io.loadmat(data_dir + "data_closure_cons.mat")["data_closure_cons"]
    dq_dy = scipy.io.loadmat(data_dir + "data_dq_dy.mat")["data_dq_dy"]
    q = scipy.io.loadmat(data_dir + "data_q.mat")["data_q"]

    _, Ny, Nt = closure.shape

    q_mean = np.mean(q[0, :, Nt // 2:], axis=1)
    dq_dy_mean = np.mean(dq_dy[0, :, Nt // 2:], axis=1)
    closure_mean = np.mean(closure[0, :, Nt // 2:], axis=1)

    return closure_mean, q_mean, dq_dy_mean


def preprocess(N_y, beta, tau_inv):
    data_dirs = ["data/beta_1.0_Gamma_1.0_relax_" + str(tau_inv[i]) + "/" for i in range(len(tau_inv))]

    N_data = len(data_dirs)
    closure_mean, q_mean, dq_dy_mean = np.zeros((N_data, N_y)), np.zeros((N_data, N_y)), np.zeros((N_data, N_y))
    for i in range(N_data):
        closure_mean[i, :], q_mean[i, :], dq_dy_mean[i, :] = load_data(data_dirs[i])

    L = 4 * np.pi
    yy = np.linspace(-L / 2.0, L / 2.0, N_y)
    dy = yy[1] - yy[0]

    omega_jet = np.zeros(N_y)
    omega_jet[0:N_y // 2] = 1.0
    omega_jet[N_y // 2:N_y] = -1.0
    q_jet = omega_jet + beta * yy

    # clean data
    chop_l = 50
    for i in range(N_data):
        q_mean[i, 0:chop_l] = np.linspace(q_jet[0], q_mean[i, chop_l - 1], chop_l)  # q_jet[0:chop_l]
        q_mean[i, -chop_l:] = np.linspace(q_mean[i, -chop_l], q_jet[-1], chop_l)  # q_jet[-chop_l:]
        dq_dy_mean[i, 0:chop_l] = np.linspace(beta, dq_dy_mean[i, chop_l - 1], chop_l)
        dq_dy_mean[i, -chop_l:] = np.linspace(dq_dy_mean[i, -chop_l], beta, chop_l)

    q_mean_abs = np.fabs(q_mean)
    mu_f = closure_mean / dq_dy_mean

    # clip and filter the data
    mu_f[mu_f >= 0.1] = 0.0
    mu_f[mu_f <= 0.0] = 0.0
    for i in range(N_data):
        mu_f[i, :] = scipy.ndimage.gaussian_filter1d(mu_f[i, :], 5)
    return closure_mean, q_mean, yy, dq_dy_mean, mu_f


def count_params(model):
    num_param = 0
    for p in model.parameters():
        num_param += p.numel()
    return num_param