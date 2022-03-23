import scipy.ndimage
import numpy as np
from utils.helper import load_data

N_y = 384
beta = 1.0
tau_inv = [0.01, 0.02, 0.04, 0.08, 0.16]

def preprocess(N_y, beta, tau_inv):
    data_dirs = ["../data/beta_1.0_Gamma_1.0_relax_" + str(tau_inv[i]) + "/" for i in range(len(tau_inv))]

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

    # TODO: clean data
    chop_l = 50
    for i in range(N_data):
        q_mean[i, 0:chop_l] = np.linspace(q_jet[0], q_mean[i, chop_l - 1], chop_l)  # q_jet[0:chop_l]
        q_mean[i, -chop_l:] = np.linspace(q_mean[i, -chop_l], q_jet[-1], chop_l)  # q_jet[-chop_l:]
        dq_dy_mean[i, 0:chop_l] = np.linspace(beta, dq_dy_mean[i, chop_l - 1], chop_l)
        dq_dy_mean[i, -chop_l:] = np.linspace(dq_dy_mean[i, -chop_l], beta, chop_l)

    q_mean_abs = np.fabs(q_mean)
    mu_f = closure_mean / dq_dy_mean

    # TODO: clip and filter the data
    mu_f[mu_f >= 0.1] = 0.0
    mu_f[mu_f <= 0.0] = 0.0
    for i in range(N_data):
        mu_f[i, :] = scipy.ndimage.gaussian_filter1d(mu_f[i, :], 5)
    return closure_mean, q_mean, yy, dq_dy_mean, mu_f

