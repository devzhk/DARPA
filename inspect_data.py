#%%
import numpy as np
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
import seaborn as sns

#%%
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


def load_data(data_dir):
    closure = scipy.io.loadmat(data_dir + "data_closure_cons.mat")["data_closure_cons"]
    dq_dy = scipy.io.loadmat(data_dir + "data_dq_dy.mat")["data_dq_dy"]
    q = scipy.io.loadmat(data_dir + "data_q.mat")["data_q"]

    _, Ny, Nt = closure.shape

    q_mean = np.mean(q[0, :, Nt // 2:], axis=1)
    dq_dy_mean = np.mean(dq_dy[0, :, Nt // 2:], axis=1)
    closure_mean = np.mean(closure[0, :, Nt // 2:], axis=1)

    return closure_mean, q_mean, dq_dy_mean



#%%
tau_inv = 0.16
data_dir = f'data/beta_1.0_Gamma_1.0_relax_{tau_inv}/'

closure_mean, q_mean, dq_dy_mean = load_data(data_dir)
# %%
plt.plot(dq_dy_mean, closure_mean)
plt.xlabel('dq dy mean')
plt.ylabel('closure mean')
plt.savefig('figs/dq_dy_vs_closure.png')
plt.show()
#%%
plt.plot(q_mean, closure_mean)
plt.xlabel('q mean')
plt.ylabel('closure mean')
plt.savefig(f'figs/q_vs_closure_{tau_inv}.png')
plt.show()
#%%
plt.plot(q_mean, closure_mean / tau_inv)
plt.xlabel('q mean')
plt.ylabel('closure mean * tau')
plt.savefig(f'figs/q_vs_tau_closure_{tau_inv}.png')
plt.show()
# %%
N_y = q_mean.shape[0]
L = 4 * np.pi
yy = np.linspace(-L / 2.0, L / 2.0, N_y)
dy = yy[1] - yy[0]

omega_jet = np.zeros(N_y)
omega_jet[0:N_y // 2] = 1.0
omega_jet[N_y // 2:N_y] = -1.0
q_jet = omega_jet + yy

chop_l = 50
q_mean[0:chop_l] = np.linspace(q_jet[0], q_mean[chop_l - 1], chop_l)
q_mean[-chop_l:] = np.linspace(q_mean[-chop_l], q_jet[-1], chop_l) 
dq_dy_mean[0:chop_l] = np.linspace(1, dq_dy_mean[chop_l - 1], chop_l)
dq_dy_mean[-chop_l:] = np.linspace(dq_dy_mean[-chop_l], 1, chop_l)
# %%
mu_f = closure_mean / dq_dy_mean
# mu_f[mu_f >= 0.1] = 0.0
# mu_f[mu_f <= 0.0] = 0.0

# mu_f = scipy.ndimage.gaussian_filter1d(mu_f, 5)
plt.plot(yy, mu_f)
plt.xlabel('y')
plt.ylabel('Mu f')
plt.savefig('figs/mu_f_y.png')
plt.show()
# %%

# %%
plt.plot(yy, mu_f)
plt.xlabel('y')
plt.ylabel('Mu f')
plt.savefig('figs/mu_f_y.png')
plt.show()
# %%
plt.plot(yy, q_mean)
plt.xlabel('y')
plt.ylabel('q_mean')
plt.savefig('figs/q_mean_y.png')
plt.show()
# %%
avg = np.mean(closure_mean)
# %%
plt.plot(q_mean, closure_mean * 100)
plt.xlabel('q mean')
plt.ylabel('closure mean / avg')
plt.show()
# %%
# inspect individual data points 
tau_inv = 0.02
data_dir = f'data/beta_1.0_Gamma_1.0_relax_{tau_inv}/'
closure = scipy.io.loadmat(data_dir + "data_closure_cons.mat")["data_closure_cons"]
dq_dy = scipy.io.loadmat(data_dir + "data_dq_dy.mat")["data_dq_dy"]
q = scipy.io.loadmat(data_dir + "data_q.mat")["data_q"]
#%%
for i in range(5):
    plt.plot(yy, np.mean(closure[0, :, 2000 + i * 100: 2100 + i * 100], axis=1) / tau_inv, label=f't=2{i}00', alpha=0.5)
plt.xlabel('q mean')
plt.ylabel('closure mean * tau')
plt.legend()
plt.show()

# %%
closure_mid = closure[0, :, 1500: 2500]
# %%
closure_avg = np.mean(closure_mid.reshape(closure_mid.shape[0], 10, 100), axis=-1)
# %%
for i in range(5):
    plt.plot(yy, closure_avg[:, i] / tau_inv, label=f't=2{i}00', alpha=0.5)
plt.xlabel('q mean')
plt.ylabel('closure mean * tau')
plt.legend()
plt.show()
# %%
q_mid = q[0, :, 1500:2500]

q_avg = np.mean(q_mid.reshape(q_mid.shape[0], 10, 100), axis=-1)
for i in range(5):
    plt.plot(yy, q_avg[:, i], label=f't=2{i}00', alpha=0.5)
plt.xlabel('q mean')
plt.ylabel('q avg')
plt.legend()
plt.show()
#
# %%
from utils.datasets import PointJet1D
#%%
dataset = PointJet1D(datapath='data', tau_invs=[0.01, 0.02, 0.04, 0.08, 0.16, 0.06])
#%%
N_y = dataset[0][0].shape[0]
L = 4 * np.pi
yy = np.linspace(-L / 2.0, L / 2.0, N_y)
# %%

closure = dataset[55][1]

plt.plot(yy, closure)
plt.xlabel('yy')
plt.ylabel('closure * tau')
plt.show()

# %%
tau_inv = 0.06
data_dir = f'data/beta_1.0_Gamma_1.0_relax_{tau_inv}/'
w = scipy.io.loadmat(data_dir + "data_w.mat")["data_w"]

# %%
w.shape
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
data = np.random.normal(loc=0.0, scale=0.1, size=3000)
sns.kdeplot(data)
plt.show()
# %%
data = np.random.normal(loc=0.0, scale=0.01, size=3000)
sns.kdeplot(data)
plt.show()
# %%
data = np.random.normal(loc=0.0, scale=0.001, size=3000)
sns.kdeplot(data)
plt.show()
# %%
