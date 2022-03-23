import matplotlib.pyplot as plt


def visualize(closure_mean, q_mean, dq_dy_mean, mu_f, yy, tau_inv, dir):
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=False, sharey=True, figsize=(16, 6))
    N_data = closure_mean.shape[0]
    for i in range(N_data):
        ax[0].plot(closure_mean[i, :], yy, label=str(tau_inv[i]))
        ax[1].plot(q_mean[i, :], yy, label=str(tau_inv[i]))
        ax[2].plot(dq_dy_mean[i, :], yy, label=str(tau_inv[i]))
        ax[3].plot(mu_f[i, :], yy, label=str(tau_inv[i]))

        ax[0].set_xlabel("closure")
        ax[1].set_xlabel("q")
        ax[2].set_xlabel("dq_dy")
        ax[3].set_xlabel("mu")
        ax[0].set_ylabel('yy')
    # plt.ylabel('yy')
    plt.legend()
    plt.savefig(f'{dir}/data_vis.png')
