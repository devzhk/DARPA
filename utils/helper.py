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