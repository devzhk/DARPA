import numpy as np
from .helper import gradient_first_f2c, gradient_first_c2f, interpolate_f2c


# the model is a function: w,t ->  M(w)
def explicit_solve(model, q_jet, tau, dt=1.0, Nt=1000, save_every=1, L=4 * np.pi):
    Ny = q_jet.size
    yy = np.linspace(-L / 2.0, L / 2.0, Ny)
    dy = L / (Ny - 1)

    t = 0.0
    # q has Dirichlet boundary condition
    q = np.copy(q_jet)

    q_data = np.zeros((Nt // save_every + 1, Ny))
    t_data = np.zeros(Nt // save_every + 1)
    q_data[0, :], t_data[0] = q, t

    res = np.zeros(Ny - 2)

    for i in range(1, Nt + 1):
        res = model(q, yy)

        # (q^{n+1} - q^n)/dt = res + (q_jet - q^{n+1})/tau
        q[1:Ny - 1] = dt * tau / (dt + tau) * (q_jet[1:Ny - 1] / tau + res + q[1:Ny - 1] / dt)

        if i % save_every == 0:
            q_data[i // save_every, :] = q
            t_data[i // save_every] = i * dt
    #             print(i, "max q", np.max(q))
    return yy, t_data, q_data


def nummodel(model, q, yy):
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)

    mu_c = model(q_c)
    # mu_c[mu_t >=0] = 0.0
    # mu_c[mu_t <=-0.1] = 0.0

    res = gradient_first_c2f(mu_c * dq_c, dy)
    return res