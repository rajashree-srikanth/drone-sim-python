import numpy as np, scipy

class WindField:
    def get(self, t, loc):
        return [0., 0.]


class Aircraft:
    i_phi, i_v, i_size = np.arange(3)
    s_size=5
    def __init__(self):
        self.tau_phi = 0.1
        self.tau_v = 1.

    def cont_dyn(self, X, t, U, v=10., wx=0., g=9.81):
        psi, phi = X[2], U[0]
        Xdot=[v*np.cos(psi)+wx, v*np.sin(psi), g/v*np.tan(phi)]
        return Xdot

    def disc_dyn(self, Xk, Uk, dt, v=10., wx=0.):
        Xk, Xkp1 = scipy.integrate.odeint(cont_dyn, Xk, [0, dt], args=(Uk, v, wx))
        return Xkp1
