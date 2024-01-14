import numpy as np, scipy.integrate

class Aircraft:
    i_phi, i_v, i_size = np.arange(3)
    s_x, s_y, s_psi, s_phi, s_v, s_size = np.arange(6)
    s_slice_pos  = slice(s_x, s_y+1)
    g = 9.81
    def __init__(self):
        self.tau_phi, self.tau_v = 0.1, 1. # roll and speed time constants

    def cont_dyn(self, X, t, U, W):
        wx, wy = W.sample(t, X[:2])
        (x, y, psi, phi, v), (phi_c, v_c) = X, U
        Xdot=[v*np.cos(psi)+wx, v*np.sin(psi)+wy,
              self.g/v*np.tan(phi),
              -1/self.tau_phi*(phi-phi_c), -1/self.tau_v*(v-v_c)]
        return Xdot

    def disc_dyn(self, Xk, Uk, W, t, dt):
        Xk, Xkp1 = scipy.integrate.odeint(self.cont_dyn, Xk, [t, t+dt], args=(Uk, W))
        return Xkp1
