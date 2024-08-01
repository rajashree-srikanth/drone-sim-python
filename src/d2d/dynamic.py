import numpy as np, scipy.integrate

import d2d.utils as d2u

class Aircraft:
    i_phi, i_va, i_size = np.arange(3)
    s_x, s_y, s_psi, s_phi, s_va, s_size = np.arange(6) # no. of states = 5
    s_slice_pos  = slice(s_x, s_y+1)
    g = 9.81
    def __init__(self):
        self.tau_phi =  0.9667# 0.01
        self.tau_v = 1. # roll and speed time constants

    def cont_dyn(self, X, t, U, W):
        wx, wy = W.sample(t, X[:2])
        # breakpoint()
        (x, y, psi, phi, v), (phi_c, v_c) = X, U
        Xdot=[ v*np.cos(psi)+wx,
               v*np.sin(psi)+wy,
               self.g/v*np.tan(phi),
               -1/self.tau_phi*(phi-phi_c),
               -1/self.tau_v*(v-v_c)]
        return Xdot

    def disc_dyn(self, Xk, Uk, W, t, dt):
        Xk, Xkp1 = scipy.integrate.odeint(self.cont_dyn, Xk, [t, t+dt], args=(Uk, W))
        Xkp1[self.s_psi] = d2u.norm_mpi_pi(Xkp1[self.s_psi])
        return Xkp1


# locally linearized dynamic state model
    def cont_jac(self, Xr, Ur, t, W):
        psi, phi, va = Xr[Aircraft.s_psi], Xr[Aircraft.s_phi], Xr[Aircraft.s_va]
        g = self.g
        spsi, cpsi = np.sin(psi), np.cos(psi)
        cphi2, tan_phi = np.cos(phi)**2, np.tan(phi)
        A = np.array([ [0., 0., -va*spsi, 0.            , cpsi],
                       [0., 0.,  va*cpsi, 0.            , spsi],
                       [0., 0.,  0.,      g/va/(1+cphi2), g/va**2*tan_phi],
                       [0., 0.,  0.,     -1/self.tau_phi, 0],
                       [0., 0.,  0.,      0.,             -1/self.tau_v]])
        B = np.array([ [0, 0], [0,0], [0,0], [1/self.tau_phi, 0], [0, 1/self.tau_v]])
        return A, B
