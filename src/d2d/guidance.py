import numpy as np, scipy.integrate


import d2d.trajectory as ddt
from d2d.trajectory import Trajectory
import d2d.dynamic as ddyn
from d2d.dynamic import Aircraft


def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

class WindField: # initializes wind vel = 0 unless specifed in scenario
    def __init__(self, w=[0.,0.]):
        self.w = w
    def sample(self, t, loc):
        return self.w
    def summarize(self):
        r = f'{self.w} m/s'
        return r

 
class DiffFlatness:
    def state_and_input_from_output(Ys, W, ac):
        X, U, Xdot = [np.zeros(_l) for _l in [Aircraft.s_size, Aircraft.i_size, Aircraft.s_size]]
        x, y = Ys[0, Trajectory.cx], Ys[0, Trajectory.cy]
        xd, yd = Ys[1,Trajectory.cx], Ys[1,Trajectory.cy]
        xdd, ydd = Ys[2,Trajectory.cx], Ys[2,Trajectory.cy]
        X[Aircraft.s_x], X[Aircraft.s_y] = x, y         # x, y
        vax, vay = xd - W[0], yd - W[1]         
        va2 = vax**2+vay**2; va = np.sqrt(va2)
        X[Aircraft.s_va] = va                           # va
        X[Aircraft.s_psi] = np.arctan2(vay, vax)        # psi
        wxdot, wydot = 0., 0. # wind is stationnary
        vaxd, vayd = xdd-wxdot, ydd-wydot
        Xdot[Aircraft.s_va] = (vax*vaxd + vay*vayd)/va  # va dot
        Xdot[Aircraft.s_psi] = (vayd*vax-vaxd*vay)/va2  # psi dot

        g=9.81 # phi
        X[Aircraft.s_phi] = np.arctan((vayd*vax-vaxd*vay)/va/g)
        # U
        #Xdot[Aircraft.s_phi] # phi_dot
        U[Aircraft.i_phi] = ac.tau_phi * Xdot[Aircraft.s_phi] + X[Aircraft.s_phi]
        U[Aircraft.i_va] = ac.tau_v * Xdot[Aircraft.s_va] + X[Aircraft.s_va]
        
        
        return X, U, Xdot


import control

class DFFFController:
    def __init__(self, traj, ac, wind):
        self.traj, self.ac, self.wind = traj, ac, wind
        self.dt = 0.01
        self.time = np.arange(0, traj.duration, self.dt)
        self.carrot, self.ref_pos = [0,0], [0,0]
        self.Xref = []
        self.K = []
        
    # the controller definition and how it works, or what it does    
    def get(self, X, t):
        Yref = self.traj.get(t) 
        W = self.wind.sample(t, Yref[0])
        Xr, Ur, Xrdot = DiffFlatness.state_and_input_from_output(Yref, W, self.ac)
        self.Xref.append(Xr)
        dX = X - Xr
        dX[Aircraft.s_psi] = norm_mpi_pi(dX[Aircraft.s_psi])
        err_sats = np.array([20, 20 , np.pi/3, np.pi/4, 1])
        dX = np.clip(dX, -err_sats, err_sats)
        A, B = self.ac.cont_jac(Xr, Ur, t, W)
        #val_p, vect_p = np.linalg.eig(A)
        #print(val_p)
        if 0: # dim 5 feedback
            Q, R = [1, 1, 0.1, 0.01, 0.01,], [8, 1]
            (K, X, E) = control.lqr(A, B, np.diag(Q), np.diag(R))
        else: # dim 3 feedback
            A1,B1 = A[:3,:3], A[:3,3:]
            Q, R = [1, 1, 0.1], [8, 1]
            (K1, X, E) = control.lqr(A1, B1, np.diag(Q), np.diag(R))
            K=np.zeros((2,5))
            K[:,:3]=K1
        self.K.append(K)
        #valp2, vectp2 =  np.linalg.eig(A-np.dot(B, K))
        dU = -np.dot(K, dX)
        U = U1 = Ur + dU
        phisat, vmin, vmax = np.deg2rad(45), 4, 20
        U = np.clip(U, [-phisat, vmin], [phisat, vmax])
        #print(valp2, K, U1, U)
        return U

    def draw_debug(self, _f, _a, time):
        Xref = np.array(self.Xref)
        _a[0,0].plot(time, Xref[:,0])
        
# gvf controller
# generating the trajectory - for now, it is just a circle
# a function to plot the trajectory we want
# a function or module for the gvf controller
# class GeneratePath:
# class CircleTraj:
#     def __init__(self):
#         pass
#     def circle(self):
        
# class GVFcontroller:
#     def __init__(self):
#         self.
# #
# #  old stuff, initial 2D pure pursuit
# #


class VelControler:
    def __init__(self):
        self.Kp, self.Ki = 2., 0.001
        self.sat_err = 10. # s
        #self.sat_sum_err = 
        self.sat_vel = 4   # m/s
        self.ref_vel = 10.
        self.sum_err = 0.  # s2
        
    def get(self, tself, tref):
        timing_error = tself - tref # s?
        timing_error = np.clip(timing_error, -self.sat_err, self.sat_err)
        self.sum_err += timing_error
        v = self.ref_vel - np.clip(self.Kp * timing_error + self.Ki * self.sum_err, -self.sat_vel, self.sat_vel) # m/s
        return v
    
class PurePursuitControler:
    def __init__(self, traj):
        self.traj = traj
        if 0:
            t0, t1, dt = 0, 60, 0.01 # FIXME
            self.time = np.arange(t0, t1, dt)
        else:
            #print(traj.duration)
            dt = 0.01
            self.time = np.arange(0, traj.duration, dt)

        self.pts_2d = np.array([traj.get(t)[0] for t in self.time])
        self.sat_phi = np.deg2rad(45.)
        self.ref_pos, self.carrot = [], []
        self.vel_ctl = VelControler()
        self.control_vel = False#True
        
    def get(self, X, t):#, time):
        dists = np.linalg.norm(self.pts_2d-X[:ddyn.Aircraft.s_y+1], axis=1)
        idx_closest = np.argmin(dists)
        self.ref_pos.append(self.pts_2d[idx_closest])
        t0, t1 = self.time[idx_closest], t
        #print(f'{t0:.1f}, {t1:.1f} {t0 - t1:.1f}')
        tself = self.time[idx_closest]
        if 1:
            lookahead_m, dt, v = 10, 0.01, 10 # lookahaead in m, dt in sec, v in m/s
            #idx_carrot = min(idx_closest + int(lookahead_m/v/dt), len(self.pts_2d)-1)
            idx_carrot = idx_closest + int(lookahead_m/v/dt)
            if idx_carrot >= len(self.pts_2d):
                idx_carrot -= len(self.pts_2d)
            carrot = self.pts_2d[idx_carrot]
        else:
            lookahead_s = 2.
            carrot = self.traj.get(t+lookahead_s)[0] 
        self.carrot.append(carrot)
        pc = carrot-X[ddyn.Aircraft.s_slice_pos]
        err_psi = norm_mpi_pi(X[2] - np.arctan2(pc[1], pc[0]))
        K= 1.#0.2
        phi_sp = -K*err_psi
        phi_sp = np.clip(phi_sp, -self.sat_phi, self.sat_phi)
        v_sp = self.vel_ctl.get(tself, t) if self.control_vel else 10
        return [phi_sp, v_sp]

    

