import numpy as np, scipy.integrate


import matplotlib.pyplot as plt


import d2d.trajectory as ddt
from d2d.trajectory import Trajectory
import d2d.dynamic as ddyn
from d2d.dynamic import Aircraft


def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

class WindField:
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
    def __init__(self, traj, ac, wind, record=True):
        self.traj, self.ac, self.wind = traj, ac, wind
        self.dt = 0.01
        self.time = np.arange(0, traj.duration, self.dt)
        self.carrot, self.ref_pos = [0,0], [0,0]
        self.record = record
        if self.record:
            self.t, self.X, self.Xref, self.K, self.U, self.fb_poles = [], [], [], [], [], []
        self.disable_feedback = False
            
    def get(self, X, t):
        _X = np.array(X)
        #print(f'{_X[0]:.1f} {_X[1]:.1f} {np.rad2deg(_X[2]):.1f}')
        Yref = self.traj.get(t)
        W = self.wind.sample(t, Yref[0])
        Xr, Ur, Xrdot = DiffFlatness.state_and_input_from_output(Yref, W, self.ac)
        U = Ur
        dX = X - Xr
        dX[Aircraft.s_psi] = norm_mpi_pi(dX[Aircraft.s_psi])
        #print(X[Aircraft.s_psi], dX[Aircraft.s_psi])
        err_sats = np.array([20, 20 , np.pi/3, np.pi/4, 1])
        dX = np.clip(dX, -err_sats, err_sats)
        A, B = self.ac.cont_jac(Xr, Ur, t, W)
        if 0: # dim 5 feedback
            Q, R = [1, 1, 0.1, 0.01, 0.01,], [8, 1]
            (K, __X, E) = control.lqr(A, B, np.diag(Q), np.diag(R))
            cl_poles, cl_vp =  np.linalg.eig(A-np.dot(B, K))
        if 1: # dim 3 feedback
            A1,B1 = A[:3,:3], A[:3,3:]
            #Q, R = [1, 1, 0.1], [2, 1]
            #Q, R = [1., 1., 20.], [200, 1000]
            Q, R = [1., 1., 20.], [500, 2000]
            (K1, __X, E) = control.lqr(A1, B1, np.diag(Q), np.diag(R))
            K=np.zeros((2,5))
            K[:,:3]=K1
            cl_poles, cl_vp =  np.linalg.eig(A1-np.dot(B1, K1))
            cl_poles, cl_vp =  np.linalg.eig(A-np.dot(B, K))
        if 0: # debuging
            vr, psir, phir = Xr[Aircraft.s_va], Xr[Aircraft.s_psi], Xr[Aircraft.s_phi] 
            A2 = np.array([[0, 0, -vr*np.sin(psir)], [0, 0, vr*np.cos(psir)], [0, 0, 0]])
            B2 = np.array([[0], [0], [9.81/vr/(1+np.cos(phir)**2)]])
            #print(A2, B2)
            #print(np.linalg.eig(A2))
            #Q, R = [1, 1, 0.1], [1]
            #breakpoint()
            #(K2, __X, E) = control.lqr(A2, B2, np.diag(Q), np.diag(R))
            K2 = np.array(control.place(A2, B2, [-1, -2, -3]))
            cl_poles, cl_vp =  np.linalg.eig(A2-np.dot(B2, K2))
            K=np.zeros((2,5))
            K[0,:3]=K2
            #cl_poles, cl_vp =  np.linalg.eig(A-np.dot(B, K))
        if 0:
            Kpsi = 0.1
            K=np.zeros((2,5))
            K[0,2]= Kpsi
            cl_poles, cl_vp =  np.linalg.eig(A-np.dot(B, K))
            
        dU = -np.dot(K, dX)
        if not self.disable_feedback: U += dU
        #phisat, vmin, vmax = np.deg2rad(30), 9, 15
        phisat, vmin, vmax = np.deg2rad(45), 9, 15
        U = np.clip(U, [-phisat, vmin], [phisat, vmax])
        #print(valp2, K, U1, U)
        if self.record:
            self.t.append(t)
            self.X.append(X)
            self.Xref.append(Xr)
            self.K.append(K)
            self.U.append(U)
            self.fb_poles.append(cl_poles)
        return U

    def draw_debug(self, _f=None, _a=None):
        #Xref = np.array(self.Xref)
        #_a[0,0].plot(time, Xref[:,0])
        _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
        _a = _f.subplots(5, 1) if _a is None else _a
        for _p in self.fb_poles:
            _a[0].plot(_p.real, _p.imag, '.')
        _a[0].set_title('cl poles')
        dts = np.diff(self.t)
        _a[1].hist(dts)
        _a[1].set_title('timing')

        K = np.array(self.K)
        _a[2].plot(K[:,0,0])
        _a[2].plot(K[:,0,1])
        _a[2].set_title('gains')

        U = np.array(self.U)
        _a[3].plot(np.rad2deg(U[:,0]))
        _a[4].plot(U[:,1])

        
#breakpoint()
        















#
#
#  old stuff, initial 2D pure pursuit
#


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
    def __init__(self, traj, record=True):
        self.traj = traj
        dt = 0.01
        self.time = np.arange(0, traj.duration, dt)
        self.pts_2d = np.array([traj.get(t)[0] for t in self.time])
        self.vel_2d = np.array([traj.get(t)[1] for t in self.time])
        self.sat_phi = np.deg2rad(45.)
        self.ref_pos, self.carrot = [], []
        self.vel_ctl = VelControler()
        self.control_vel = False#True
        self.record = record
        if self.record:
            self.t, self.X, self.Xref, self.K, self.U = [], [], [], [], []
        print('dfctl init')
        
    def get(self, X, t):
        dists = np.linalg.norm(self.pts_2d-X[:ddyn.Aircraft.s_y+1], axis=1)
        idx_closest = np.argmin(dists)
        self.ref_pos.append(self.pts_2d[idx_closest])
        t0, t1 = self.time[idx_closest], t
        tself = self.time[idx_closest]

        lookahead_m, dt, v = 20, 0.01, 10 # lookahaead in m, dt in sec, v in m/s
        #idx_carrot = min(idx_closest + int(lookahead_m/v/dt), len(self.pts_2d)-1)
        idx_carrot = idx_closest + int(lookahead_m/v/dt)
        if idx_carrot >= len(self.pts_2d):
            idx_carrot -= len(self.pts_2d)
        carrot = self.pts_2d[idx_carrot]
        xref, yref = self.pts_2d[idx_closest][0], self.pts_2d[idx_closest][1]
        psiref =  np.arctan2(self.vel_2d[idx_closest][1], self.vel_2d[idx_closest][0])
        phi, v = 0., 12.
        Xref = [xref, yref, psiref, phi, v]

        self.carrot.append(carrot)
        pc = carrot-X[ddyn.Aircraft.s_slice_pos]
        err_psi = norm_mpi_pi(X[2] - np.arctan2(pc[1], pc[0]))
        K= 0.2 #1.#0.2
        phi_sp = -K*err_psi
        phi_sp = np.clip(phi_sp, -self.sat_phi, self.sat_phi)
        v_sp = self.vel_ctl.get(tself, t) if self.control_vel else 12.
        U = [phi_sp, v_sp]
        if self.record:
            self.t.append(t)
            self.X.append(X)
            #print(X, self.X)
            self.Xref.append(Xref)
            self.K.append(K)
            self.U.append(U)
        return  U

    

