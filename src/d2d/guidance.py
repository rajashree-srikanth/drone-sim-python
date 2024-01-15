import numpy as np, scipy.integrate


import d2d.trajectory as ddt
from d2d.trajectory import Trajectory
import d2d.dynamic as ddyn



def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

class WindField:
    def __init__(self, w=[0.,0]):
        self.w = w
    def sample(self, t, loc):
        return self.w

 
class DiffFlatness:
    def state_and_input_from_output(Ys, W):
        X, U = [np.zeros(_l) for _l in [ddyn.Aircraft.s_size, ddyn.Aircraft.i_size]]
        X[ddyn.Aircraft.s_x], X[ddyn.Aircraft.s_y] = Ys[0, ddt.Trajectory.cx], Ys[0, ddt.Trajectory.cy]
        xd, yd = Ys[1,Trajectory.cx], Ys[1,Trajectory.cy]
        xda, yda = xd - W[0], yd - W[1]   # airspeed
        v = np.sqrt(xda**2+yda**2)
        X[ddyn.Aircraft.s_v] = v
        invv = 1./v                       # heading
        psi = np.arctan2(invv*yd, invv*xd)
        X[ddyn.Aircraft.s_psi] = psi
        ## TODO
        if 0:
            yda_ov_xda =  yda / xda           # roll
            invv_ydaovxda = invv*yda_ov_xda
            invv_ydaovxda_dot = 1.
            psid = invv_ydaovxda_dot / (1.+invv_ydaovxda**2)
            g = 9.81
            phi = np.arctan2(v*psid, g)
            X[ddyn.Aircraft.s_phi] = 0. #phi
            # rolldot
            # vdot
            # roll_sp, v_sp
        return X, U


class VelControler:
    def get(self, tself, tref):
        dt = tself - tref
        #print(f'{tself:.1f}, {tref:.1f} {dt:.1f}')
        v = 10. -np.clip(2. * dt, -3., 3.) # m/s
        return v
    
class PurePursuitControler:
    def __init__(self, traj):
        self.traj = traj
        t0, t1, dt = 0, 60, 0.01 # FIXME
        self.time = np.arange(t0, t1, dt)
        self.pts_2d = np.array([traj.get(t)[0] for t in self.time])
        self.sat_phi = np.deg2rad(45.)
        self.ref_pos, self.carrot = [], []
        self.vel_ctl = VelControler()
        
    def get(self, X, t):#, time):
        dists = np.linalg.norm(self.pts_2d-X[:ddyn.Aircraft.s_psi], axis=1)
        idx_closest = np.argmin(dists)
        self.ref_pos.append(self.pts_2d[idx_closest])
        t0, t1 = self.time[idx_closest], t
        #print(f'{t0:.1f}, {t1:.1f} {t0 - t1:.1f}')
        tself = self.time[idx_closest]
        if 1:
            lookahead_m, dt, v = 10, 0.01, 10 # lookahaead in m, dt in sec, v in m/s
            idx_carrot = min(idx_closest + int(lookahead_m/v/dt), len(self.pts_2d)-1)
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
        v_sp = 10. #self.vel_ctl.get(tself, t)
        return [phi_sp, v_sp]

    
