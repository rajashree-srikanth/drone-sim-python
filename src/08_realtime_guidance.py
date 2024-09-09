#!/usr/bin/env python3

import sys, time
import numpy as np
import json

import matplotlib.pyplot as plt
from timeit import default_timer as timer

import d2d.paparazzi_backend as d2pb

import d2d.trajectory as d2traj
import d2d.trajectory_factory as d2trajfact

import d2d.opty_utils as d2ou
import d2d.opty_planner as d2op

import d2d.guidance as d2guid
import d2d.dynamic as d2dyn

import d2d.ploting as d2plot

class Scenario:
    def __init__(self, p0, p1, dt, v):
        self.tol, self.max_iter = 1e-5, 500
        self.t0, self.p0 = 0., p0
        self.t1, self.p1 =  dt, p1
        self.vref = v
        self.cost, self.obj_scale = d2ou.CostInput(vsp=self.vref, kvel=50., kbank=1.), 1#.e-2
        self.cost, self.obj_scale = d2ou.CostAirVel(self.vref), 1
        self.wind = d2ou.WindField(w=[0.,0.])
        self.x_constraint, self.y_constraint = None, None
        #x_constraint, y_constraint = (-5., 100.), (-50., 50.) 
        self.phi_constraint = (-np.deg2rad(30.), np.deg2rad(30.))
        self.v_constraint = (v-1.1, v+1.1)
        self.obstacles = []
        self.hz = 25. #50.
        self.name = 'exp0'
        self.desc = 'test paparazzi'

def get_trajectory(idx, x, y, psi, phi, vel):
    GOAL = np.array([0., 200., np.pi, 0., 12.])

    if idx == 0: # circle
        traj = d2traj.TrajectoryCircle( c=[-50, 0],  r=80., v=vel)
    elif idx == 1: # line
        dt, dpsi, dl = 40., np.deg2rad(10), 25.
        dx, dy = vel*dt*np.array([np.cos(psi+dpsi), np.sin(psi+dpsi)])
        dx2, dy2 = dl*np.array([np.sin(psi+dpsi), np.cos(psi+dpsi)])
        traj = d2traj.TrajectoryLine((x+dx2,y+dy2), (x+dx+dx2, y+dy+dy2))
    elif idx == 2:
        #dt = np.hypot((x, y))/v
        traj = d2traj.TrajectoryLine((x,y), (0, 0))
    elif idx==3:
        traj = d2trajfact.TrajSquare()
    elif idx == 4:
        p0 = [x, y, psi, phi, vel]
        dt = 40.
        dx, dy = 0.9*vel*dt*np.array([np.cos(psi), np.sin(psi)])
        p1 = [x+dx, y+dy, psi, 0., vel]
        scenario = Scenario(p0, p1, dt, vel)
        planner = d2op.Planner(scenario, initialize=True)
        initial_guess = planner.get_initial_guess()
        planner.run(initial_guess)
        planner.save_solution('/tmp/foo.npz')
        traj = d2trajfact.TrajTabulated('/tmp/foo.npz')
    elif idx == 5:
        p0 = [x, y, psi, phi, vel]
        dt = 40.
        p1 = GOAL
        print('GOAL', GOAL)
        scenario = Scenario(p0, p1, dt, vel)
        planner = d2op.Planner(scenario, initialize=True)
        planner.configure(tol=scenario.tol, max_iter=scenario.max_iter)
        initial_guess = planner.get_initial_guess()
        planner.run(initial_guess)
        print('run done')
        planner.save_solution('/tmp/foo.npz')
        traj = d2trajfact.TrajTabulated('/tmp/foo.npz')
        #d2ou.plot2d(self.planner, None)
        #plt.show()
    elif idx == 6: # circle to goal
        p0 = np.array([x, y, psi, phi, vel])
        p1 = GOAL
        p0p1 = p1[:2]-p0[:2];
        n = np.linalg.norm(p0p1)
        dt = 30.
        dist_to_travel = vel*dt
        if dist_to_travel < n: # we're too slow
            print(f"traj 6: can't reach goal at {dist_to_travel} with speed {vel} in {dt}") 
            return None
        else:
            d = 0.5*n; u = p0p1/n; v = np.array([u[1], -u[0]])
            print(d, u, v, dist_to_travel)
            p2 = p0[:2] + d*u
            Rs = np.arange(d, 500, 5)
            err_dist = np.array([2*_R * np.arcsin(d/_R) - dist_to_travel for _R in Rs])
            idx1 = np.argmin(err_dist**2)
            print(f'R: {Rs[idx1]} err {err_dist[idx1]}')
            #plt.plot(Rs, err_dist)
            #plt.show()
            R = Rs[idx1]
            p3 = p2 + np.sqrt(R**2-d**2)*v
            alpha = np.arcsin(d/R)
            p3p0 = p0[:2]-p3
            alpha0 = np.arctan2(p3p0[1], p3p0[0])
            #breakpoint()
            print(p3, R, np.rad2deg(alpha0), np.rad2deg(2*alpha))
            #breakpoint()
            traj = d2traj.TrajectoryCircle(c=p3, r=R, v=vel, t0=0., alpha0=alpha0, dalpha=2*alpha)
            #traj = d2traj.TrajectoryCircle( c=[-50, 0],  r=80., v=vel)
            print(traj.duration)
    elif idx==7: # triangle
        p0 = np.array([x, y, psi, phi, vel])
        p1 = GOAL
        traj = d2trajfact.Triangle(p0, p1, va=vel, duration=40., cw=1)

        
    return traj

        
        
class Controller:
    def __init__(self, conf, traj_id, ctl_id):
        self.conf = conf
        self.backend = d2pb.PprzBackend()
        self.timestep = 1./conf['hz']
        self.scenario = None
        self.traj = None
        self.initialized = False
        self.ctl_id, self.traj_id = ctl_id, traj_id
        print(f'using trajectory {traj_id}')
        self.last_display, self.display_dt = None, 1./2.
        self.ac_id = 10

    def run(self):
        start = timer()
        self.last_display = 0.
        done = False
        try:
            while not done:
                time.sleep(self.timestep)
                now = timer(); elapsed = now - start
                self.step(elapsed)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        fallback_block_name = "Standby"
        self.backend.jump_to_block_name(self.ac_id, fallback_block_name)
        self.backend.publish_track(self.traj, 0., delete=True)
        self.backend.shutdown()

    def step(self, t):
        try:
            X = [x, y, psi, phi, v] = self.backend.aircraft[self.ac_id].get_state()
        except AttributeError:  # aircraft not initialized
            return
        if not self.initialized and self.backend.nav_initialized:
            # Trajectory and control initialization
            print(f'Computing trajectory at {x}, {y}, {psi}, {phi}, {v}')
            self.traj = get_trajectory(self.traj_id, x, y, psi, phi, v)
            if self.ctl_id == 0:
                self.ctl = d2guid.PurePursuitControler(self.traj)
            else:
                ac , wind = d2dyn.Aircraft(), d2guid.WindField([0., 0.])
                self.ctl = d2guid.DFFFController(self.traj, ac, wind)
            ext_guid_block_name = "Ext Guidance"
            self.backend.jump_to_block_name(self.ac_id, ext_guid_block_name)
            self.initialized = True
            print('trajectory computed, starting control')
            self.backend.publish_track(self.traj, t, full=True)
        if self.initialized:
            # Control
            U = self.ctl.get(X, t)
            self.backend.send_command(self.ac_id, -np.rad2deg(U[0]), U[1])
        if t > self.last_display + self.display_dt:
            self.last_display += self.display_dt
            self.backend.publish_track(self.traj, t, full=False)

def plot_pprz_debug(ctl):
    plt.figure()
    Xs, ts = np.array(ctl.backend.aircraft[4].Xs), np.array(ctl.backend.aircraft[4].ts)
    plt.plot(ts, Xs[:,0], '.')
            
def plot(ctl):
    t, X, Xr, U = np.array(ctl.ctl.t), np.array(ctl.ctl.X), np.array(ctl.ctl.Xref), np.array(ctl.ctl.U)#, np.array(ctl.ctl.Yr)
    d2plot.plot_trajectory_chrono(t, X)
    d2plot.plot_control_chrono(t, X=X, U=U, Yref=None, Xref=Xr)
    ctl.ctl.draw_debug()
    plot_pprz_debug(ctl)
    plt.show()
        
def main(args):
    # with open(args.config_file, 'r') as f:
    #     conf = json.load(f)
    #     if args.verbose:
    #         print(json.dumps(conf))
    conf = {'hz':10}
    ctl_id = 1
    traj_id = int(args.traj)
    c = Controller(conf, traj_id, ctl_id)
    c.run()
    if args.plot: xplot(c)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="3dplust Guidance")
    #parser.add_argument('config_file', help="JSON configuration file")
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true', help="display debug messages")
    parser.add_argument('--traj', help='trajectory index', default=0)
    parser.add_argument('--plot', help='trajectory index', default=False)
    args = parser.parse_args()
    main(args)
