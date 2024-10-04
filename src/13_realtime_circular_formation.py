#!/usr/bin/env python3
# before launching program, run these!
# export PAPARAZZI_HOME=/home/rajashree/paparazzi
# export PAPARAZZI_SRC=/home/rajashree/paparazzi

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

import d2d.guidance_circform as d2guid
import d2d.dynamic as d2dyn

import d2d.ploting as d2plot
def ConstructBMatrix(n_ac):
    B = np.zeros((n_ac, n_ac-1))
    for i in range(n_ac):
        for j in range(n_ac-1):
            if i==j:
                B[i][j] = -1
            elif (i<j) or (i>j+1):
                B[i][j] = 0
            elif i>j:
                B[i][j] = 1
    return B
    
def CircularFormationGVF(c, r, n_ac, t_end):
    
    # initializing parameters
    t_start, t_step = 0, 0.05
    time = np.arange(t_start, t_end, t_step)
    # n_ac = 1 # no. of aircraft in formation flight
    windfield = d2guid.WindField() # creating windfield object
    aircraft = []
    controllers = []
    
    # for changing centres - for now, n_ac = 4
    d = 30
    c_ = np.ones((n_ac, 2))*c
    # c_[0,:] += [-d, 0]
    # c_[1,:] += [0, d]
    # c_[2,:] += [d, 0]
    # c_[3,:] += [0, -d]
    c_[0,:] = [0,-20]
    c_[1,:] = [25,-40]
    c_[2,:] = [25,-80]
    c_[3,:] = [0,-100]
    # c_[0,:] = [-15,-30]
    # c_[1,:] = [-25,-90]
    # c_[2,:] = [0,-60]
    # c_[3,:] = [15,-30]
    # c_[4,:] = [15,-90]
    
    # U_array = np.zeros((len(time),n_ac)) # rows - time; columns - ac no
    # U1_array = np.zeros((len(time),n_ac)) # useful for debugging
    # U2_array = np.zeros((len(time),n_ac)) # useful for debuggung
    # Ur_array = np.zeros((len(time),n_ac))
    # e_theta_array = np.zeros((len(time),n_ac-1))
    # X_array = np.zeros((len(time),n_ac,5))
    
    # controller gains
    ke = 0.0004 # aggressiveness of the gvf guidance
    kd = 25 # speed of exponential convergence to required guidance trajectory
    kr = 20 # controls the speed of convergence to required phase separation
    R = r*np.ones((n_ac,1))
    
    X1 = np.array([20,30,-np.pi/2,0,10]) # initial state conditions
    p = np.zeros((2, n_ac)) # columns are the different aircrafts

    # building B matrix based on no_ac - assuming a straight formation
    # we're also building the initial position matrix p and state matrix X_array
    B = np.zeros((n_ac, n_ac-1))
    for i in range(n_ac):
        p[0][i] = X1[0]
        p[1][i] = X1[1]
        X_array[0][i][:] = X1
        for j in range(n_ac-1):
            if i==j:
                B[i][j] = -1
            elif (i<j) or (i>j+1):
                B[i][j] = 0
            elif i>j:
                B[i][j] = 1
        
    # z_des is a row matrix here, it is converted to column matrix later within the DCF class
    # z_des = np.array([np.deg2rad(180)]) 
    # z_des = np.ones(n_ac-1)*(np.pi*2/n_ac) # separation angles between adjacent aircraft are equal
    z_des = np.zeros(n_ac-1) # separation angles between adjacent aircraft are equal
    
    dcf = d2guid.DCFController()
    for i in range(n_ac): 
        aircraft.append(d2dyn.Aircraft()) # creating aircraft objects
        traj = d2guid.CircleTraj(c_[i,:]) # creating traj object
        # calling GVF controller
        controllers.append(d2guid.GVFcontroller(traj, aircraft[i], windfield))   
        
    for i in range(1, len(time)):
        t = time[i-1]
        U_r, e_theta = dcf.get(n_ac, B, c_, p, z_des, kr)
        Rr = U_r + R # new required radii to be tracked for each ac
        # breakpoint()
        # print(i, j)
        Ur_array[i] = Rr.T
        e_theta_array[i] = e_theta.T
        for j in range(n_ac):
            X = X_array[i-1,j,:]
            gvf, ac = controllers[j], aircraft[j] # object names
            r = Rr[j]
            traj = d2guid.CircleTraj(c_[j,:])
            e, n, H = traj.get(X,r) # obtaining required traj and their derivatives
            U, U1, U2 = gvf.get(X, ke, kd, e, n, H) # obtaining control input
            U = np.arctan(U/9.81) # roll setpoint angle in radians
            U_array[i-1][j] = U
            U1_array[i-1][j] = U1
            U2_array[i-1][j] = U2
            # new state from prev time step and required control action
            X_new = ac.disc_dyn(X, [U, 15], windfield, t, t_step) 
            X = X_new
            X_array[i][j] = X
            p[0][j] = X[0]
            p[1][j] = X[1]
        # breakpoint()
    
    return X_array, U_array, time, U1_array, U2_array, Ur_array, e_theta_array

        
class Controller:
    def __init__(self, conf, traj_id, ctl_id):
        self.conf = conf
        self.backend = d2pb.PprzBackend()
        self.aircraft_desc = self.backend.aircraft # dictionary containing ac id ac id as keys and all info about all
        # existing ac from pprz
        self.timestep = 1./conf['hz']
        #self.timestep = 1/(20)
        print('frequency', conf['hz'])
        self.scenario = None
        self.traj = None
        self.initialized = False
        self.ctl_id, self.traj_id = ctl_id, traj_id
        print(f'using trajectory {traj_id}')
        self.last_display, self.display_dt = None, 1./2.
        self.ac_id = 5
        

    def run(self):
        self.start = timer()
        self.last_display = 0.
        done = False
        try:
            
            while not done:
                time.sleep(self.timestep)
                now = timer(); elapsed = now - self.start
                self.step(elapsed)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        fallback_block_name = "Standby"
        for id in self.ac_dict:
            self.backend.jump_to_block_name(id, fallback_block_name)
        self.backend.publish_track(self.traj, 0., delete=True)
        self.backend.shutdown()
        # print(self.X_array)
        plt.show()
        
    def ParameterInit_CircleFormation(self):
        # indexing aircraft ids
        self.ac_dict = {}
        ac_count = 0
        print("indexing acs. details - ", self.aircraft_desc)
        for key in self.aircraft_desc:
            self.ac_dict[key] = ac_count
            ac_count+=1
        self.n_ac = len(self.ac_dict)
        
        # requirements
        self.c = np.array([0,0])
        self.c = np.ones((self.n_ac, 2))*self.c  # this will be useful for non-origin centre
        # print("old c", self.c)
        # self.c[0,:], self.c[1,:] = [0,0], [0,20]
        # print("new c", self.c)
        self.c[0,:], self.c[1,:], self.c[2,:], self.c[3,:] = [0,-20], [50,-20], [50,-100], [0,-100]
        self.r = 60
        self.v = 15 # flight speed
        self.z_des = np.zeros(self.n_ac-1) # separation angles between adjacent aircraft are equal
        
        # controller parameters
        self.ke = 0.0001
        self.kd = 5
        self.kr = 20
        self.R = self.r*np.ones((self.n_ac,1))
        
        # storing
        # self.X_array = np.zeros((1,self.n_ac, 5))
        # self.U_array = np.zeros((1,self.n_ac))
        # self.time_array = np.zeros(1)
        
        # instantiating classes
        self.dcf = d2guid.DCFController()
        self.windfield = d2guid.WindField() # creating windfield object
        self.controllers = []
        for i in range(self.n_ac): 
            self.traj = d2guid.CircleTraj() # creating traj object
            # calling GVF controller
            self.controllers.append(d2guid.GVFcontroller(self.traj, self.windfield))
        self.B = ConstructBMatrix(self.n_ac)
        self.p = np.zeros((2,self.n_ac)) # stores instantaneous position of all aircraft

    def gvf_implementation(self, j, id, Rr):
        print("gvf computation")
        X = self.backend.aircraft[id].get_state()
        # X_list = np.array(X)
        # self.temp.append(X_list[:, np.newaxis])
        # print(f'state for ac id {id} is {X} for gvf comp')
        gvf= self.controllers[j] # object names
        self.r = Rr[j]
        e, n, H = self.traj.get(X,self.r, self.c[j,:]) # obtaining required traj and their derivatives
        U_phi, U1, U2 = gvf.get(X, self.ke, self.kd, e, n, H) # obtaining control input
        U_phi = np.arctan(U_phi/9.81) # roll setpoint angle in radians
        return U_phi
    
    def step(self, t):
        # try:
        #     X = [x, y, psi, phi, v] = self.backend.aircraft[self.ac_id].get_state()
        #     t = timer() - self.start 
        #     plt.figure(1)
        #     plt.plot(x,y,"k.")
        #     plt.title("x and y pos (in m)")
        #     plt.figure(2)
        #     plt.plot(t, X[0],'r.')
        #     plt.title("x pos (in m)")
        #     plt.figure(3)
        #     plt.plot(t, X[1],'r.')
        #     plt.title("y pos (in m)")
        #     # print("states",X)
        # except AttributeError:  # aircraft not initialized
        #     print("fail",self.ac_id)
        #     return
        if not self.initialized and self.backend.nav_initialized:
            # Trajectory and control initialization
            # TODO
            print("initializing parameters")
            self.ParameterInit_CircleFormation()
            print("initialized")
            
            for id in self.ac_dict:
                ext_guid_block_name = "Joystick"
                self.backend.jump_to_block_name(id, ext_guid_block_name)
                self.initialized = True
            
            print('trajectory computed, starting control')
            #self.backend.publish_track(self.traj, t, full=True)
        if self.initialized:
            # Control
            print("in self.init block")
            U = 0. # TODO DCF
            i=0
            for id in self.ac_dict: # assigns/updates inst pos of all ac into p
                X1 = self.backend.aircraft[id].get_state()
                self.p[0][i], self.p[1][i] =  X1[0], X1[1]
                # print(f'ac id {id} state {X1} while init p')
                i+=1
            # solution of DCF alg
            Ur, e_theta = self.dcf.get(self.n_ac, self.B, self.c, self.p, self.z_des, self.kr)
            Rr = Ur + self.R # new required radii to be tracked for each ac
            # gvf alg
            j=0
            # self.temp = []
            for id in self.ac_dict:
                U = self.gvf_implementation(j, id, Rr)    
                if id==4:
                    t = timer() - self.start 
                    plt.figure(4)
                    plt.plot(t, -np.rad2deg(U),"b.")
                    plt.title("control input (in deg)")
                    # plt.figure(5)
                    plt.plot(t, e_theta[0],'k.')
                    # plt.title("e_theta (in deg)")
                    plt.figure(6)
                    plt.plot(t, Rr[0], "k.")
                    plt.title("traacking radius")
                # print(f'roll command input {U}')
                if np.isfinite(U):
                    self.backend.send_command(id, -np.rad2deg(U), 15)
                j+=1
            # self.X_array=np.append(self.X_array, [np.vstack(self.temp)], axis=0)
        #if t > self.last_display + self.display_dt:
        #    self.last_display += self.display_dt
        #    self.backend.publish_track(self.traj, t, full=False)

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
