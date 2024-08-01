# attempt to implement diff controller
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

import d2d.guidance as ddg
import d2d.dynamic as ddyn
import d2d.trajectory as dtraj
from mpl_toolkits import mplot3d

import control

# creating a differential flatness controller
""" 
Input - the trajectory required to be tracked 
            - and its derivatives (1,2,3)
      - the wind speed (in m/s)        
      - the aircraft dynamical equations
      - the no. of aircraft that are in formation
    
Output - the control input 
"""
class CircleTraj:
    def __init__(self, v, r=40, c = [0,0] ):
        # v is flight velocity/airspeed
        self.c = c
        self.r = r
        self.v = v
        self.omega = self.v/self.r
    
    # computing ref traj position and their derivatives for a time instant t
    def TrajPoints(self, t):
        theta_ref = self.omega*t
        Y_ref = [self.r*np.cos(theta_ref), self.r*np.sin(theta_ref)]
        Yd_ref = [-self.r*np.sin(theta_ref)*self.omega, self.r*np.cos(theta_ref)*self.omega]
        Ydd_ref = [-self.r*np.cos(theta_ref)*self.omega**2, -self.r*np.sin(theta_ref)*self.omega**2]
        Yddd_ref = [self.r*np.sin(theta_ref), -self.r*np.cos(theta_ref)]
        return Y_ref, Yd_ref, Ydd_ref, Yddd_ref
        
class DiffFlatness:
    def __init__(self, w=[0,0]):
        # self.traj = traj
        self.w = w # wind velocity
        # self.n_ac = n_ac
        self.g = 9.81 # acceleration due to gravity
        
    # expressing states in terms of (x,y) and their derivatives       
    def ComputeFlatness (self,t, Y_ref, Yd_ref, Ydd_ref, Yddd_ref): 
        X, U = np.zeros(5), np.zeros(2)
        # Y_ref, Yd_ref, Ydd_ref, Yddd_ref = self.traj.TrajPoints(t)
        x, y = Y_ref[0], Y_ref[1]
        x_dot, y_dot = Yd_ref[0], Yd_ref[1]
        x_ddot, y_ddot = Ydd_ref[0], Ydd_ref[1]
        x_dddot, y_dddot = Yddd_ref[0], Yddd_ref[1]
        v_ax = x_dot - self.w[0]
        v_ay = y_dot - self.w[1]
        w_dot = [0,0] # assuming wind vel remains constant with time
        v_ax_dot = x_ddot - w_dot[0]
        v_ay_dot = y_ddot - w_dot[1]
        v_ax_ddot = x_dddot 
        v_ay_ddot = y_dddot
        v2 = v_ax**2 + v_ay**2
        X[0], X[1] = x, y
        X[2] = np.arctan2(v_ay, v_ax) # psi
        X[3] = np.arctan2((v_ax*v_ay_dot - v_ay*v_ax_dot), self.g*np.sqrt(v2)) # phi
        X[4] = np.sqrt(v2) # airspeed v
        # computing input u in terms of (x,y)
        psi_dot = (v_ay_dot*v_ax - v_ax_dot*v_ay)/(v2)
        va_dot = (v_ax*v_ax_dot + v_ay*v_ay_dot)/np.sqrt(v2)
        phi_dot = 1/(1+)
        # breakpoint()
        ac = ddyn.Aircraft()
        U[0] = ac.tau_phi * phi_dot + X[3] # U_phi
        U[1] = ac.tau_v * va_dot + X[4] # U_va
        # breakpoint()
        
        return X, U
    

"""

DiffController class :

    Uses reference state Xr and reference input Ur computed at a time instant t 
    using differential flatness.
    Computes the required gain K using any method (here, using LQR) that allows
    tracking of reference trajectory Xr through differential flatness
    The computation is performed by linearizing the non-linear dynamic model about 
    this reference trajectory and input - A, B matrices
    This is then used to obtain the corresponding LQR gain. 
    
    Arguments as input:
            reference trajectory (x,y) -> used to obtain Xr, Ur from DiffFlatness
            time instant - t
            aircraft dynamics - can be internal?
            current state - X = [x_pos, y_pos, psi, phi, vel]
            
    Internal Parameters:
            instance of aicraft dynamics class
            saturation limits on parameters
            Q, R weight matrices
            
    Output:
            Control Input - U = [U_phi, U_vel]

"""
        
class DiffController:
    def __init__(self, w=[0,0]):
        self.w = w # creating instance of Aircraft class - flight dynamics eqns
        self.DF = DiffFlatness(self.w) # creating instance of DiffFlatness class
        self.psi_i = 2 # index location of heading psi in state X
        self.phi_i = 3 # index location of bank phi in state X
        # limits on dX, vel, phi
        self.err_sats = np.array([20, 20 , np.pi/3, np.pi/4, 1]) 
        self.v_min, self.v_max = 4, 20
        self.phi_lim = np.deg2rad(45)
        # controller parameters
        self.Q, self.R = [1, 1, 0.1, 0.01, 0.01], [8, 1] # full state feedback
        self.K = []
        
    def RestrictAngle(self, theta): # ensures angle limits between 0 and 2*pi
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        return theta
    
    def ComputeGain(self, t, X, Y_ref, Yd_ref, Ydd_ref, Yddd_ref, ac):
        # Xr, Ur are the reference states to be tracked for the required trajectory
        Xr, Ur = self.DF.ComputeFlatness(t, Y_ref, Yd_ref, Ydd_ref)
        # ensuring angle limits for psi and psi angles (in rad)
        # Xr[self.psi_i] = self.RestrictAngle(Xr[self.psi_i]) 
        # Xr[self.phi_i] = self.RestrictAngle(Xr[self.phi_i]) 
        dX = X-Xr
        dX[self.psi_i] = self.RestrictAngle(dX[self.psi_i]) 
        # dX[self.phi_i] = self.RestrictAngle(dX[self.phi_i]) 
        # imposing saturation limits on state error
        dX = np.clip(dX, -self.err_sats, self.err_sats) 
        A, B = ac.cont_jac(Xr, Ur, t, self.w)
        # print(A, B)
        if 0:
            K, S, E = control.lqr(A, B, np.diag(self.Q), np.diag(self.R))
        else:
            A1,B1 = A[:3,:3], A[:3,3:] # ignoring phi and theta for feedback ?!
            Q, R = [1, 1, 0.1], [8, 1]
            (K1, X, E) = control.lqr(A1, B1, np.diag(Q), np.diag(R))
            K=np.zeros((2,5))
            # K[:,:3]=K1
        self.K.append(K)
        
        # breakpoint()
        U = Ur - np.dot(K, dX)
        # specifying saturation limits to control inputs
        # breakpoint()
        U = np.clip(U, [-self.phi_lim, self.v_min], [self.phi_lim, self.v_max])
        return Xr, - np.dot(K, dX), U
        
def main():
    # introducing parameters
    v = 10 # flight speed/airspeed
    w = [0, 0]
    windfield = ddg.WindField(w) # creating windfield object
    n_ac = 1
    dt = 0.01
    t0, tf = 0, 40
    time = np.arange(t0, tf, dt)
    aircraft = []
    for i in range(n_ac): # creating aircraft class instances 
        aircraft.append(ddyn.Aircraft())
    
    U_array = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no; 2 inputs
    Y_ref_array = np.zeros((len(time),n_ac,2))
    Yd_ref_array = np.zeros((len(time),n_ac,2))
    Ydd_ref_array = np.zeros((len(time),n_ac,2))
    X_array = np.zeros((len(time),n_ac,5))
    traj = CircleTraj(v, r=30)    
    # traj = dtraj.TrajectoryCircle()
    ctrl = DiffController(w)
    X1 = np.array([0,0,0,0,1]) # initial state conditions
    # breakpoint()
    # ac = ddyn.Aircraft()
    # i,j = 0, 0
    
    # for i in range(time):
    #     t = time[i]
    #     for j in range(aircraft):
    #         ac = aircraft[j]
    #         Y_ref, Yd_ref, Ydd_ref = traj.TrajPoints(t)
    #         U = controller.ComputeGain(Y_ref, Yd_ref, Ydd_ref, t, ac, X, w)
    #         X_new = ac.disc_dyn(X, U, w, t, dt)
    #         X = X_new
    #         X_array[i][j] = X
    
    for i in range(n_ac):
        X_array[0][i][:] = X1
    
    for i in range(1, len(time)):
        t = time[i-1]
        for j in range(n_ac):
            ac = aircraft[j]
            Y_ref, Yd_ref, Ydd_ref, Yddd_ref = traj.TrajPoints(t)
            # Y_ref, Yd_ref, Ydd_ref = traj.get(t)
            X = X_array[i-1, j, :]
            ctrl = DiffController(w)
            Xr, Ur, U = ctrl.ComputeGain(t, X, Y_ref, Yd_ref, Ydd_ref, ac)
            # print("reference", Xr, Ur)
            X_new = ac.disc_dyn(X, U, windfield, t, dt)
            X_array[i][j] = X_new
            U_array[i-1][j] = Ur
            Y_ref_array[i-1][j] = Y_ref
            Yd_ref_array[i-1][j] = Yd_ref
            Ydd_ref_array[i-1][j] = Ydd_ref
            
                 
    # t = 3
    # traj = CircleTraj(v)
    # Y_ref, Yd_ref, Ydd_ref = traj.TrajPoints(t=3)
    # print("ref traj values for time t = 2", Y_ref, Yd_ref, Ydd_ref)
    # flatness = DiffFlatness(w)
    # Xr, Ur = flatness.ComputeFlatness(2, Y_ref, Yd_ref, Ydd_ref)
    # print(Xr, Ur)
    # ctrl = DiffController(w)
    # ac = ddyn.Aircraft()
    # # breakpoint()
    # Xr, Ur, U = ctrl.ComputeGain(t, X, Y_ref, Yd_ref, Ydd_ref)
    # print("Controller", Xr, Ur, U)
    print(X_array)
    # print(np.shape(U_array))
    plt.figure(1)
    plt.plot(X_array[:,0,0], X_array[:,0,1])
    plt.plot(Y_ref_array[:, 0, 0], Y_ref_array[:, 0, 1], label='ref')
    plt.title("traj computed")
    plt.figure(2)
    plt.plot(time, X_array[:,0,0])
    plt.title("time, x computed")
    plt.figure(3)
    plt.plot(time, X_array[:,0,1])
    plt.title("time, y computed")
    plt.figure(4)
    plt.plot(time, U_array[:,0,0],label="Command dX phi")
    plt.plot(time, X_array[:,0,3],label="Measured")
    plt.legend()
    plt.title("uphi, t computed VS measured")
    plt.figure(5)
    plt.plot(time, U_array[:,0,1],label="Command dX v")
    plt.plot(time, X_array[:,0,4],label="Measured")
    plt.legend()
    plt.title("u_v computed computed VS measured")
    plt.figure(6)
    plt.plot(time, Y_ref_array[:, 0, 0], label="Yrefx")
    plt.plot(time, Yd_ref_array[:, 0, 0], label="Ydrefx")
    plt.plot(time, Ydd_ref_array[:, 0, 0], label="Yddrefx")
    plt.legend()
    plt.figure(7)
    plt.plot(time, Y_ref_array[:, 0, 1], label="Yrefy")
    plt.plot(time, Yd_ref_array[:, 0, 1], label="Ydrefy")
    plt.plot(time, Ydd_ref_array[:, 0, 1], label="Yddrefy")
    plt.legend()
    # plt.figure(8)
    # plt.plot(time, X_array[:, 0, 2], label='psi')
    # plt.legend()
    plt.show()
    
main()