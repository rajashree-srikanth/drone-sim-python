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

# sample trajectory - just for testing
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
        
""" 
Computing Differential Flatness 

Inputs - 
Class Initialization:
      - the wind speed (in m/s) 
Computing Flatness fn:
      - time instant t
      - the trajectory required to be tracked [xref,yref]
            - and their derivatives (1,2,3)       
      - the aircraft dynamical equations
    
Output 
      - the reference state and reference control input - Xr, Ur

"""
class DiffFlatness:
    def __init__(self, w=[0,0]):
        # self.traj = traj
        self.w = w # wind velocity
        self.g = 9.81 # acceleration due to gravity
        self.x_i = 0 # index location of pos x in state X
        self.y_i = 1 # index location of pos y in state X
        self.psi_i = 2 # index location of heading psi in state X
        self.phi_i = 3 # index location of bank phi in state X
        self.v_i = 4 # index location of airspeed v in state X
        
    # expressing states in terms of (x,y) and their derivatives       
    def ComputeFlatness (self,t, Y_ref, Yd_ref, Ydd_ref, Yddd_ref): 
        X, U = np.zeros(5), np.zeros(2)
        # Y_ref, Yd_ref, Ydd_ref, Yddd_ref = self.traj.TrajPoints(t)
        
        # extracting values 
        x, y = Y_ref[0], Y_ref[1]
        x_dot, y_dot = Yd_ref[0], Yd_ref[1]
        x_ddot, y_ddot = Ydd_ref[0], Ydd_ref[1]
        x_dddot, y_dddot = Yddd_ref[0], Yddd_ref[1]
        
        # velocity components
        v_ax = x_dot - self.w[0]
        v_ay = y_dot - self.w[1]
        v2 = v_ax**2 + v_ay**2
        
        # computing basic derivatives
        w_dot = [0,0] # assuming wind vel remains constant with time
        w_ddot = [0,0] # assuming wind vel remains constant with time
        v_ax_dot = x_ddot - w_dot[0]
        v_ay_dot = y_ddot - w_dot[1]
        v_ax_ddot = x_dddot - w_ddot[0]
        v_ay_ddot = y_dddot - w_ddot[1]
        
        # computing states in terms of x,y and derivatives
        X[self.x_i], X[self.y_i] = x, y
        X[self.psi_i] = np.arctan2(v_ay, v_ax) # psi
        X[self.phi_i] = np.arctan2((v_ax*v_ay_dot - v_ay*v_ax_dot), self.g*np.sqrt(v2)) # phi
        X[self.v_i] = np.sqrt(v2) # airspeed v
        
        # computing state derivative(1) in terms of x,y and derivatives
        psi_dot = (v_ay_dot*v_ax - v_ax_dot*v_ay)/(v2)
        va_dot = (v_ax*v_ax_dot + v_ay*v_ay_dot)/np.sqrt(v2)
        
        c1 = 1 + ((v_ax*v_ax_dot - v_ax_dot*v_ay)**2)/v2
        c2 = v_ax * v_ay_ddot + v_ax_dot*v_ay_dot - v_ax_ddot*v_ay - v_ax_dot*v_ay_dot
        c3 = (v_ax*v_ax_dot + v_ay*v_ay_dot)*(v_ax*v_ay_dot - v_ax_dot*v_ay)
        phi_dot = (1/c1)*(1/v2)*(c2*np.sqrt(v2) - c3/np.sqrt(v2))
        # print(phi_dot) - always almost 0 (order 10e-16)
        
        # computing input u in terms of (x,y)
        ac = ddyn.Aircraft()
        U[0] = ac.tau_phi * phi_dot + X[self.phi_i] # U_phi
        U[1] = ac.tau_v * va_dot + X[self.v_i] # U_va
        
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
        self.v_min, self.v_max = 9, 15
        # self.v_min, self.v_max = 4, 20 # for main simulation
        # self.phi_lim = np.deg2rad(60)
        self.phi_lim = np.deg2rad(45)
        
        # controller parameters
        self.Q, self.R = [1, 1, 0.1, 0.01, 0.01], [8, 1] # full state feedback
        # self.Q, self.R = np.dot(20,[1, 1, 0.1, 0.01, 0.01]), np.dot(10,[8, 1]) # full state feedback
        self.K = []
        
    def RestrictAngle(self, theta): # ensures angle limits between -pi and pi
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        return theta
    
    def ComputeGain(self, t, X, Y_ref, Yd_ref, Ydd_ref, Yddd_ref, ac):
        # Xr, Ur are the reference states to be tracked for the required trajectory
        Xr, Ur = self.DF.ComputeFlatness(t, Y_ref, Yd_ref, Ydd_ref, Yddd_ref)
        dX = X-Xr
        
        # ensuring angle limits for dpsi and dpsi angles (in rad)
        dX[self.psi_i] = self.RestrictAngle(dX[self.psi_i])  
        dX[self.phi_i] = self.RestrictAngle(dX[self.phi_i]) # ??
        
        # imposing saturation limits on state error
        dX = np.clip(dX, -self.err_sats, self.err_sats) 
        A, B = ac.cont_jac(Xr, Ur, t, self.w)
        # print(A, B)
        
        if 1: # Full state feedback
            K, S, E = control.lqr(A, B, np.diag(self.Q), np.diag(self.R))
        if 0: # is this even needed?
            A1,B1 = A[:3,:3], A[:3,3:] 
            Q, R = [1, 1, 0.1], [8, 1]
            (K1, X, E) = control.lqr(A1, B1, np.diag(Q), np.diag(R))
            K=np.zeros((2,5))
            K[:,:3]=K1
        self.K.append(K)

        U = Ur - np.dot(K, dX)
        plt.figure(10)
        # breakpoint()
        plt.plot(t, np.rad2deg(dX[3]), ".k")
        # U = - np.dot(K, dX)
        # specifying saturation limits to control inputs
        U = np.clip(U, [-self.phi_lim, self.v_min], [self.phi_lim, self.v_max])
        return Xr, dX, U