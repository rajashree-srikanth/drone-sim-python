#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# main script for running circular formation control

import argparse
import numpy as np
import matplotlib.pyplot as plt

import d2d.dynamic as ddyn
import d2d.guidance as ddg
import d2d.utils as du
import d2d.ploting as d2plot
import d2d.animation as dda
import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.scenario as dds

def CircularFormationGVF(c, r, n_ac):
    
    # initializing parameters
    t_start, t_step, t_end = 0, 0.01, 120
    time = np.arange(t_start, t_end, t_step)
    # n_ac = 1 # no. of aircraft in formation flight
    windfield = ddg.WindField() # creating windfield object
    aircraft = []
    controllers = []
    
    U_array = np.zeros((len(time),n_ac)) # rows - time; columns - ac no
    U1_array = np.zeros((len(time),n_ac)) # useful for debugging
    U2_array = np.zeros((len(time),n_ac)) # useful for debuggung
    X_array = np.zeros((len(time),n_ac,5))  
    
    ke = 0.005
    kd = 10
    kr = 1
    R = r*np.ones((n_ac,1))
    
    X1 = np.array([30,90,-np.pi/2,0,10]) # initial state conditions
    X2 = np.array([20,50,-np.pi/2,0,10]) # initial state conditions
    p0  = np.array([[X1[0], X1[1]], [X2[0], X2[1]]]).T # initial coordinates
    p = p0
    X_init = [X1, X2]
    B = np.array([[-1], [1]])
    z_des = np.array([np.pi]) 
    
    X_array[0][0][:] = X1
    X_array[0][1][:] = X2
    
    dcf = ddg.DCFController()
    traj = ddg.CircleTraj(c) # creating traj object
    for i in range(n_ac): 
        aircraft.append(ddyn.Aircraft()) # creating aircraft objects
        # calling GVF controller
        controllers.append(ddg.GVFcontroller(traj, aircraft[i], windfield))   
    
    for t in time:
        U_r = dcf.get(n_ac, B, c, p, z_des, kr)
        R = U_r + R # new required radii to be tracked for each ac
        for j in range(n_ac):
            X = X_array[i,j,:]
            gvf, ac = controllers[j], aircraft[j] # object names
            r = R[j]
            e, n, H = traj.get(X,r) # obtaining required traj and their derivatives
            U, U1, U2 = gvf.get(X, ke, kd, e, n, H) # obtaining control input
            U = np.arctan(U/9.81) # roll setpoint angle in radians
            # breakpoint()
            U_array[i][j] = U
            U1_array[i][j] = U1
            U2_array[i][j] = U2
            # new state from prev time step and required control action
            X_new = ac.disc_dyn(X, [U, 15], windfield, t, t_step) 
            X = X_new
            X_array[i+1][j] = X
            p[j][:] = X[0], X[1]
        i = i+1
        p = p.T # transposing to get p in required format for dcf computation
    return X_array, U_array, time, U1_array, U2_array

def plotting(X_array, U_array, U1, U2, Y_ref, time):
        plt.gca().set_aspect('equal')
        plt.figure(1)
        plt.plot(Y_ref[1][:], Y_ref[0][:])
        plt.plot(X_array[:,0,0], X_array[:,0,1])
        plt.plot(X_array[:,1,0], X_array[:,1,1])
        plt.title('XY Trajectory vs reference')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.figure(2)
        plt.plot(time, np.degrees(U_array[:,0]))
        plt.plot(time, np.degrees(U_array[:,1]))
        plt.title('Control Input')
        plt.xlabel("time (s)")
        plt.ylabel("U (degrees)")
        plt.figure(3)
        plt.plot(time, X_array[:,0,0])
        plt.plot(time, X_array[:,1,0])
        plt.title('X position')
        plt.xlabel("time (s)")
        plt.ylabel("X (m)")
        plt.figure(4)
        plt.title("Velocity")
        plt.xlabel("time (s)")
        plt.ylabel("V (m/s)")
        plt.plot(time, X_array[:,0,4])
        plt.plot(time, X_array[:,1,4])
        # plt.figure(4)
        # plt.plot(time, U1)
        # plt.plot(time, U2)
        plt.show()

def main():
    c = np.array([0,0])
    r = 80
    n_ac = 2 # no. of aircraft in formation flight
    X_array, U_array, time, U1, U2 = CircularFormationGVF(c, r, n_ac)
    theta_ref = np.arange(0, 2*np.pi, 0.01)
    Y_ref = [r*np.cos(theta_ref), r*np.sin(theta_ref)]
    # breakpoint()
    # plotting results
    # d2plot.plot_trajectory_2d(time, X_array, [U_array,0], Y_ref)
    plotting(X_array, U_array, U1, U2, Y_ref, time)
    
main()