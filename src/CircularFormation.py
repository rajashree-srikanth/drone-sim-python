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
    ke = 0.005
    kd = 10
    X = [30,90,-np.pi/2,0,10] # initial state conditions

    traj = ddg.CircleTraj(c, r) # creating traj object
    for i in range(n_ac): 
        aircraft.append(ddyn.Aircraft()) # creating aircraft objects
        # calling GVF controller
        controllers.append(ddg.GVFcontroller(traj, aircraft[i], windfield)) 

    i, j = 0, 0 # time and aicraft count index
    U_array = np.zeros((len(time),n_ac)) # rows - time; columns - ac no
    U1_array = np.zeros((len(time),n_ac)) # useful for debugging
    U2_array = np.zeros((len(time),n_ac)) # useful for debuggung
    X_array = np.zeros((len(time),n_ac,5))    

    for t in time:
        for j in range(n_ac):
            gvf, ac = controllers[j], aircraft[j] # object names
            e, n, H = traj.get(X) # obtaining required traj and their derivatives
            U, U1, U2 = gvf.get(X, ke, kd, e, n, H) # obtaining control input
            U = np.arctan(U/9.81) # roll setpoint angle in radians
            # breakpoint()
            U_array[i][j] = U
            U1_array[i][j] = U1
            U2_array[i][j] = U2
            # new state from prev time step and required control action
            X_new = ac.disc_dyn(X, [U, 15], windfield, t, t_step) 
            X = X_new
            X_array[i][j] = X
            j = j+1
        i = i+1
    return X_array, U_array, time, U1_array, U2_array

def main():
    c = [0,0]
    r = 80
    n_ac = 1 # no. of aircraft in formation flight
    X_array, U_array, time, U1, U2 = CircularFormationGVF(c, r, n_ac)
    theta_ref = np.arange(0, 2*np.pi, 0.01)
    Y_ref = [r*np.cos(theta_ref), r*np.sin(theta_ref)]
    # # # # # plotting results # # # # #
    # breakpoint()
    # d2plot.plot_trajectory_2d(time, X_array, [U_array,0], Y_ref)
    plt.gca().set_aspect('equal')
    plt.figure(1)
    plt.plot(Y_ref[1][:], Y_ref[0][:])
    plt.plot(X_array[:,0,0], X_array[:,0,1])
    plt.figure(2)
    plt.plot(time, np.degrees(U_array))
    plt.figure(3)
    plt.plot(time, X_array[:,0,0])
    plt.figure(4)
    plt.title("Velocity")
    plt.plot(time, X_array[:,0,4])
    # plt.figure(4)
    # plt.plot(time, U1)
    # plt.plot(time, U2)
    plt.show()
    
main()