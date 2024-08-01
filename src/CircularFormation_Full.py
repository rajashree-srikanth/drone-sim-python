#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# main script for running distributed circular formation control

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

import d2d.dynamic as ddyn
import d2d.guidance as ddg
import d2d.utils as du
import d2d.ploting as d2plot
import d2d.animation as dda
import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.scenario as dds
from mpl_toolkits import mplot3d

def CircularFormationGVF(c, r, n_ac, t_end):
    
    # initializing parameters
    t_start, t_step = 0, 0.05
    time = np.arange(t_start, t_end, t_step)
    # n_ac = 1 # no. of aircraft in formation flight
    windfield = ddg.WindField() # creating windfield object
    aircraft = []
    controllers = []
    
    U_array = np.zeros((len(time),n_ac)) # rows - time; columns - ac no
    U1_array = np.zeros((len(time),n_ac)) # useful for debugging
    U2_array = np.zeros((len(time),n_ac)) # useful for debuggung
    Ur_array = np.zeros((len(time),n_ac))
    e_theta_array = np.zeros((len(time),n_ac-1))
    X_array = np.zeros((len(time),n_ac,5))
    
    # controller gains
    ke = 0.0004 # aggressiveness of the gvf guidance
    kd = 15 # speed of exponential convergence to required guidance trajectory
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
    z_des = np.ones(n_ac-1)*(np.pi*2/n_ac) # separation angles between adjacent aircraft are equal
    
    dcf = ddg.DCFController()
    traj = ddg.CircleTraj(c) # creating traj object
    for i in range(n_ac): 
        aircraft.append(ddyn.Aircraft()) # creating aircraft objects
        # calling GVF controller
        controllers.append(ddg.GVFcontroller(traj, aircraft[i], windfield))   
        
    for i in range(1, len(time)):
        t = time[i-1]
        U_r, e_theta = dcf.get(n_ac, B, c, p, z_des, kr)
        Rr = U_r + R # new required radii to be tracked for each ac
        Ur_array[i] = Rr.T
        e_theta_array[i] = e_theta.T
        for j in range(n_ac):
            X = X_array[i-1,j,:]
            gvf, ac = controllers[j], aircraft[j] # object names
            r = Rr[j]
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

def plotting(n_ac, X_array, U_array, U1, U2, Y_ref, time, Ur, e_theta_arr):
        plt.gca().set_aspect('equal')
        plt.figure(1)
        plt.plot(Y_ref[1][:], Y_ref[0][:])
        for i in range(n_ac):  
            plt.plot(X_array[:,i,0], X_array[:,i,1])
        plt.title('XY Trajectory vs reference')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        
        plt.figure(2)
        for i in range(n_ac):
            plt.plot(time, np.degrees(U_array[:,i]))
        plt.title('GVF Control Input')
        plt.xlabel("time (s)")
        plt.ylabel("U (degrees)")
        
        plt.figure(3)
        for i in range(n_ac):
            plt.plot(time, X_array[:,i,0])
        plt.title('X position')
        plt.xlabel("time (s)")
        
        # plt.figure(8)
        # for i in range(n_ac):
            # plt.plot(time, X_array[:,i,1])
        # plt.title('Y position')
        # plt.xlabel("time (s)")
        # plt.ylabel("Y (m)")
        plt.figure(4)
        plt.title("Velocity")
        plt.xlabel("time (s)")
        plt.ylabel("V (m/s)")
        for i in range(n_ac):
            plt.plot(time, X_array[:,i,4])
            
        plt.figure(5)
        plt.plot(time, Ur)
        plt.title("Actual radius")
        
        # plt.figure(6)
        # plt.plot(time, U1)
        # plt.plot(time, U2)
        
        plt.figure(7)
        plt.plot(time, e_theta_arr)
        plt.title("Phase error (in degrees)")
        # plt.figure(2)
        # ax = plt.axes(projection='3d')
        # ax.plot3D(time, X_array[:,0,0], X_array[:,0,1])
        # ax.plot3D(time, X_array[:,1,0], X_array[:,1,1])
        
        # animating
        fig, ax = plt.subplots()
        line = ax.plot(X_array[0,:,0], X_array[0,:,1], "k.", label='Time: 0 s')[0]
        ax.plot(Y_ref[1][:], Y_ref[0][:], "--", label="Reference Trajectory")
        ax.set(xlim=[-150, 150], ylim=[-150, 150], xlabel='X (m)', ylabel='Y [m]', 
               title='Trajectory')
        l = ax.legend()
        def update(frame):
            x = X_array[:,:,0]
            y = X_array[:,:,1]
            line.set_xdata(x[frame-10:frame])
            line.set_ydata(y[frame-10:frame])
            t = "Time: " + str(round(time[frame],1)) + " s"
            l.get_texts()[0].set_text(t)
            return (line)
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(time), interval=1)
        writervideo = animation.FFMpegWriter(fps=60)
        # ani.save('circular_formation.avi',writer=writervideo)
        plt.show()

def main():
    c = np.array([0,0])
    r = 60
    n_ac = int(input("Enter no. of aircraft: ")) # no. of aircraft in formation flight
    t_end = 150*(int(n_ac/3) + 1) # simulation time 
    # phase convergence time increases with no. of aicraft in simulation, hence 
    # it is made variable
    print(t_end)
    X_array, U_array, time, U1, U2, Ur, e_theta_arr = CircularFormationGVF(c, r, n_ac, t_end)
    theta_ref = np.arange(0, 2*np.pi, 0.01)
    Y_ref = [r*np.cos(theta_ref), r*np.sin(theta_ref)]
    
    # converting 3d array X_array to 2d before exporting to csv
    # states = time[:, np.newaxis]
    symbols = ['x', 'y', 'psi', 'phi', 'v']
    states = {"time": time}
    for i in range(n_ac):
        for j in range(len(symbols)):
            states[f'{symbols[j]}_{i+1}'] = list(X_array[:, i, j])
    breakpoint()
    
    
    # # exporting results to csv file
    df = pd.DataFrame(states)
    df.to_csv(r"states_over_time.csv", index=False)
    
    # plotting results
    # d2plot.plot_trajectory_2d(time, X_array, [U_array,0], Y_ref)
    plotting(n_ac, X_array, U_array, U1, U2, Y_ref, time, Ur, e_theta_arr)
    
main()