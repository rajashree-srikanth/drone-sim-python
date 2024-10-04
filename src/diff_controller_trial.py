# attempt to implement diff flatness controller
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
import Controllers as tracking

def main():
    # introducing parameters
    v = 10 # flight speed/airspeed
    # w = [2/np.sqrt(2), 2/np.sqrt(2)] # wind vel
    w = [0,0]
    n_ac = 1
    dt = 0.05
    t0, tf = 0, 100
    r = 50
    time = np.arange(t0, tf, dt)
    
    # initializing matrices
    U_array = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no; 2 inputs
    Y_ref_array = np.zeros((len(time),n_ac,2))
    Yd_ref_array = np.zeros((len(time),n_ac,2))
    Ydd_ref_array = np.zeros((len(time),n_ac,2))
    X_array = np.zeros((len(time),n_ac,5))
    aircraft = []
    
    # creating class instances
    windfield = ddg.WindField(w)
    for i in range(n_ac):
        aircraft.append(ddyn.Aircraft())
    traj = tracking.CircleTraj(v, r)    
    ctrl = tracking.DiffController(w)
    
    # initial state conditions
    X1 = np.array([0.5,0,-np.pi/2,0,10]) # initial state conditions
    # X1 = np.array([0,0,np.deg2rad(np.pi/2),0,15]) 
    for i in range(n_ac):
        X_array[0][i][:] = X1
    
    for i in range(1, len(time)):
        t = time[i-1]
        for j in range(n_ac):
            ac = aircraft[j]
            Y_ref, Yd_ref, Ydd_ref, Yddd_ref = traj.TrajPoints(t)
            # Y_ref, Yd_ref, Ydd_ref = traj.get(t)
            X = X_array[i-1, j, :]
            # ctrl = tracking.DiffController(w)
            Xr, dX, U = ctrl.ComputeGain(t, X, Y_ref, Yd_ref, Ydd_ref, Yddd_ref, ac)
            # print("reference", Xr, Ur)
            X_new = ac.disc_dyn(X, U, windfield, t, dt)
            X_array[i][j] = X_new
            U_array[i-1][j] = U
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
    # print(X_array)
    # print(np.shape(U_array))
    
    
    # plotting
    plt.figure(1)
    plt.plot(X_array[:,0,0], X_array[:,0,1], "k", label='$\\text{Real Traj}$')
    plt.plot(Y_ref_array[:-1, 0, 0], Y_ref_array[:-1, 0, 1], color="orange",label='$\\text{Ref Traj}$')
    # plt.plot(X_array[:,1,0], X_array[:,1,1], label='real 2')
    # plt.plot(Y_ref_array[:-1, 1, 0], Y_ref_array[:-1, 1, 1], label='ref 2')
    plt.legend(loc="upper right")
    plt.title('$\\text{Ref vs Tracked Trajectory}$')
    
    # Define the subplots with a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Set main title for the entire figure
    fig.suptitle("Controller Performance Using Feedback Control")

    # Plot on the first subplot (top-left)
    axs[0, 0].plot(time, X_array[:,0,0], label="X actual")
    axs[0, 0].plot(time[:-1], Y_ref_array[:-1, 0, 0], label="X reference")
    axs[0, 0].set_title("X Position")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("X (in m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot on the second subplot (top-right)
    axs[0, 1].plot(time, X_array[:,0,1], label="Y actual")
    axs[0, 1].plot(time[:-1], Y_ref_array[:-1, 0, 1], label="Y reference")
    axs[0, 1].set_title("Y position")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Y (in m)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot on the third subplot (bottom-left)
    axs[1, 0].plot(time[:-1], np.rad2deg(U_array[:-1,0,0]), label="Commanded bank input")
    axs[1, 0].plot(time[:-1], np.rad2deg(X_array[:-1,0,3]), label="Measured bank angle")
    axs[1, 0].set_title("Bank Angle Input vs Measured")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Angle (deg)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot on the fourth subplot (bottom-right)
    axs[1, 1].plot(time[:-1], U_array[:-1,0,1], label="Commanded velocity input")
    axs[1, 1].plot(time[:-1], X_array[:-1,0,4], label="Measured velocity")
    axs[1, 1].set_title("Velocity Input vs Measured")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Velocity (m/s)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    # plt.figure(2)
    # plt.plot(time, X_array[:,0,0])
    # plt.title("time, x computed")
    # plt.figure(3)
    # plt.plot(time, X_array[:,0,1])
    # plt.title("time, y computed")
    # plt.figure(4)
    # plt.plot(time, np.rad2deg(U_array[:,0,0]),label="Commanded input \phi_c")
    # plt.plot(time, np.rad2deg(X_array[:,0,3]),label="Measured bank angle")
    # plt.ylabel("Angle (deg)")
    # plt.xlabel("time (s)")
    # plt.legend()
    # plt.title("Bank angle input vs measured")
    # plt.figure(5)
    # plt.plot(time, U_array[:,0,1],label="Commanded input v_c")
    # plt.plot(time, X_array[:,0,4],label="Measured")
    # plt.ylabel("Vel (m/s)")
    # plt.xlabel("time (s)")
    # plt.legend()
    # plt.title("Velocity input vs measured")
    # plt.figure(6)
    # plt.plot(time, Y_ref_array[:, 0, 0], label="Yrefx")
    # plt.plot(time, Yd_ref_array[:, 0, 0], label="Ydrefx")
    # plt.plot(time, Ydd_ref_array[:, 0, 0], label="Yddrefx")
    # plt.legend()
    # plt.figure(7)
    # plt.plot(time, Y_ref_array[:, 0, 1], label="Yrefy")
    # plt.plot(time, Yd_ref_array[:, 0, 1], label="Ydrefy")
    # plt.plot(time, Ydd_ref_array[:, 0, 1], label="Yddrefy")
    # plt.legend()
    plt.figure(8)
    plt.plot(time, np.rad2deg(X_array[:, 0, 2]), label='psi')
    plt.legend()
    plt.show()
    
main()