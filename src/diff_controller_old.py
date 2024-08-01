# trial of other controller
# this seems to work, so why is the other thing not working? 
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

def main():
    v = 10 # flight speed/airspeed
    w = [0, 0]
    windfield = ddg.WindField() # creating windfield object
    n_ac = 1
    dt = 0.01
    t0, tf = 0, 40
    time = np.arange(t0, tf, dt)
    aircraft = []
    ctlrs = []
    
    U_array = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no; 2 inputs
    Y_ref_array = np.zeros((len(time),n_ac,2))
    Yd_ref_array = np.zeros((len(time),n_ac,2))
    Ydd_ref_array = np.zeros((len(time),n_ac,2))
    X_array = np.zeros((len(time),n_ac,5))
    
    traj = dtraj.TrajectoryCircle(c=[0,0])
    X1 = np.array([0,30,0,0,10]) # initial state conditions
    
    for i in range(n_ac): # creating aircraft class instances 
        aircraft.append(ddyn.Aircraft())
        ctlrs.append(ddg.DFFFController(traj, aircraft[i], windfield))
    
    for i in range(n_ac):
        X_array[0][i][:] = X1
    
    for i in range(1, len(time)):
        t = time[i-1]
        for j in range(n_ac):
            ac = aircraft[j]
            ctrl = ddg.DFFFController(traj, ac, windfield)
            X = X_array[i-1, j, :]
            ref = traj.get(t)
            Y_ref, Yd_ref, Ydd_ref = ref[0], ref[1], ref[2]
            U = ctrl.get(X, t)
            X_new = ac.disc_dyn(X, U, windfield, t, dt)
            X_array[i][j] = X_new
            U_array[i-1][j] = U
            Y_ref_array[i-1][j] = Y_ref
            Yd_ref_array[i-1][j] = Yd_ref
            Ydd_ref_array[i-1][j] = Ydd_ref
    
    plt.figure(1)
    plt.plot(X_array[:,0,0], X_array[:,0,1], label='actual')
    plt.plot(Y_ref_array[:, 0, 0], Y_ref_array[:, 0, 1], label='ref')
    plt.title("traj")
    plt.legend()
    # plt.figure(2)
    # plt.plot(time, X_array[:,0,0])
    # plt.title("time, x computed")
    # plt.figure(3)
    # plt.plot(time, X_array[:,0,1])
    # plt.title("time, y computed")
    # plt.figure(4)
    # plt.plot(time, U_array[:,0,0],label="Command")
    # plt.plot(time, X_array[:,0,3],label="Measured")
    # plt.legend()
    # plt.title("uphi, t computed VS measured")
    # plt.figure(5)
    # plt.plot(time, U_array[:,0,1],label="Command")
    # plt.plot(time, X_array[:,0,4],label="Measured")
    # plt.legend()
    # plt.title("u_v computed computed VS measured")
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
    plt.show()

# there seems to be an issue in this controller too?
main()
