# standalone implementation of diff flatness controller for traj generated
# from optimization algorithm

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

def ComputeDerivatives(x_ref, y_ref, dt):
    Fdx = np.gradient(x_ref,edge_order=2)/dt
    Fddx = np.gradient(Fdx, edge_order=2)/dt
    
    Fdy = np.gradient(y_ref, edge_order=2)/dt
    Fddy = np.gradient(Fdy, edge_order=2)/dt
    
    return Fdx, Fdy, Fddx, Fddy

def implement_controller(n_ac, df, v, w, X0s):
    
    # extracting data from df table
    time_opt = np.array(df['time'])
    x_ref = np.zeros((len(time_opt),n_ac))
    y_ref = np.zeros((len(time_opt),n_ac))
    psi_ref = np.zeros((len(time_opt),n_ac))
    
    for i in range(n_ac):
        x_ref[:,i] = np.array(df[f'x_{i+1}'])
        y_ref[:,i] = np.array(df[f'y_{i+1}'])
        psi_ref[:,i] = np.array(df[f'psi_{i+1}'])

    # breakpoint()
    
    # introducing other parameters
    dt = time_opt[1] - time_opt[0]
    
    # initializing matrices
    U_array = np.zeros((len(time_opt),n_ac,2)) # rows - time; columns - ac no; 2 inputs
    dX_array = np.zeros((len(time_opt),n_ac,5))
    X_ref_array = np.zeros((len(time_opt),n_ac,5))
    Yd_ref_array = np.zeros((len(time_opt),n_ac,2))
    Ydd_ref_array = np.zeros((len(time_opt),n_ac,2))
    X_array = np.zeros((len(time_opt),n_ac,5))
    aircraft = []
    
    Fdx = np.zeros((len(time_opt),n_ac))
    Fdy = np.zeros((len(time_opt),n_ac))
    Fddx = np.zeros((len(time_opt),n_ac))
    Fddy = np.zeros((len(time_opt),n_ac))
    
    # creating class instances
    windfield = ddg.WindField(w)
    for i in range(n_ac):
        aircraft.append(ddyn.Aircraft())
        Fdx[:,i], Fdy[:,i], Fddx[:,i], Fddy[:,i] = ComputeDerivatives(x_ref[:,i], y_ref[:,i], dt) # change how this is done later!
    # traj = tracking.CircleTraj(v, r)    
    ctrl = tracking.DiffController(w)
    
    for i in range(n_ac):
        # X_array[0][i][:] = X1
        X_array[0][i][:] = np.array(X0s[i])
        # breakpoint()
    for i in range(1, len(time_opt)):
        t = time_opt[i-1]
        for j in range(n_ac):
            ac = aircraft[j]
            Y_ref = [x_ref[i][j], y_ref[i][j]]
            Yd_ref = [Fdx[i][j], Fdy[i][j]]
            Ydd_ref = [Fddx[i][j], Fddy[i][j]]
            Yddd_ref = [0, 0]
            X = X_array[i-1, j, :]
            # ctrl = tracking.DiffController(w)
            Xr, dX, U = ctrl.ComputeGain(t, X, Y_ref, Yd_ref, Ydd_ref, Yddd_ref, ac)
            # print("reference", Xr, UkX)
            X_new = ac.disc_dyn(X, U, windfield, t, dt)
            X_array[i][j] = X_new
            U_array[i-1][j] = U
            dX_array[i-1][j] = dX
            X_ref_array[i-1][j] = Xr
            Yd_ref_array[i-1][j] = Yd_ref
            Ydd_ref_array[i-1][j] = Ydd_ref
    return X_array, U_array, X_ref_array, Yd_ref_array, Ydd_ref_array, time_opt, dX_array
    
# plotting
def plotting_states(X_array, X_ref_array, time_opt):
    plt.figure(1)
    plt.plot(X_array[:,:,0], X_array[:,:,1], label='real')
    plt.plot(X_ref_array[:-1, :, 0], X_ref_array[:-1, :, 1], label='ref')
    plt.legend()
    plt.title("trajectory computed")
    
    plt.figure(2)
    plt.plot(time_opt, X_array[:,:,0], label="x real")
    plt.plot(time_opt[:-1], X_ref_array[:-1, :, 0], label='x ref')
    plt.title("time_opt, x computed")
    plt.legend()
    
    plt.figure(3)
    plt.plot(time_opt, X_array[:,:,1],label="y real")
    plt.plot(time_opt[:-1], X_ref_array[:-1, :, 1], label='y ref')
    plt.title("time_opt, y computed")
    plt.legend()
    
    plt.figure(4)
    plt.plot(time_opt, X_array[:, :, 2], label='psi')
    plt.plot(time_opt[:-1], X_ref_array[:-1, :, 2], label='psi ref')
    plt.legend()
    plt.show()
    
def plotting_inputs(U_array, X_array, dX_array, time_opt):
    plt.figure(4)
    plt.plot(time_opt, np.rad2deg(U_array[:,0,0]),label="Commanded input \phi_c")
    plt.plot(time_opt, np.rad2deg(X_array[:,0,3]),label="Measured bank angle")
    plt.ylabel("Angle (deg)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.title("Bank angle input vs measured")
    
    plt.figure(5)
    plt.plot(time_opt, np.rad2deg(dX_array[:,0,2]),label="Measured dX")
    plt.ylabel("dpsi")
    plt.title("error in psi with time")
    plt.xlabel("time (s)")
    plt.legend()
    
    plt.figure(6)
    plt.plot(time_opt, np.sqrt(dX_array[:,0,0]**2+dX_array[:,0,1]**2),label="Measured distance error")
    plt.ylabel("distance")
    plt.title("error in distance with time")
    plt.xlabel("time (s)")
    plt.legend()
    
    plt.figure(7)
    plt.plot(time_opt, U_array[:,0,1],label="Commanded input v_c")
    plt.plot(time_opt, X_array[:,0,4],label="Measured")
    plt.ylabel("Vel (m/s)")
    plt.title("Velocity input vs measured")
    plt.xlabel("time (s)")
    plt.legend()
    plt.show()
    
    
def plotting_derivatives(X_ref_array, Yd_ref_array, Ydd_ref_array, time_opt):
    plt.figure(6)
    plt.plot(time_opt, X_ref_array[:, 0, 0], label="Yrefx")
    plt.plot(time_opt, Yd_ref_array[:, 0, 0], label="Ydrefx")
    plt.plot(time_opt, Ydd_ref_array[:, 0, 0], label="Yddrefx")
    plt.legend()
    
    plt.figure(7)
    plt.plot(time_opt, X_ref_array[:, 0, 1], label="Yrefy")
    plt.plot(time_opt, Yd_ref_array[:, 0, 1], label="Ydrefy")
    plt.plot(time_opt, Ydd_ref_array[:, 0, 1], label="Yddrefy")
    plt.legend()
    plt.show()
    
def main():
    n_ac = 4
    
    # importing data for optimization alg
    df = pd.read_csv('opt_states_simple_traj.csv')
                 
    # introducing parameters
    v = 10 # flight speed/airspeed
    w = [0, 0] # wind vel
    
    # initial state conditions
    # X1 = np.array([50,30,np.deg2rad(np.pi/2),0,10]) 
    X0s = ((0, 40, np.deg2rad(0), 0, 12), (25, 20, np.deg2rad(0), 0, 12), (25, -20, np.deg2rad(0), 0, 12), (0, -40, np.deg2rad(0), 0, 12))
    # X0s = ((0, 40, np.deg2rad(0), 0, 12), (40, 0, np.deg2rad(0), 0, 12), (0, -40, np.deg2rad(0), 0, 12), (-40, 0, np.deg2rad(0), 0, 12))
    # X0s = ((0, 40, np.deg2rad(0), 0, 12), (25, 20, np.deg2rad(0), 0, 12), (25, -20, np.deg2rad(0), 0, 12), (0, -40, np.deg2rad(0), 0, 12))
    # breakpoint()
    
    # calling functions
    X_array, U_array, X_ref_array, Yd_ref_array, Ydd_ref_array, time_opt, dX_array = implement_controller(n_ac, df, v, w, X0s)
    plotting_states(X_array, X_ref_array, time_opt)
    plotting_inputs(U_array, X_array, dX_array, time_opt)
    
main()
def useless_code():
    # redundant code IGNORE!!!!!
        # t = 3
        # traj = CircleTraj(v)
        # Y_ref, Yd_ref, Ydd_ref = traj.TrajPoints(t=3)
        # print("ref traj values for time t = 2", Y_ref, Yd_ref, Ydd_ref)
        # flatness = DiffFlatness(w)
        # Xr, UkX = flatness.ComputeFlatness(2, Y_ref, Yd_ref, Ydd_ref)
        # print(Xr, UkX)
        # ctrl = DiffController(w)
        # ac = ddyn.Aircraft()
        # # breakpoint()
        # Xr, UkX, U = ctrl.ComputeGain(t, X, Y_ref, Yd_ref, Ydd_ref)
        # print("Controller", Xr, UkX, U)
        # print(X_array)
        # print(np.shape(U_array))
        pass
