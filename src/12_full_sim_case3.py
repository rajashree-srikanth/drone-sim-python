# full simulation of all phases - from phase I -> III
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# importing required modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import control
import pandas as pd

import d2d.dynamic as ddyn
import d2d.guidance as ddg
import Controllers as tracking
import d2d.utils as du
import d2d.ploting as d2plot
import d2d.animation as dda
import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.scenario as dds
from mpl_toolkits import mplot3d
# import multi_opt_planner as traj_generator
import single_opt_planner as traj_generator
import d2d.opty_utils as d2ou

'''
The code performs a full simulation of a fleet of fixed-wing drones executing 
coordinated final formation flight for a simple case
circle -> square in circle -> square -> infinity loop

NOTE: the final formation here refers to the formation required before the desired formation
      flight in phase III.

DETAILS:
phase 1 - formation goal - square through gvf circular formation
        - the 4 a/cs track 4 circles with different centres respectively, such that when 
        algorithm convergence is reached, they fly in-phase in circles in a square-like formation
        - condition to exit phase 1: when all 4 aircraft reach phase convergence:
            i.e., when they execute circles with the required z_theta (or e_theta is almost 0).

phase 2 - formation goal -> square + tracking to get to final formation
        - contains 2 components - traj generation and tracking using Diff flatness controller
        - t_opt = 12s - trajectory is mostly straight line 
        - we compute optimal traj for just one of the ac and the remaining ac traj and computed
        by adding offset based on the difference in initial position relative to the ac whose
        optimal trajectory has been computed
        TLDR - one traj only is optimized
        - optimization computation time is low (approx. 0.4s)

phase 3 - tracking of pref-defined traj - infinity loop
        - uses diff flatness controller
        - the tracking of this trajectory is performed until total simulation time exceeds 150s
        
input parameters:
                n_ac - no. of aicraft required in formation (n_ac=4 for this simulation)
                c - centres of the ac circles for phase I (to achieve the required finalformation)
                (this is to be computed beforehand)
                r - radius of formation circle to be tracked in phase I
                v - flight speed/airspeed (magnitude)
                w - [wx, wy] - windspeed
                t_start - start time of simulation (default=0)
                X2_f - fomration states desired at end of phase II
                (this parameter is key to computing optimized traj to be tracked in phase II)
                t_opt - flight time to get to X2_f states at end of phase II
                (modifying this provides different optimal tracjetory that ensures all ac reach 
                desired states at the same time)
                "*.csv" - file name containing desired final formation flight for phase III
                (this is precomputed and the states(t) is direclty provided)
                t_sim_end - total required simulation time including all phases
                t_step - time step
                scen.hz - collocation discretization for phase II optimization (OPTIONAL)

output parameters: (all paramters in SI units)
                X_array - computed states (x, y, psi, phi, v_mag) for all phases
                U_array - commanded input (phi, v_mag) for all phases
                Y_ref - reference trajectory (x,y) coordinates for all phases
                time - simulation time points (in s)
                # other phase-specific paramters
                TO BE FILLED
'''

# phase 1 - circular formation
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

def CircularFormationGVF(c, r, v, n_ac, t_start=0, t_step=0.05, t_end=1000):
    t_f = t_end
    time = np.arange(t_start, t_end, t_step)
    windfield = ddg.WindField() # creating windfield object
    aircraft = []
    controllers = []
    
    U_array = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no
    U1_array = np.zeros((len(time),n_ac)) # useful for debugging
    U2_array = np.zeros((len(time),n_ac)) # useful for debuggung
    Ur_array = np.zeros((len(time),n_ac))
    e_theta_array = np.zeros((len(time),n_ac-1))
    X_array = np.zeros((len(time),n_ac,5))
    
    # controller gains
    ke = 0.1 # aggressiveness of the gvf guidance
    kd = 2 # speed of exponential convergence to required guidance trajectory
    kr = 5 # controls the speed of convergence to required phase separation
    R = r*np.ones((n_ac,1))
    
    X1 = np.array([20,30,-np.pi/2,0,10]) # initial state conditions
    p = np.zeros((2, n_ac)) # columns are the different aircrafts
    # building the initial position matrix p and state matrix X_array
    for i in range(n_ac):
        p[0][i] = X1[0]
        p[1][i] = X1[1]
        X_array[0][i][:] = X1
        
    # building B matrix based on no_ac - assuming a straight formation
    B= ConstructBMatrix(n_ac)
             
    # z_des is a row matrix here, it is converted to column matrix later within the DCF class
    # z_des = np.ones(n_ac-1)*(np.pi*2/n_ac) # separation angles between adjacent aircraft are equal
    z_des = np.zeros(n_ac-1) # separation angles between adjacent aircraft are equal
    
    dcf = ddg.DCFController()
    for i in range(n_ac): 
        aircraft.append(ddyn.Aircraft()) # creating aircraft objects
        traj = ddg.CircleTraj(c[i,:]) # creating traj object
        # calling GVF controller
        controllers.append(ddg.GVFcontroller(traj, aircraft[i], windfield))   
        
    stop, dt, count = 0, 0, 0
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
            traj = ddg.CircleTraj(c[j,:])
            e, n, H = traj.get(X,r) # obtaining required traj and their derivatives
            U, U1, U2 = gvf.get(X, ke, kd, e, n, H) # obtaining control input
            U = np.arctan(U/9.81) # roll setpoint angle in radians
            U_array[i-1][j] = [U, v]
            U1_array[i-1][j] = U1
            U2_array[i-1][j] = U2
            # new state from prev time step and required control action
            X_new = ac.disc_dyn(X, [U, v], windfield, t, t_step) 
            X = X_new
            X_array[i][j] = X
            p[0][j] = X[0]
            p[1][j] = X[1]
        if (e_theta.T<=2.).all():
            stop = 1
            t_opt_comp = 0.7
            if (stop==1) and (count==0):
                print("convergence!", i-1, t)
                t_convergence = t
                index = i-1
                count+=1
            if (stop==1) and (dt>=t_opt_comp):  
                print ("stop due to e_theta convergence", t)
                X_array, U_array = X_array[:i+1,:,:], U_array[:i+1,:]
                U1_array, U2_array = U1_array[:i+1,:], U2_array[i+1,:]
                Ur_array, e_theta_array = Ur_array[:i+1,:], e_theta_array[:i+1,:]
                t_f = t # time when circle simulation computation ends
                time = time[:i+1]
                break
            dt += t_step
            print("convergence reached. waiting until traj computed...", dt)
    convergence = [index, t_convergence, t_f]
    return X_array, U_array, U1_array, U2_array, Ur_array, e_theta_array, time, convergence
  
# phase 2.1 - trajectory generation using optimization
def trajectory_optimization_single(scen, delta):
    print(f'Scenario: {scen.name} - {scen.desc}')
    p = traj_generator.Planner(scen)
    print("planner initialized...")
    p.configure(tol=1e-5, max_iter=1500)
    p.run(initial_guess=p.get_initial_guess())
    f1, a1 = d2ou.plot2d(p, None)
    f2, a2 = d2ou.plot_chrono(p, None)
    p = trajectory_gen_multi(p, delta)
    return p    

def trajectory_gen_multi(p, delta):
    x = p.sol_x.reshape(-1,1) # reshaping 1d to 2d array is mandatory before appending
    y = p.sol_y.reshape(-1,1)
    x = np.append(x, x+delta[0], axis=1)
    y = np.append(y, y+delta[1], axis=1)
    p.sol_x, p.sol_y = x, y
    return p
# phase 2.2, 3 - trajectory tracking
def ComputeDerivatives(x_ref, y_ref, dt):
    Fdx = np.gradient(x_ref,edge_order=2)/dt
    Fddx = np.gradient(Fdx, edge_order=2)/dt
    
    Fdy = np.gradient(y_ref, edge_order=2)/dt
    Fddy = np.gradient(Fdy, edge_order=2)/dt
    
    return Fdx, Fdy, Fddx, Fddy

def ExtractTrajData(df, n_ac):
    time_track = np.array(df['time'])
    x_ref = np.zeros((len(time_track), n_ac))
    y_ref = np.zeros((len(time_track), n_ac))
    psi_ref = np.zeros((len(time_track), n_ac))
    
    for i in range(n_ac):
        x_ref[:,i] = np.array(df[f'x_{i+1}'])
        y_ref[:,i] = np.array(df[f'y_{i+1}'])
        psi_ref[:,i] = np.array(df[f'psi_{i+1}'])
    
    return time_track, x_ref, y_ref, psi_ref

def ExtendTraj_symm (n_ac, x_ref, y_ref, psi_ref, time):
    '''
    this is for the specific case of the infinity traj 
    only one half of inf is ref traj for each ac
    the other half can be obtained from the other aircraft
    
    This can be used to complete any symmetric trajectory abt y axis
    '''
    time = np.append(time, time+time[-1])
    # breakpoint()
    x_ref_sym, y_ref_sym, psi_ref_sym = x_ref, y_ref, psi_ref
    ax = []
    x0, xf, y0, yf = x_ref[0,:], x_ref[-1,:], y_ref[0,:], y_ref[-1,:]
    for i in range(n_ac):
        j = np.nonzero((x0==xf[i]) & (y0==yf[i])) # this is a numpy array within a tuple! which is super weird!!!
        j=(j[0])[0]
        ax.append(j)
    x_ref_sym, y_ref_sym, psi_ref_sym = x_ref_sym[:,ax], y_ref_sym[:, ax], psi_ref_sym[:,ax]
    x_ref, y_ref = np.append(x_ref, x_ref_sym, axis=0) , np.append(y_ref, y_ref_sym, axis=0) 
    psi_ref = np.append(psi_ref, y_ref_sym, axis=0) 
    return time, x_ref, y_ref, psi_ref
    
def implement_controller(n_ac, time, x_ref, y_ref, v, w, X0s):
    
    # introducing other parameters
    dt = time[1] - time[0]
    
    # initializing matrices
    U_array = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no; 2 inputs
    dX_array = np.zeros((len(time),n_ac,5))
    X_ref_array = np.zeros((len(time),n_ac,5))
    Yd_ref_array = np.zeros((len(time),n_ac,2))
    Ydd_ref_array = np.zeros((len(time),n_ac,2))
    X_array = np.zeros((len(time),n_ac,5))
    aircraft = []
    
    Fdx = np.zeros((len(time),n_ac))
    Fdy = np.zeros((len(time),n_ac))
    Fddx = np.zeros((len(time),n_ac))
    Fddy = np.zeros((len(time),n_ac))
    
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
    for i in range(1, len(time)):
        t = time[i-1]
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
    return X_array, U_array, X_ref_array, Yd_ref_array, Ydd_ref_array, dX_array

# plotting
def plotting_states(c, n_ac, X_array, U_array, U1, U2, Y_ref, time, Ur, e_theta_arr):
        
        # plt.gca().set_aspect('equal')
        plt.figure(1)
        for i in range(n_ac):  
            plt.plot(Y_ref[0][:]+c[i,0], Y_ref[1][:]+c[i,1], "k--")
            plt.plot(X_array[:,i,0], X_array[:,i,1], label=f'ac_{i+1}')
        plt.title('XY Trajectory vs reference')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        
        plt.figure(2)
        for i in range(n_ac):
            plt.plot(time, X_array[:,i,0], label=f'ac_{i+1}')
        plt.title('X position')
        plt.xlabel("time (s)")
        plt.legend()
        
        plt.figure(3)
        for i in [0, 2]:
            plt.plot(time, X_array[:,i,1], label=f'ac_{i+1}')
        plt.title('Y position')
        plt.xlabel("time (s)")
        plt.ylabel("Y (m)")
        plt.legend()
        
        plt.figure(4)
        plt.plot(time, np.rad2deg(X_array[:,:,2]))
        plt.title("heading angle psi")
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        
        plt.figure(5)
        for i in [1,3]:
            plt.plot(time, X_array[:,i,1], label=f'ac_{i+1}')
        plt.title('Y position')
        plt.xlabel("time (s)")
        plt.ylabel("Y (m)")
        plt.legend()
        
        plt.figure(6)
        for i in [1,2]:
            plt.plot(time, X_array[:,i,1], label=f'ac_{i+1}')
        plt.title('Y position')
        plt.xlabel("time (s)")
        plt.ylabel("Y (m)")
        plt.legend()
        
        plt.figure(7)
        for i in [0,3]:
            plt.plot(time, X_array[:,i,1], label=f'ac_{i+1}')
        plt.title('Y position')
        plt.xlabel("time (s)")
        plt.ylabel("Y (m)")
        plt.legend()

def plotting_inputs(time, time_1, X_array, U_array, Ur, U1=[0,0], U2=[0,0]):
    plt.figure()
    plt.plot(time, np.rad2deg(U_array[:,:,0]))
    plt.title('GVF Control Input')
    plt.xlabel("time (s)")
    plt.ylabel("U (degrees)")
    
    plt.figure()
    plt.plot(time, X_array[:,:,4])
    plt.title("Velocity")
    plt.xlabel("time (s)")
    plt.ylabel("V (m/s)")
    
    # specific to phase I only
    plt.figure()
    plt.plot(time_1, Ur)
    plt.title("Actual radius")
    
    # plt.figure(6)
    # plt.plot(time, U1)
    # plt.plot(time, U2)
    
def plotting_derivatives():
    pass

def plotting_misc(time, e_theta_arr):
    plt.figure()
    plt.plot(time, e_theta_arr)
    plt.title("Phase error (in degrees)")

def display_animation(time, X_array, Y_ref):
    fig, ax = plt.subplots()
    line = ax.plot(X_array[0,:,0], X_array[0,:,1], "k.", label='Time: 0 s')[0]
    ax.plot(Y_ref[1][:], Y_ref[0][:], "--", label="Reference Trajectory")
    ax.set(xlim=[-200, 200], ylim=[-200, 200], xlabel='X (m)', ylabel='Y [m]', 
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

# main code that combines everything
def main():
    ## input parameters ## 
    ## generic parameters
    v = 15 # flight speed
    w = [0, 0] # wind
    n_ac = 4
    
    ### phase I parameters
    r = 60
    # centres - this will probably be automated later!!!!
    c = np.array([0,0])
    c = np.ones((n_ac, 2))*c  # this will be useful for non-origin centre
    c[0,:], c[1,:], c[2,:], c[3,:] = [0,-20], [25,-20], [25,-100], [0,-100]
    t_start, t_step = 0, 0.05
    t_end_1 = 1000 # change this to be based on a/c count or something else
    # final states we want to achieve to start phase 2
    # X1_f = ((0, 40, np.deg2rad(0), 0, 12), (25, 40, np.deg2rad(0), 0, 12), (25, -40, np.deg2rad(0), 0, 12), (0, -40, np.deg2rad(0), 0, 12))
    
    ### phase II parameters
    # final states we want to reach to start phase 3
    X2_f = ((75, 40, 0, 0, 12), (100, 40, 0, 0, 12), (100,-40, 0, 0, 12), (75, -40, 0, 0, 12))
    t_opt = 15

    ### phase III parameters
    df = pd.read_csv('inf_traj_10s.csv') # phase III formation flight traj
    t_sim_end = 150
    
    
    #### phase 1 of flight ####
    print(t_end_1)
    X_array_1, U_array_1, U1_1, U2_1, Ur_1, e_theta_arr_1, time_1, convergence = CircularFormationGVF(c, r, v, n_ac, t_start, t_step, t_end_1)
    [index_convergence, t_convergence, t1_f] = convergence
    print (convergence)
    X_array, U_array, U1, U2, Ur, e_theta_arr, time = X_array_1, U_array_1, U1_1, U2_1, Ur_1, e_theta_arr_1, time_1
    print("duration of phase I:", t1_f)
    print("time elapsed = ", time[-1])
    print("phase 1 computation complete")
    # breakpoint()
    
    #### phase 2 of flight ####
    xy_conv = X_array_1[index_convergence,:,0:2]
    delta = xy_conv[1:] - xy_conv[0]
    delta = delta.T # rows - x, y; columns - a/c number
    xy_actual = X_array_1[-1,:,0:2]
    # print("coordinates at convergence - computation:", xy_conv)
    # print("coordinates actual execution", xy_actual)
    X2_i = tuple(map(tuple, X_array_1[index_convergence,:,:]))
    X2_actual = X_array_1[-1,:,:]
    # print("final for phase 2 is", X_array_1[index_convergence,:,:], "not", X_array_1[-1,:,:])
    # print(np.round(X_array_1[-1,:,:],2))
    # generating trajectory
    scen = traj_generator.exp_1
    scen.t1 = t_opt
    scen.p0 = X2_i[0]
    scen.p1 = X2_f[0]
    
    p = trajectory_optimization_single(scen, delta)
    print("phase 2 trajectory optimization complete")
    # breakpoint()
    plt.show()
    
    # tracking trajectory
    # x_ref_2, y_ref_2 - rows: time, columns: no. of ac
    x_ref_2 = np.array(p.sol_x)
    y_ref_2 = np.array(p.sol_y)
    time_opt = np.array(p.sol_time)
    time_2 = time_opt + t1_f
    X_array_2, U_array_2, X_ref_array_2, Yd_ref_array_2, Ydd_ref_array_2, dX_array_2 = implement_controller(n_ac, time_opt, x_ref_2, y_ref_2, v, w, X2_actual)
    X_array = np.append(X_array, X_array_2, axis=0)
    U_array = np.append(U_array, U_array_2, axis=0)
    time = np.append(time, time_2, axis=0)
    print("phase 2 tracking computation complete")
    print("time elapsed = ", time[-1])
    # breakpoint()
    
    #### phase 3 of flight ####
    X3_i = tuple(map(tuple, X_array_2[-1, :, :]))
    time_3, x_ref_3, y_ref_3, psi_ref_3 = ExtractTrajData(df, n_ac)
    # since traj is symm, we only have half, so extending it
    time_3, x_ref_3, y_ref_3, psi_ref = ExtendTraj_symm(n_ac, x_ref_3, y_ref_3, psi_ref_3, time_3)
    t_phase3tot = []
    t_final = time[-1]
    s = np.shape(X_array)
    X_array_3t = np.empty((0, s[1], s[2]))
    
    while t_final<=t_sim_end:
        X_array_3, U_array_3, X_ref_array_3, Yd_ref_array_3, Ydd_ref_array_3, dX_array_3 = implement_controller(n_ac, time_3, x_ref_3, y_ref_3, v, w, X3_i)    
        # appending results to main array
        X_array = np.append(X_array, X_array_3, axis=0)
        X_array_3t = np.append(X_array_3t, X_array_3, axis=0)
        U_array = np.append(U_array, U_array_3, axis=0)
        t_phase3tot = np.append(t_phase3tot, time_3+time[-1])
        time = np.append(time, time_3+time[-1], axis=0)
        t_final = time[-1]
        print(t_final)
    
    print("phase 3 tracking complete")
    print("time elapsed = ", time[-1])
    
    #### plotting ####
    print("plotting results")
    theta_ref = np.arange(0, 2*np.pi, 0.01)
    Y_ref = [r*np.cos(theta_ref), r*np.sin(theta_ref)]
    plotting_states(c, n_ac, X_array, U_array, U1, U2, Y_ref, time, Ur, e_theta_arr)
    plotting_inputs(time, time_1, X_array, U_array, Ur)
    plotting_misc(time_1, e_theta_arr_1)
    plt.figure(1)
    plt.plot(xy_actual[:,0], xy_actual[:,1],".k", label="actual")
    plt.plot(xy_conv[:,0],xy_conv[:,1],".r", label="convergence")
    plt.legend()
    plt.show()
    display_animation(time, X_array, Y_ref)
    
main()

def commented_code():
    # df = pd.read_csv('inf_traj_10s.csv')
    # n_ac = 4
    # # time_track = np.array(df['time'])
    # time_track, x_ref, y_ref, psi_ref = ExtractTrajData(df, n_ac)
    # ExtendTraj_symm(n_ac, x_ref, y_ref, psi_ref, time_track)
    # print(time_track)
    # breakpoint()
    pass