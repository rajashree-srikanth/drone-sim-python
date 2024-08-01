# more comparisons :)
# first, comparing the trajectories side by side
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

import d2d.guidance as ddg
import d2d.dynamic as ddyn
import d2d.trajectory as dtraj
import diff_controller_trial as dct
from mpl_toolkits import mplot3d

import control

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

Y_ref_array_o = np.zeros((len(time),n_ac,2))
Yd_ref_array_o = np.zeros((len(time),n_ac,2))
Ydd_ref_array_o = np.zeros((len(time),n_ac,2))
Y_ref_array_n = np.zeros((len(time),n_ac,2))
Yd_ref_array_n = np.zeros((len(time),n_ac,2))
Ydd_ref_array_n = np.zeros((len(time),n_ac,2))
U_array_o = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no; 2 inputs
U_array_n = np.zeros((len(time),n_ac,2)) # rows - time; columns - ac no; 2 inputs
Xr_array_o = np.zeros((len(time),n_ac,5))
Xr_array_n = np.zeros((len(time),n_ac,5))
X_array_o = np.zeros((len(time),n_ac,5))
X_array_n = np.zeros((len(time),n_ac,5))


    
traj_new = dct.CircleTraj(v, r=30)    
traj_old = dtraj.TrajectoryCircle(c=[0, 0])
X1 = np.array([20,30,-np.pi/2,0,10]) # initial state conditions

# flat_o = ddg.DiffFlatness()
flat_n = dct.DiffFlatness(w)

for i in range(n_ac):
    X_array_o[0][i][:] = X1
    X_array_n[0][i][:] = X1

for i in range(1, len(time)):
    t = time[i-1]
    for j in range(n_ac):
        ac = aircraft[j]
        # new
        Y_ref, Yd_ref, Ydd_ref, Yddd_ref = traj_new.TrajPoints(t)
        Y_ref_array_n[i-1][j] = Y_ref
        Yd_ref_array_n[i-1][j] = Yd_ref
        Ydd_ref_array_n[i-1][j] = Ydd_ref
        X = X_array_n[i-1, j, :]
        Xr, Ur = flat_n.ComputeFlatness(t, Y_ref, Yd_ref, Ydd_ref)
        ctrl = dct.DiffController(w)
        Xr, Ur, U = ctrl.ComputeGain(t, X, Y_ref, Yd_ref, Ydd_ref, ac)
        Un = U
        # if t==0:
        #     # print(U-Un, "U")
        #     breakpoint()
        U_array_n[i-1][j] = Ur
        Xr_array_n[i-1][j] = Xr
        # old
        ref = traj_old.get(t)
        Xr, Ur, Xdot = ddg.DiffFlatness.state_and_input_from_output(ref, w, ac)
        ctrl = ddg.DFFFController(traj_old, ac, windfield)
        X = X_array_o[i-1, j, :]
        U = ctrl.get(X, t)
        X_new = ac.disc_dyn(X, U, windfield, t, dt)
        X_array_o[i][j] = X_new
        # outputs
        Y_ref, Yd_ref, Ydd_ref = ref[0], ref[1], ref[2]
        Y_ref_array_o[i-1][j] = Y_ref
        Yd_ref_array_o[i-1][j] = Yd_ref
        Ydd_ref_array_o[i-1][j] = Ydd_ref
        U_array_o[i-1][j] = Ur
        Xr_array_o[i-1][j] = Xr
        # if t>=2.16 and t<=2.18:
        #     print("A", Ao-An, "B=", Bo-Bn)
        #     print("U", U-Un)
        # if t==0:
        #     print(U-Un, "U")
        #     breakpoint()


# plt.figure(1)
# plt.plot(Y_ref_array_o[:,0,0], Y_ref_array_o[:,0,1], label='old')
# plt.plot(Y_ref_array_n[:, 0, 0], Y_ref_array_n[:, 0, 1], label='new')
# plt.legend()
# plt.figure(4)
# plt.plot(Yd_ref_array_o[:,0,0], Yd_ref_array_o[:,0,1], label='old d')
# plt.plot(Yd_ref_array_n[:, 0, 0], Yd_ref_array_n[:, 0, 1], label='new d')
# plt.legend()
# plt.figure(5)
# plt.plot(Ydd_ref_array_o[:,0,0], Yd_ref_array_o[:,0,1], label='old dd')
# plt.plot(Ydd_ref_array_n[:, 0, 0], Yd_ref_array_n[:, 0, 1], label='new dd')
# plt.legend()
plt.figure(1)
plt.plot(time, Y_ref_array_o[:,0,0], label='old x')
plt.plot(time, Y_ref_array_n[:,0,0], label='new x')
plt.legend()
plt.figure(2)
plt.plot(time, Y_ref_array_o[:,0,1], label='old y')
plt.plot(time, Y_ref_array_n[:,0,1], label='new y')
plt.legend()
plt.figure(3)
plt.plot(time, Yd_ref_array_o[:,0,0], label='old x')
plt.plot(time, Yd_ref_array_n[:,0,0], label='new x')
plt.legend()
plt.figure(4)
plt.plot(time, Yd_ref_array_o[:,0,1], label='old y')
plt.plot(time, Yd_ref_array_n[:,0,1], label='new y')
plt.legend()
# plt.figure(2)
# plt.plot(time, U_array_o[:, 0, 0], label="old input phi")
# plt.plot(time, U_array_n[:, 0, 0], label="new input phi")
# plt.legend()
# plt.figure(3)
# plt.plot(time, U_array_o[:, 0, 1], label="old input v")
# plt.plot(time, U_array_n[:, 0, 1], label="new input v")
# plt.legend()
# plt.figure(6)
# plt.plot(time, Xr_array_o[:, 0, 0], label="old ref state - track x")
# plt.plot(time, Xr_array_n[:, 0, 0], label="new ref state - track x")
# plt.legend()
# plt.figure(7)
# plt.plot(time, Xr_array_o[:, 0, 1], label="old ref state - track y")
# plt.plot(time, Xr_array_n[:, 0, 1], label="new ref state - track y")
# plt.legend()
# plt.figure(8)
# plt.plot(time, Xr_array_o[:, 0, 2], label="old ref state - track psi")
# plt.plot(time, Xr_array_n[:, 0, 2], label="new ref state - track psi")
# plt.legend()
# plt.figure(9)
# plt.plot(time, Xr_array_o[:, 0, 3], label="old ref state - track phi")
# plt.plot(time, Xr_array_n[:, 0, 3], label="new ref state - track phi")
# plt.legend()
# plt.figure(10)
# plt.plot(time, Xr_array_o[:, 0, 4], label="old ref state - track v")
# plt.plot(time, Xr_array_n[:, 0, 4], label="new ref state - track v")
# plt.legend()
# plt.figure(11)
# plt.plot(X_array_o[:, 0, 0], X_array_o[:, 0, 1], label="old computed state - track xy")
# plt.plot(Xr_array_o[:, 0, 0], Xr_array_o[:, 0, 1], label="old ref state - track xy")
# plt.legend()
plt.show()