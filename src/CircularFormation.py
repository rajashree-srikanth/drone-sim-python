#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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

def CircularFormationSim(c, r, n_ac):
    t_start, t_step, t_end = 0, 0.01, 20
    time = np.larange(t_start, t_end, t_step)
    # n_ac = 1 # no. of aircraft in formation flight
    windfield = ddg.WindField() # creating windfield object
    aircraft = []
    controllers = []
    ke = 1
    kd = 1
    X = [0, 0, 0, 0, 0] # initializing state
    for i in range(n_ac): # creating aircraft objects for each reqd no. of aircraft
        aircraft.append(ddyn.Aircraft())
    # calling GVF controller
    i = 0 # time index
    U_array = [[]] # rows - time; columns - ac no
    X_array = [[]]
    for t in time:
        for ac in aircraft:
            traj = ddg.CircleTraj(X, c, r)
            gvf = ddg.GVFcontroller(traj, ac, windfield)
            U = gvf.get(X, ke, kd)
            U_array[i].append(U)
            ac = ddyn.Aircraft
            X_new = ac.disc_dyn(X, U, windfield, t, t_step)
            X = X_new
            X_array[i].append(X)
        i = i+1
    return X_array, U_array, time

def main():
    c = [0,0]
    r = 1
    n_ac = 1 # no. of aircraft in formation flight
    X_array, U_array, time = CircularFormationSim(c, r, n_ac)
    theta_ref = np.arange(0, 2*np.pi, 0.01)
    Y_ref = [r*np.cos(theta_ref), r*np.sin(theta_ref)]
    
    # plotting results
    d2plot.plot_trajectory_2d(time, X_array, U_array, Y_ref)