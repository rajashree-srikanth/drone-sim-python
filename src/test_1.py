#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import dronisos_guidance as dg


def run_simulation(t0=0, t1=60, dt=0.01):
    time = np.arange(t0, t1, dt)
    X,U = np.zeros((len(time), dg.Aircraft.s_size)), np.zeros((len(time), dg.Aircraft.i_size))
    for i in range(1, len(time)):
        #U[i-1] = pure_pursuit(X[i-1], traj, lookahead, K)
        U[i-1] = [np.deg2rad(0.), 10.]
        #X[i] = disc_dyn(X[i-1], U[i-1], dt, v, wx)
    #U[-1] = pure_pursuit(X[-1], traj, lookahead, K)
    return time, X, U

def plot_trajectory(time, X, U, _f=None, _a=None):
    _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
    _a = _f.subplots(2, 1) if _a is None else _a
    _a[0].plot(X[:,0], X[:,1]); _a[0].set_title('2D')
    _a[0].axis('equal')
    _a[1].plot(time, np.rad2deg(U))
    _a[1].set_title('Roll'); _a[1].grid(True)
    return _f, _a


def main():
    time, X, U = run_simulation()
    plot_trajectory(time, X, U)
    plt.show()
    
if __name__ == "__main__":
    main()
