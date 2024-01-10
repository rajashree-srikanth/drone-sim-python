#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import scipy.integrate

def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

def carrot_line(X, lookahead=10., p0=[0, 20], p1=[100, 20]):
    p0P = X[:2]-p0
    p0p1 = np.asarray(p1)-p0
    n = p0p1/np.linalg.norm(p0p1)
    p3 = p0+n*np.dot(p0P,n)+n*lookahead*n
    return p3

def carrot_circle(X, lookahead=10, c=[0, 0], r=30.):
    cP = X[:2]-np.asarray(c)
    alpha1 = np.arctan2(cP[1], cP[0])
    alpha2 = alpha1 + lookahead/r
    p3 = c + np.array([r*np.cos(alpha2), r*np.sin(alpha2)])
    return p3

def cont_dyn(X, t, U, v=10., wx=0., g=9.81):
    psi, phi = X[2], U[0]
    Xdot=[v*np.cos(psi)+wx, v*np.sin(psi), g/v*np.tan(phi)]
    return Xdot

def disc_dyn(Xk, Uk, dt, v=10., wx=0.):
    Xk, Xkp1 = scipy.integrate.odeint(cont_dyn, Xk, [0, dt], args=(Uk, v, wx))
    return Xkp1

def pure_pursuit(X, traj, lookahead=20., K=0.2):
    carrot = traj(X, lookahead)
    pc = carrot-X[:2]
    err_psi = norm_mpi_pi(X[2] - np.arctan2(pc[1], pc[0]))
    phi_sp = -K*err_psi
    return phi_sp

def run_simulation(t0=0, t1=60, dt=0.01, traj=carrot_line, lookahead=20., K=0.2, v=10., wx=0.):
    time = np.arange(t0, t1, dt)
    X,U = np.zeros((len(time), 3)), np.zeros((len(time), 1))
    for i in range(1, len(time)):
        U[i-1] = pure_pursuit(X[i-1], traj, lookahead, K)
        X[i] = disc_dyn(X[i-1], U[i-1], dt, v, wx)
    U[-1] = pure_pursuit(X[-1], traj, lookahead, K)
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
    time, X, U = run_simulation(traj=carrot_circle)
    plot_trajectory(time, X, U)
    plt.show()
    
if __name__ == "__main__":
    main()

