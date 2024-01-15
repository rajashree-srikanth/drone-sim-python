#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import d2d.guidance as ddg
import d2d.dynamic as ddd

def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab: ax.xaxis.set_label_text(xlab)
    if ylab: ax.yaxis.set_label_text(ylab)
    if title: ax.set_title(title, {'fontsize': 20 })
    if legend is not None:
        if legend == True: ax.legend(loc='best')
        else: ax.legend(legend, loc='best')
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    if min_yspan is not None: ensure_yspan(ax, min_yspan)
    
def plot_trajectory_2d(time, X=None, U=None, Yref=None, Xref=None, _f=None, _a=None):
    _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
    _a = _f.subplots(1, 1) if _a is None else _a
    if X is not None: _a.plot(X[:,0], X[:,1], label='aircraft')
    #breakpoint()
    if Xref is not None: _a.plot(Xref[:,0], Xref[:,1], label='reference')
    if Yref is not None: _a.plot(Yref[:,0,0], Yref[:,0,1], label='reference')
    _a.set_title('2D')
    _a.axis('equal')
    _a.legend(loc='best')
    _f.canvas.set_window_title('2D trajectory')
    return _f, _a

def plot_trajectory_chrono(time, X=None, U=None, Yref=None, Xref=None, _f=None, _a=None, title=None):
    _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
    _a = _f.subplots(3, 2) if _a is None else _a
    if X is not None: _a[0,0].plot(time, X[:, ddd.Aircraft.s_x], label='aircraft')
    if Xref is not None: _a[0,0].plot(time, Xref[:, ddd.Aircraft.s_x], label='reference')
    if Yref is not None: _a[0,0].plot(time, Yref[:,0,0], label='reference')
    decorate(_a[0,0], title='x', xlab='s', ylab='m', legend=True)

    if X is not None: _a[0,1].plot(time, X[:, ddd.Aircraft.s_y], label='aircraft')
    if Xref is not None: _a[0,1].plot(time, Xref[:, ddd.Aircraft.s_y], label='reference')
    if Yref is not None: _a[0,1].plot(time, Yref[:,0,1], label='reference')
    decorate(_a[0,1], title='y', xlab='s', ylab='m', legend=True)

    if X is not None: _a[1,0].plot(time, np.rad2deg(X[:,ddd.Aircraft.s_psi]), label='aircraft')
    if Xref is not None: _a[1,0].plot(time, np.rad2deg(Xref[:, ddd.Aircraft.s_psi]), label='reference')
    decorate(_a[1,0], title='Yaw', xlab='s', ylab='deg', legend=True)

    if X is not None: _a[1,1].plot(time, X[:,ddd.Aircraft.s_v], label='aircraft')
    if Xref is not None: _a[1,1].plot(time, Xref[:, ddd.Aircraft.s_v], label='reference')
    if U is not None: _a[1,1].plot(time, (U[:,ddd.Aircraft.i_v]), label='setpoint')
    decorate(_a[1,1], title='Vel', xlab='s', ylab='m/s', legend=True)

    if X is not None: _a[2,0].plot(time, np.rad2deg(X[:,ddd.Aircraft.s_phi]), label='aircraft')
    if Xref is not None: _a[2,0].plot(time, np.rad2deg(Xref[:, ddd.Aircraft.s_phi]), label='reference')
    if U is not None: _a[2,0].plot(time, np.rad2deg(U[:,ddd.Aircraft.i_phi]), label='setpoint')
    decorate(_a[2,0], title='Roll', xlab='s', ylab='deg', legend=True)
    if title is not None: _f.canvas.set_window_title(title)
    _f.canvas.set_window_title('State trajectory')
    return _f, _a

def plot_flat_output_trajectory_chrono(time, Yref, _f=None, _a=None):
    _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
    _a = _f.subplots(ddg.Trajectory.nder+1, ddg.Trajectory.ncomp) if _a is None else _a
    for _i in range(ddg.Trajectory.nder+1):
        _a[_i,0].plot(time, Yref[:,_i,ddg.Trajectory.cx])#, label=f'x^({i})')
        decorate(_a[_i,0], title=f'$x^{{({_i})}}$', xlab='s', ylab=f'$m/s^{{{_i}}}$', legend=True)
        _a[_i,1].plot(time, Yref[:,_i,ddg.Trajectory.cy])
        decorate(_a[_i,1], title=f'$y^{{({_i})}}$', xlab='s', ylab=f'$m/s^{{{_i}}}$', legend=True)
    _f.canvas.set_window_title('Flat output trajectory')



