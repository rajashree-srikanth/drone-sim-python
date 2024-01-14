#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import d2d.guidance as ddg

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
    
def plot_trajectory_2d(time, X=None, U=None, Xref=None, _f=None, _a=None):
    _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
    _a = _f.subplots(1, 1) if _a is None else _a
    if X is not None: _a.plot(X[:,0], X[:,1], label='aircraft')
    #breakpoint()
    if Xref is not None: _a.plot(Xref[:,0,0], Xref[:,0,1], label='reference')
    #_a.plot(Xref[:,0], Xref[:,1], label='reference')
    _a.set_title('2D')
    _a.axis('equal')
    _a.legend(loc='best')
    return _f, _a

def plot_trajectory_chrono(time, X=None, U=None, Xref=None, _f=None, _a=None, title=None):
    _f = plt.figure(tight_layout=True, figsize=[16., 9.]) if _f is None else _f
    _a = _f.subplots(3, 2) if _a is None else _a
    if X is not None: _a[0,0].plot(time, X[:, ddg.Aircraft.s_x], label='aircraft')
    if Xref is not None: _a[0,0].plot(time, Xref[:, ddg.Aircraft.s_x], label='reference')
    decorate(_a[0,0], title='x', xlab='s', ylab='m', legend=True)

    if X is not None: _a[0,1].plot(time, X[:, ddg.Aircraft.s_y], label='aircraft')
    if Xref is not None: _a[0,1].plot(time, Xref[:, ddg.Aircraft.s_y], label='reference')
    decorate(_a[0,1], title='y', xlab='s', ylab='m', legend=True)

    if X is not None: _a[1,0].plot(time, np.rad2deg(X[:,ddg.Aircraft.s_psi]), label='aircraft')
    if Xref is not None: _a[1,0].plot(time, np.rad2deg(Xref[:, ddg.Aircraft.s_psi]), label='reference')
    decorate(_a[1,0], title='Yaw', xlab='s', ylab='deg', legend=True)

    if X is not None: _a[1,1].plot(time, X[:,ddg.Aircraft.s_v], label='aircraft')
    if Xref is not None: _a[1,1].plot(time, Xref[:, ddg.Aircraft.s_v], label='reference')
    if U is not None: _a[1,1].plot(time, (U[:,ddg.Aircraft.i_v]), label='setpoint')
    decorate(_a[1,1], title='Vel', xlab='s', ylab='m/s', legend=True)

    if X is not None: _a[2,0].plot(time, np.rad2deg(X[:,ddg.Aircraft.s_phi]), label='aircraft')
    if Xref is not None: _a[2,0].plot(time, np.rad2deg(Xref[:, ddg.Aircraft.s_phi]), label='reference')
    if U is not None: _a[2,0].plot(time, np.rad2deg(U[:,ddg.Aircraft.i_phi]), label='setpoint')
    decorate(_a[2,0], title='Roll', xlab='s', ylab='deg', legend=True)
    if title is not None: _f.canvas.set_window_title(title)
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



#
# Animations
#
    
# import matplotlib.animation as animation
# import matplotlib.image, matplotlib.offsetbox, matplotlib.transforms

# def animate(time, X=None, U=None, Xref=None, Yrefs=None, title=None, _drawings=False, _imgs=True, figure=None, ax=None, extends=None):
#     if extends is None: extends = (0, 100, 0, 100)
#     _xmin, _xmax, _ymin, _ymax = extends
#     time_factor = 1. # Nope :(2.
#     _decim = int(4*time_factor) # sim at 100hz, I want anim at 25 fps
#     fig = figure or plt.figure(figsize=(10., 8.))
#     if ax is None:
#         ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(_xmin, _xmax),
#                              ylim=(_ymin, _ymax), facecolor=(0.5, 0.9, 0.9))
#     if title is not None: ax.set_title(title, {'fontsize': 20 })
#     ax.grid()
#     time_template = 'time = {:0.1f}s'
#     time_text = ax.text(0.025, 0.92, 'Hello', transform=ax.transAxes)
#     _line_ac, = ax.plot([], [], '-', lw=3, color='r', zorder=1, label='aircraft')
#     marker_refs = []
#     if Yrefs is not None:
#         for i, Yref in enumerate(Yrefs):
#             _marker_ref, = ax.plot([], [], 'o-', lw=3, color='g', zorder=1, label=f'reference {i}')
#             marker_refs.append(_marker_ref)
#             _track_ref = ax.plot(Yref[:,0,0], Yref[:,0,1], label='_ref', alpha=0.5)
#     if X is not None:
#         _track_ac = ax.plot(X[:,0], X[:,1], label='_ac', alpha=0.5)
         
#     ax.legend(loc='best')

#     #breakpoint()
#     #if Yrefs is not None:
#     #    for Yref in Yrefs:
#     #        ax.plot(Yref[:,0,0], Yref[:,0,1], label='ref')
    
#     def init():
#         _line_ac.set_data([], [])
#         for _l in marker_refs:
#             _l.set_data([], [])
#         return [time_text, _line_ac] + marker_refs

#     def _get_points(x,y,psi,l=0.5):
#         _c = np.array([x, y])                             # center of gravity
#         _b = l*np.array([np.cos(psi), np.sin(psi)])       # front facing unit vect
#         _p1 = _c + _b                                     # front
#         _p2 = _c - _b                                     # back
#         return [_p1[0], _p2[0]], [_p1[1], _p2[1]]

#     def _get_points2(x,y,psi,l=0.5): return [[x]], [[y]]

        
#     def animate(i):
#         if X is not None:
#             x, y, psi, phi, v = X[int(i*_decim), :]
#             ps = _get_points(x,y,psi,l=0.5)
#             _line_ac.set_data(ps)
#         if Yrefs is not None:
#             for _l, Yref in zip(marker_refs, Yrefs):
#                 xr, yr = Yref[int(i*_decim), :][0]
#                 prefs = _get_points2(xr, yr, 0.)
#                 _l.set_data(prefs)
#         time_text.set_text(time_template.format(i*_decim * dt))
#         return [time_text, _line_ac] + marker_refs

#     dt = time[1]-time[0]
#     dt_mili = dt*1000*_decim
#     anim = animation.FuncAnimation(fig, animate, np.arange(1, len(time)/_decim),
#                                    interval=dt_mili, blit=True, init_func=init, repeat_delay=200)

#     return anim

# def save_anim(filename, an, dt):
#     print('encoding animation video, please wait, it will take a while')
#     _start = time.time()
#     fps = 1./dt/4; print(f'dt {dt} fps {fps}')
#     an.save(filename, writer=animation.PillowWriter(fps=1./dt/4)) # gif?
#     _end = time.time()
#     print(f'video encoded, saved to {filename}, Bye (took {_end-_start:.1f} s)')


