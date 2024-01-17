#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

import d2d.guidance as dg
import d2d.utils as du
import d2d.ploting as d2plot
import d2d.dynamic as ddd
import d2d.animation as dda
import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf


def display_trajectory(traj, show_Yref=True, show_Xref=True, show_2d=True):
    t0, t1, dt = 0, traj.duration, 0.01
    time = np.arange(t0, t1, dt)
    Yref = np.array([traj.get(t) for t in time])
    Xref = None
    if show_Yref:
        d2plot.plot_flat_output_trajectory_chrono(time, Yref)
    if show_Xref:
        aircraft, windfield = ddd.Aircraft(), dg.WindField()
        Wref = np.array([windfield.sample(_t, _l) for _t, _l in zip(time, Yref[:,0,:])])
        XUref = [dg.DiffFlatness.state_and_input_from_output(Y, W, aircraft)  for Y,W in zip(Yref, Wref)]
        Xref, Uref = np.array([_Xu[0] for _Xu in XUref]), [_Xu[1] for _Xu in XUref]
        d2plot.plot_trajectory_chrono(time, X=None, U=None, Xref=Xref, _f=None, _a=None)
    if show_2d:
        d2plot.plot_trajectory_2d(time, X=None, U=None, Yref=Yref, Xref=Xref)
    plt.show()


def display_animation(traj):
    t0, t1, dt = 0, traj.duration, 0.01
    time = np.arange(t0, t1, dt)

    Yref = np.array([traj.get(t) for t in time])
    Yrefs = [Yref]
    windfield = dg.WindField([0, 0])
    Wref = np.array([windfield.sample(_t, _l) for _t, _l in zip(time, Yref[:,0,:])])
    X, U, Xref = None, None, None
    anim = dda.animate(time, X, U, Yrefs, Xref, title=f'trajectory: {traj.name}', extends=traj.extends)
    return anim


def parse_command_line():
    parser = argparse.ArgumentParser(description='Plot a trajectory.')
    parser.add_argument('--traj', help='the name of the trajectory', default=None)
    parser.add_argument('--anim', help='plot animation', action='store_true', default=False)
    parser.add_argument('--twod', help='plot 2d track', action='store_true', default=False)
    parser.add_argument('--X', help='plot state', action='store_true', default=False)
    parser.add_argument('--Y', help='plot output', action='store_true', default=False)
    parser.add_argument('--list', help='list all known trajectories', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    if args.list or not args.traj:
        ddtf.print_available()
        return
    try:
        traj_idx = int(args.traj)
        args.traj = sorted(ddtf.trajectories.items())[traj_idx][0]
    except ValueError:
        pass
    try:
        print('loading trajectory: {}'.format(args.traj))
        traj, desc = ddtf.get(args.traj)
        print('  description: {}'.format(desc))
    except KeyError:
        print('unknown trajectory {}'.format(args.traj))
        return
    #ddt.check_si(traj)
    if args.anim:
        anim = display_animation(traj)
    display_trajectory(traj, show_Yref=args.Y, show_Xref=args.X, show_2d=args.twod)
    plt.show()
    
if __name__ == "__main__":
    main()
