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

#
# Dynamic Simulation
#
   
def run_simulation(time, aircraft, windfield, ctl, X0, perts):
    X,U = np.zeros((len(time), ddyn.Aircraft.s_size)), np.zeros((len(time), ddyn.Aircraft.i_size))
    Yref = np.array([ctl.traj.get(t) for t in time])
    X[0] = X0
    for i in range(1, len(time)):
        U[i-1] = ctl.get(X[i-1], time[i-1])#, time)
        X[i] = aircraft.disc_dyn(X[i-1], U[i-1], windfield, time[i-1], time[i]-time[i-1])
        X[i] += perts[i]
    U[-1] = ctl.get(X[-1], time[-1])
    return X, U, Yref

def test_simulation(scen, show_chrono, show_2d, show_anim, show_extra):
    windfield = scen.windfield
    aircrafts = [ddyn.Aircraft() for i in range(len(scen.trajs))]
    #ctls = [ddg.PurePursuitControler(traj) for traj in scen.trajs]
    ctls = [ddg.DFFFController(traj, ac, windfield) for traj, ac in zip(scen.trajs, aircrafts)]
    #np.set_printoptions(precision=2, linewidth=600)
   
    Xs, Us, Yrefs = [], [], []
    for traj, aircraft, X0, ctl, pert in zip(scen.trajs, aircrafts, scen.X0s, ctls, scen.perts):
        X, U, Yref = run_simulation(scen.time, aircraft, windfield, ctl, X0, pert)
        Xs.append(X); Us.append(U); Yrefs.append(Yref)
        
    if show_2d:
        d2plot.plot_trajectory_2d(scen.time, X, U, Yref)
    if show_chrono:
        #ctls[0].draw_debug(_f, _a)
        d2plot.plot_trajectories_chrono(scen.time, Xs, Us, Yrefs)
        
        
    if show_anim:
        extra = [np.array(ctl.carrot), np.array(ctl.ref_pos)] if show_extra else None
        anim = dda.animate(scen.time, Xs, Us, Yrefs, Xref=None, Extra=extra, title=f'Simulation (scen {scen.name})', extends=scen.extends)
        #dda.save_anim('/home/poine/tmp/foo.gif', anim, 0.01)
        #dda.save_anim('/tmp/foo.apng', anim, 0.01)

    else: anim=None
    return anim



def parse_command_line():
    parser = argparse.ArgumentParser(description='Runs a simulation.')
    parser.add_argument('--scen', help='the name of the trajectory', default=None)
    parser.add_argument('--anim', help='plot animation', action='store_true', default=False)
    parser.add_argument('--twod', help='plot 2d track', action='store_true', default=False)
    parser.add_argument('--X', help='plot state', action='store_true', default=False)
    parser.add_argument('--Y', help='plot output', action='store_true', default=False)
    parser.add_argument('--list', help='list all available scenarios', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    if args.list:
        dds.print_available()
        return
    try:
        scen_idx = int(args.scen)
        args.scen = sorted(dds._scenarios.items())[scen_idx][0]
        print(f'resolved scenario {scen_idx} to {args.scen}'.format(args.scen))
    except ValueError:
        pass
    try:
        print('loading scenario: {}'.format(args.scen))
        scen, desc = dds.get(args.scen)
        print('  description: {}'.format(desc))
        print(f'{scen.summarize()}')
    except KeyError:
        print('unknown scenario {}'.format(args.scen))
        return
    show_extra = False
    anim = test_simulation(scen, args.X, args.twod, args.anim, show_extra)
    plt.show()
    
if __name__ == "__main__":
    main()
