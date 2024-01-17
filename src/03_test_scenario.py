#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

import d2d.guidance as dg
import d2d.utils as du
import d2d.ploting as d2plot
import d2d.animation as dda
import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.scenario as dds
from d2d.dynamic import Aircraft

def test_scenario(scen, show_Yref, show_Xref, show_anim, show_2d):
    Yrefs = []
    for trj in scen.trajs:
        Yrefs.append(np.array([trj.get(t) for t in scen.time]))
    if show_Yref:
        _f, _a = None, None
        for i, Yref in enumerate(Yrefs):
            _f, _a = d2plot.plot_flat_output_trajectory_chrono(scen.time, Yref, _f, _a, f'ref_{i}') 

    Wrefs, Xrefs, Urefs = [], [], []
    ac = Aircraft() # FIXME, needs per Yref instance
    if show_Xref:
        _f, _a = None, None
        for Yref in Yrefs:
            Wref = np.array([scen.windfield.sample(_t, _l) for _t, _l in zip(scen.time, Yref[:,0,:])])
            Wrefs.append(Wref)
            XUref = [dg.DiffFlatness.state_and_input_from_output(Y, W, ac)  for Y,W in zip(Yref, Wref)]
            Xref, Uref = np.array([_Xu[0] for _Xu in XUref]), [_Xu[1] for _Xu in XUref]
            Xrefdot = np.array([_Xu[2] for _Xu in XUref])
            _f, _a = d2plot.plot_trajectory_chrono(scen.time, X=None, U=None, Xref=Xref, _f=_f, _a=_a)
            if 0: # check vadot: yes
                _a[2,1].plot(scen.time, Xrefdot[:, Aircraft.s_va], label='df')
                _a[2,1].plot(scen.time, 100*np.gradient(Xref[:, Aircraft.s_va]), alpha=0.5, label='numeric')
                d2plot.decorate(_a[2,1], title='$\dot{v}_a$', legend=True)
            if 1: # check psidot: yes
                _a[2,1].plot(scen.time, Xrefdot[:, Aircraft.s_psi], label='df')
                psi_dot_num = 100*np.gradient(Xref[:, Aircraft.s_psi])
                for i in range(len(psi_dot_num)):
                    if abs(psi_dot_num[i]) > 10:psi_dot_num[i]= psi_dot_num[i-1]
                _a[2,1].plot(scen.time, psi_dot_num, alpha=0.5, label='numeric')
                #_a[2,1].set_ylim(-0.5, 0.5)
                d2plot.decorate(_a[2,1], title='$\dot{\psi}$', legend=True)
            
    if show_2d:
        _f, _a = None, None
        for i, Yref in enumerate(Yrefs):
            _f, _a = d2plot.plot_trajectory_2d(scen.time,X=None, U=None, Yref=Yref, Xref=None, _f=_f, _a=_a, label=f'ref_{i}')

    X, U = None, None
    #Yref, Xref = None, None
    #Yref = [Yref1]
    #Yrefs = [Yref1, Yref2, Yref3]
    Xref = None
    anim = None
    if show_anim:
        anim = dda.animate(scen.time, X, U, Yrefs, Xref, title=f'scenario: {scen.name}', extends=scen.extends)
    return anim


def parse_command_line():
    parser = argparse.ArgumentParser(description='Plot a trajectory.')
    parser.add_argument('--scen', help='the name of the scenario', default=None)
    parser.add_argument('--anim', help='plot animation', action='store_true', default=False)
    parser.add_argument('--twod', help='plot 2d track', action='store_true', default=False)
    parser.add_argument('--X', help='plot state', action='store_true', default=False)
    parser.add_argument('--Y', help='plot output', action='store_true', default=False)
    parser.add_argument('--list', help='list all known trajectories', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    if args.list or not args.scen:
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
    except KeyError:
        print('unknown scenario {}'.format(args.scen))
        return
    anim = test_scenario(scen, show_Yref=args.Y, show_Xref=args.X, show_anim=args.anim, show_2d=args.twod)
    plt.show()
    
if __name__ == "__main__":
    main()
