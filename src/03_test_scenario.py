#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

import d2d.guidance as dg
import d2d.utils as du
import d2d.animation as dda
import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.scenario as dds


def test_scenario(scn):
    Yrefs = []
    for trj in scn.trajs:
        Yrefs.append(np.array([trj.get(t) for t in scn.time]))
    #Wref = np.array([windfield.sample(_t, _l) for _t, _l in zip(time, Yref[:,0,:])])
    X, U = None, None
    #Yref, Xref = None, None
    #Yref = [Yref1]
    #Yrefs = [Yref1, Yref2, Yref3]
    Xref = None
    anim = dda.animate(scn.time, X, U, Xref, Yrefs, title=f'scenario: {scn.name}', extends=scn.extends)
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
    except KeyError:
        print('unknown scenario {}'.format(args.scen))
        return
    if args.anim:
        anim = test_scenario(scen)
    plt.show()
    
if __name__ == "__main__":
    main()
