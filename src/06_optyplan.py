#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Trajectory planning with Opty (direct collocation)
"""
seed = None
#seed = 12345

#
# TODO
#    - initial guess
#    - constant speed

import sys, os, argparse
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation

import d2d.ploting as d2p
import d2d.opty_utils as d2ou
import d2d.opty_planner as d2op
import d2d.optyplan_scenarios as d2oscen


def compute_or_load(_p, force_recompute=False, filename='/tmp/optyplan.npz', tol=1e-5, max_iter=1500, initial_guess=None):
    if force_recompute or not os.path.exists(filename):
        print(f'{filename} { os.path.exists(filename)}')
        _p.configure(tol, max_iter)
        initial_guess = _p.get_initial_guess()
        _p.run(initial_guess)
        if 0:
            _p.prob.plot_objective_value()
            _p.prob.plot_trajectories(_p.solution)
            #_p.prob.plot_constraint_violations(_p.solution)
        _p.save_solution(filename)
    else:
        _p.load_solution(filename)
        
def parse_command_line():
    parser = argparse.ArgumentParser(description='Trajectory planning.')
    parser.add_argument('--scen', help='index of the scenario', default=None)
    parser.add_argument('--force', help='force recompute', action='store_true', default=False)
    parser.add_argument('--list', help='list all known trajectories', action='store_true', default=False)
    parser.add_argument('--save', help='save plot', action='store', default=None)
    args = parser.parse_args()
    return args
    
def main():
    args = parse_command_line()
    if args.list or not args.scen:
        print(d2oscen.desc_all()); return
    try:
        scen_idx = int(args.scen)
        scen = d2oscen.scens[scen_idx]
        print(f'#Scenario:\n{d2oscen.desc_one(scen_idx)}')
    except ValueError:
        print(f'unknown scenario: {args.scen}'); return

    f1, a1, f2, a2 = None, None, None, None
    for _case in range(scen.ncases):
        scen.set_case(_case)
        cache_filename = f'./cache/optyplan_{scen.name}_{_case}.npz' if scen.ncases>1 else f'./cache/optyplan_{scen.name}.npz'
        _p = d2op.Planner(scen, initialize=args.force or not os.path.exists(cache_filename))
        import time
        start = time.time()
        print('Planner initialized')
        compute_or_load(_p, args.force, cache_filename, tol=scen.tol, max_iter=scen.max_iter, initial_guess=None)
        end = time.time()
        print(f'Planner ran {end-start}s')
        f1, a1 = d2ou.plot2d(_p, args.save, f1, a1, scen.label(_case))
        f2, a2 = d2ou.plot_chrono(_p, args.save, f2, a2)
    plt.show()

    
if __name__ == '__main__':
    main()




