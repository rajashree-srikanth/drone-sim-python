#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Trajectory planning with Opty (direct collocation)
 Multiple vehicles
"""

import sys, os, argparse
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation

import d2d.ploting as d2p
import d2d.opty_utils as d2ou

class AircraftSet:
    def __init__(self, n=2):
        self.st = sym.symbols('t')
        self.aircraft = [d2ou.Aircraft(self.st) for i in range(n)]
        self._state_symbols = tuple(np.array([ac._state_symbols for ac in self.aircraft]).flatten())
        self._input_symbols = tuple(np.array([ac._input_symbols for ac in self.aircraft]).flatten())

    def get_eom(self, wind, g=9.81):
        return sym.Matrix.vstack(*[_a.get_eom(wind, g) for _a in self.aircraft])


class Planner:
    def __init__(self, scen, initialize=True):
        self.scen = scen
        self.obj_scale = scen.obj_scale

        self.wind = d2ou.WindField()
        self.acs = AircraftSet(n=2)
        #foo = self.acs.get_eom(self.wind)
        #breakpoint()
        self.num_nodes, self.time_step, self.duration = d2ou.planner_timing(scen.t0, scen.t1, scen.hz)


        # Nodes indexing

        
        # Known system parameters.
        self._par_map = collections.OrderedDict()
        # Symbolic instance constraints, i.e. initial and end conditions.
        def ic(_ac, _p, _t): return (_ac._sx(_t)-_p[0], _ac._sy(_t)-_p[1], _ac._spsi(_t)-_p[2])
        self._instance_constraints = [ic(_ac, _p, scen.t0) for _ac, _p in zip(self.acs.aircraft, scen.p0s)]
        self._instance_constraints += [ic(_ac, _p, scen.t1) for _ac, _p in zip(self.acs.aircraft, scen.p1s)]
        
        # Bounds
        self._bounds = {}

        # Objective
        obj = scen.cost

        if initialize:
            self.prob =  opty.direct_collocation.Problem(lambda _free: obj.cost(_free, self),
                                                         lambda _free: obj.cost_grad(_free, self),
                                                         self.acs.get_eom(self.wind),
                                                         self.acs._state_symbols,
                                                         self.num_nodes,
                                                         self.time_step,
                                                         known_parameter_map=self._par_map,
                                                         instance_constraints=self._instance_constraints,
                                                         bounds=self._bounds,
                                                         parallel=False)
    def run(self, initial_guess=None):
        initial_guess = np.random.randn(self.prob.num_free) if initial_guess is None else initial_guess
        self.solution, info = self.prob.solve(initial_guess)   
        #self.interpret_solution()
        
def parse_command_line():
    parser = argparse.ArgumentParser(description='Trajectory planning.')
    parser.add_argument('--scen', help='scenario index', default=None)
    parser.add_argument('--force', help='force recompute', action='store_true', default=False)
    parser.add_argument('--list', help='list all known scenarios', action='store_true', default=False)
    parser.add_argument('--save', help='save plot', action='store', default=None)
    args = parser.parse_args()
    return args


class exp_0:
    t0, t1, hz = 0., 15., 50.
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( 0., 100.,  0.,  0., 10.), )
    #p0s = (( 0.,  0.,   0.,  0., 10.), ( 100.,  0.,  np.pi, 0., 10.))
    #p1s = (( 0., 100.,  0.,  0., 10.), (   0.,  0.,  np.pi, 0., 10.))
    cost = d2ou.CostAirVel(12.)
    obj_scale = 1.e1
    x_constraint, y_constraint = None, None
    phi_constraint = (-np.deg2rad(30.), np.deg2rad(30.))
    v_constraint = (9., 12.)
    name = 'exp_0'
    desc = 'foo'
    
class exp_1(exp_0):
    pass

scens = [exp_0, exp_1]
def desc_all_scens():
    return '\n'.join([f'{i}: {s.name} {s.desc}' for i, s in enumerate(scens)])
def get_scen(idx): return scens[idx]

def main():
    args = parse_command_line()
    if args.list or not args.scen:
        print(desc_all_scens())
        return
    try:
        scen = get_scen(int(args.scen))
    except ValueError:
        print(f'unknown scen {args.scen}')
        return
        
    _p = Planner(scen, initialize=True)
    #_p.run()
    

if __name__ == '__main__':
    main()
