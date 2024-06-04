#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Trajectory planning with Opty (direct collocation)
 Multiple vehicles
"""

seed = None
#seed = 12345


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
        self.nb_aicraft = n
        self.aircraft = [d2ou.Aircraft(self.st, i) for i in range(self.nb_aicraft)]
        self._state_symbols = tuple(np.array([ac._state_symbols for ac in self.aircraft]).flatten())
        self._input_symbols = tuple(np.array([ac._input_symbols for ac in self.aircraft]).flatten())

    def get_eom(self, wind, g=9.81):
        return sym.Matrix.vstack(*[_a.get_eom(wind, g) for _a in self.aircraft])


class SetCostNull:
    def cost(self, free, _p): return 0.
    def cost_grad(self, free, _p): return np.zeros_like(free)
         
class SetCostAirvel:
    def __init__(self, vsp=10.):
        self.vsp = vsp
    def cost(self, free, _p):
        sum_err_vel_squared = np.sum([np.sum(np.square(free[slice_v]-self.vsp)) for slice_v in _p._slice_v])
        return _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * sum_err_vel_squared
    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        for _s in _p._slice_v:
            grad[_s] = _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * 2*(free[_s]-self.vsp)
        return grad
        
class SetCostBank:
    def cost(self, free, _p):
        sum_phi_squared = np.sum([np.sum(np.square(free[slice_phi])) for slice_phi in _p._slice_phi])
        return _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * sum_phi_squared
    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        for _s in _p._slice_phi:
            grad[_s] = _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * 2*free[_s]
        return grad

class SetCostInput:
    def __init__(self, vsp=10., kv=1., kphi=1.):
        self.vsp, self.kv, self.kphi = vsp, kv, kphi

    def cost(self, free, _p):
        sum_phi_squared = np.sum([np.sum(np.square(free[slice_phi])) for slice_phi in _p._slice_phi])
        sum_err_vel_squared = np.sum([np.sum(np.square(free[slice_v]-self.vsp)) for slice_v in _p._slice_v])
        return _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * (self.kv*sum_err_vel_squared + self.kphi*sum_phi_squared)

    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        for _s in _p._slice_phi:
            grad[_s] = self.kphi*2*free[_s]
        for _s in _p._slice_v:
            grad[_s] =  self.kv*2*(free[_s]-self.vsp)
        grad *= _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft
        return grad
    
    
# FIXME: all airplane don't start at the same time, must be smarter with time
import itertools
class Planner:
    def __init__(self, scen, initialize=True):
        self.scen = scen
        self.obj_scale = scen.obj_scale

        self.wind = d2ou.WindField()
        self.acs = AircraftSet(n=len(scen.p0s))
        #foo = self.acs.get_eom(self.wind)
        #breakpoint()
        self.num_nodes, self.time_step, self.duration = d2ou.planner_timing(scen.t0, scen.t1, scen.hz)

        
        # Nodes indexing
        self._slice_x   = [slice((0+3*_i)*self.num_nodes, (1+3*_i)*self.num_nodes, 1) for _i in range(self.acs.nb_aicraft)]
        self._slice_y   = [slice((1+3*_i)*self.num_nodes, (2+3*_i)*self.num_nodes, 1) for _i in range(self.acs.nb_aicraft)]
        self._slice_psi = [slice((2+3*_i)*self.num_nodes, (3+3*_i)*self.num_nodes, 1) for _i in range(self.acs.nb_aicraft)]
        i_in = self.num_nodes * self.acs.nb_aicraft * 3
        self._slice_phi = [slice(i_in+_i*self.num_nodes, i_in+(1+_i)*self.num_nodes, 1) for _i in range(self.acs.nb_aicraft)]
        i_in += self.num_nodes * self.acs.nb_aicraft
        self._slice_v   = [slice(i_in+_i*self.num_nodes, i_in+(1+_i)*self.num_nodes, 1) for _i in range(self.acs.nb_aicraft)]
        #breakpoint()
        
        # Known system parameters.
        self._par_map = collections.OrderedDict()
        # Symbolic instance constraints, i.e. initial and end conditions.
        def ic(_ac, _p, _t): return (_ac._sx(_t)-_p[0], _ac._sy(_t)-_p[1], _ac._spsi(_t)-_p[2])
        self._instance_constraints = [ic(_ac, _p, scen.t0) for _ac, _p in zip(self.acs.aircraft, scen.p0s)]
        self._instance_constraints += [ic(_ac, _p, scen.t1) for _ac, _p in zip(self.acs.aircraft, scen.p1s)]
        self._instance_constraints = tuple([i for i in itertools.chain(*self._instance_constraints)])
        #print(self._instance_constraints)
        # Bounds
        self._bounds = {}
        for _ac in self.acs.aircraft:
            self._bounds[_ac._sphi(_ac._st)] = scen.phi_constraint
            self._bounds[_ac._sv(_ac._st)] = scen.v_constraint
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

    def run(self, initial_guess=None, tol=1e-8, max_iter=500):
        initial_guess = np.random.randn(self.prob.num_free) if initial_guess is None else initial_guess

        
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000
        self.solution, info = self.prob.solve(initial_guess)   
        #self.interpret_solution()
        if 1:
            # Make some plots
            self.prob.plot_trajectories(self.solution)
            #self.prob.plot_constraint_violations(self.solution)
            #self.prob.plot_objective_value()
            
    def get_initial_guess(self, what='rnd'):
        initial_guess = np.zeros(self.prob.num_free)
        rng = np.random.default_rng(seed)
        if what == 'rnd':
            for i in range(self.acs.nb_aicraft):
                cx = [-100, 100] or self.scen.x_constraint
                initial_guess[self._slice_x[i]] =  rng.uniform(cx[0],cx[1], self.num_nodes) 
                cy = [-100, 100] or self.scen.y_constraint
                initial_guess[self._slice_y[i]] =  rng.uniform(cx[0],cx[1], self.num_nodes) 
                initial_guess[self._slice_psi[i]] =  rng.uniform(-np.pi, np.pi, self.num_nodes)
                initial_guess[self._slice_phi[i]] =  rng.uniform(self.scen.phi_constraint[0], self.scen.phi_constraint[1], self.num_nodes) 
                initial_guess[self._slice_v[i]] =  rng.uniform(self.scen.v_constraint[0], self.scen.v_constraint[1], self.num_nodes) 
                
        else : # triangle
            self.initial_guesses = [d2ou.triangle(np.array(p0)[:2], np.array(p1)[:2], self.scen.vref, self.duration, self.num_nodes, go_left=-1.) for p0, p1 in zip(self.scen.p0s, self.scen.p1s)]
            for i, ig in enumerate(self.initial_guesses):
                initial_guess[self._slice_x[i]], initial_guess[self._slice_y[i]], initial_guess[self._slice_psi[i]], initial_guess[self._slice_phi[i]], initial_guess[self._slice_v[i]] = ig
        
        return initial_guess
            
        

    def interpret_solution(self):
        self.sol_time = np.linspace(0.0, self.duration, num=self.num_nodes)
        self.sol_x   = [self.solution[_sx] for _sx in self._slice_x]
        self.sol_y   = [self.solution[_sy] for _sy in self._slice_y]
        self.sol_psi   = [self.solution[_spsi] for _spsi in self._slice_psi]
        self.sol_v   = [self.solution[_sv] for _sv in self._slice_v]
        self.sol_phi   = [self.solution[_sphi] for _sphi in self._slice_phi]


def plot2d(_p, _f=None, _a=None, label=''):
    _f = _f or plt.figure()
    _a = _a or plt.gca()
    for i in range(_p.acs.nb_aicraft):
        _a.plot(_p.sol_x[i], _p.sol_y[i], solid_capstyle='butt', label=label)
    d2p.decorate(_a, title='$2D$', xlab='x in m', ylab='y in m', legend=True, xlim=None, ylim=None, min_yspan=0.1)  
    _a.axis('equal')
    return _f, _a
 
def plot_chrono(_p, _f=None, _a=None, label=''):
    if _f is None: fig, axes = plt.subplots(5, 1)
    else: fig, axes = _f, _a
    for i in range(_p.acs.nb_aicraft):
        axes[0].plot(_p.sol_time, _p.sol_x[i], label=f'{i}')
        d2p.decorate(axes[0], title='x', xlab='t in s', ylab='m', legend=None, xlim=None, ylim=None, min_yspan=None)
        axes[1].plot(_p.sol_time, _p.sol_y[i], label=f'{i}')
        d2p.decorate(axes[1], title='y', xlab='t in s', ylab='m', legend=None, xlim=None, ylim=None, min_yspan=None)
        axes[2].plot(_p.sol_time, np.rad2deg(_p.sol_psi[i]), label=f'{i}')
        d2p.decorate(axes[2], title='$\\psi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
        axes[3].plot(_p.sol_time, np.rad2deg(_p.sol_phi[i]), label=f'{i}')
        d2p.decorate(axes[3], title='$\\phi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
        axes[4].plot(_p.sol_time, _p.sol_v[i], label=f'{i}')
        d2p.decorate(axes[4], title='$v_a$', xlab='t in s', ylab='m/s', legend=None, xlim=None, ylim=None, min_yspan=0.1)
    return fig, axes
  
def parse_command_line():
    parser = argparse.ArgumentParser(description='Trajectory planning.')
    parser.add_argument('--scen', help='scenario index', default=None)
    parser.add_argument('--force', help='force recompute', action='store_true', default=False)
    parser.add_argument('--list', help='list all known scenarios', action='store_true', default=False)
    parser.add_argument('--save', help='save plot', action='store', default=None)
    args = parser.parse_args()
    return args


class exp_0:
    t0, t1, hz = 0., 7.5, 50.
    dx, dy = 0.,50.
    dx, dy = 50, 50
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( dx, dy,  np.pi/2,  0., 10.), )
    vref = 12.
    #initial_guess = 'rnd'
    initial_guess = 'tri'
    tol, max_iter = 1e-5, 5000
    tol, max_iter = 1e-5, 1500
    #tol, max_iter = 1e-5, 0
    #cost, obj_scale = SetCostBank(), 1.
    #cost, obj_scale = SetCostAirvel(vsp=12.), 1.
    cost, obj_scale = SetCostInput(vsp=14., kv=1., kphi=1.), 1.
    #cost; obj_scale = SetCostNull(), 1.
    x_constraint, y_constraint = None, None
    phi_constraint = (-np.deg2rad(30.), np.deg2rad(30.))
    v_constraint = (9., 15.)
    name = 'exp_0'
    desc = 'foo'
    
class exp_1(exp_0):
    t1 = 5.5
    p0s = (( 0.,  0.,   0.,  0., 10.), ( 50.,  0.,  np.pi, 0., 10.))
    p1s = (( 50., 0.,   0.,  0., 10.), (  0.,  0.,  np.pi, 0., 10.))


class exp_2(exp_0):
    t1 = 5.5
    vref = 12.
    overtime=1.5#1.75
    d = t1*vref/2 / overtime
    p0s = ((-d, 0., 0., 0., vref), ( d,  0.,  np.pi, 0., vref), ( 0.,  d, -np.pi/2,  0., vref), ( 0., -d,  np.pi/2, 0., vref))
    p1s = (( d, 0., 0., 0., vref), (-d,  0.,  np.pi, 0., vref), ( 0., -d, -np.pi/2,  0., vref), ( 0.,  d,  np.pi/2, 0., vref))
    cost, obj_scale = SetCostInput(vsp=vref, kv=1., kphi=1.), 1.
     

scens = [exp_0, exp_1, exp_2]
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
    initial_guess = None
    initial_guess = _p.get_initial_guess(scen.initial_guess)
    _p.run(initial_guess=initial_guess, tol=scen.tol, max_iter=scen.max_iter)
    _p.interpret_solution()
    plot2d(_p)
    plot_chrono(_p)
    plt.show()
    

if __name__ == '__main__':
    main()
