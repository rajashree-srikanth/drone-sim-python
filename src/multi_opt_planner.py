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
import pandas as pd

import opty.direct_collocation

import d2d.ploting as d2p
import d2d.opty_utils as d2ou
import d2d.multiopty_utils as d2mou


    
# FIXME: all airplane don't start at the same time, must be smarter with time
import itertools
class Planner:
    def __init__(self, scen, initialize=True):
        self.scen = scen
        self.obj_scale = scen.obj_scale

        self.wind = scen.wind #d2ou.WindField()
        self.acs = d2mou.AircraftSet(n=len(scen.p0s))
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
            if scen.x_constraint is not None: self._bounds[_ac._sx(_ac._st)] = scen.x_constraint
            if scen.y_constraint is not None: self._bounds[_ac._sy(_ac._st)] = scen.y_constraint
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

        
        self.prob.add_option('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000
        self.solution, info = self.prob.solve(initial_guess)   
        #self.interpret_solution()
        if 0:
            # Make some plots
            self.prob.plot_trajectories(self.solution)
            #self.prob.plot_constraint_violations(self.solution) # fixme: fails
            self.prob.plot_objective_value()
            
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

    for _o in _p.scen.obstacles:
        cx, cy, rm = _o
        _a.add_patch(plt.Circle((cx,cy),rm, color='r', alpha=0.1))
        
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
        if _p.acs.nb_aicraft < 2:
            axes[2].plot(_p.sol_time, np.rad2deg(_p.sol_psi[i]), label=f'{i}')
            d2p.decorate(axes[2], title='$\\psi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
        else:
            dx, dy =  _p.sol_x[1] - _p.sol_x[0], _p.sol_y[1] - _p.sol_y[0] 
            dist = np.hypot(dx,dy)
            axes[2].plot(_p.sol_time, dist)
            d2p.decorate(axes[2], title='$d$', xlab='t in s', ylab='m', ylim=[0, 20])
        axes[3].plot(_p.sol_time, np.rad2deg(_p.sol_phi[i]), label=f'{i}')
        d2p.decorate(axes[3], title='$\\phi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=1.)
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


class exp_0:  # single aircraft
    name = 'exp_0'
    desc = 'single aircraft'
    t0, t1, hz = 0., 10., 50.
    dx, dy = 0.,50.
    #dx, dy = 50, 50

    wind = d2ou.WindField()

    #initial_guess = 'rnd'
    initial_guess = 'tri'
    
    tol, max_iter = 1e-5, 5000
    #tol, max_iter = 1e-5, 1500
    #tol, max_iter = 1e-5, 0
    #cost, obj_scale = SetCostBank(), 1.
    #cost, obj_scale = SetCostAirvel(vsp=12.), 1.
    vref = 12.
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=5., kphi=1.), 1.e-1
    #cost; obj_scale = SetCostNull(), 1.
    x_constraint, y_constraint = None, None
    x_constraint, y_constraint = (-5, 50), (-5, 50)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    v_constraint = (9., 15.)
    obstacles = []

    ncases = 1
    def set_case(idx): pass
    def label(idx): return ''
    
class exp_5(exp_0): # testing collisions
    name = 'exp_5'
    desc = '2 aicraft face to face'
    t1 = 4
    vref = 12.
    dpsi = 0.
    x_constraint, y_constraint = None, None
    p0s = (( 0.,  0.,   0.,  0., 12.), ( 50.,  0.,  np.pi-dpsi, 0., 12.))
    p1s = (( 50., 0.,   0.,  0., 12.), (  0.,  0.,  np.pi+dpsi, 0., 12.))
    obstacles = []
    #initial_guess = 'rnd'
    initial_guess = 'tri'
    ncases = 2
    def set_case(idx):
        if idx==0:
            #exp_5.cost, exp_5.obj_scale = d2mou.CostInput(vsp=exp_5.vref, kv=70., kphi=1.), 1.e0
            exp_5.cost, exp_5.obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=float('NaN'), vsp=exp_5.vref, obss=[], obs_kind=0, rcol=3.), 1.e0
        else:
            #exp_5.cost, exp_5.obj_scale = d2mou.CostCollision(r=6, k=2), 1.e0
            exp_5.cost, exp_5.obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=10., vsp=exp_5.vref, obss=[], obs_kind=0, rcol=10.), 1.e0
            
    def label(idx):  return f'obj {["Ref", "AntiCol"][idx]}'

class trap_4(exp_5):
   name = 'trap_4'
   desc = 'trapezoidal formation with 4 aircraft'
   hz = 10
#    t1 = 5.5# 14.2
   vref = 12
   dpsi = 0
#    p0s = ((0, 40, np.deg2rad(0), 0, 12), (40, 0, np.deg2rad(0), 0, 12), (0, -40, np.deg2rad(0), 0, 12), (-40, 0, np.deg2rad(0), 0, 12))
#    p0s = ((0, 40, np.deg2rad(0), 0, 12), (25, 20, np.deg2rad(0), 0, 12), (25, -20, np.deg2rad(0), 0, 12), (0, -40, np.deg2rad(0), 0, 12))
#    p0s = ((0, 40, np.deg2rad(-2.00691223e+01), 0, 12), (25, 20, np.deg2rad(-2.00691223e+01), 0, 12), (25, -20, np.deg2rad(-2.00691223e+01), 0, 12), (0, -40, np.deg2rad(-2.00691223e+01), 0, 12))
#    p0s = ((0, 40, np.deg2rad(0), 0, 12), (40, 0, np.deg2rad(-90), 0, 12), (0, -40, np.deg2rad(180), 0, 12), (-40, 0, np.deg2rad(90), 0, 12))
#    p1s = ((75, 40, 0, 0, 12), (100, 20, 0, 0, 12), (100,-20, 0, 0, 12), (75, -40, 0, 0, 12))
#    p1s = ((75, 100, 0, 0, 12), (100, 60, 0, 0, 12), (75, -100, 0, 0, 12), (100, -60, 0, 0, 12))
   x_constraint = (-150, 150)
   y_constraint = (-150, 150)
   initial_guess = 'tri'
   ncases = 1
   
   # wind
#    wind = d2ou.WindField([5,2])
   
   cost, obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=10., vsp=vref, obss=[], obs_kind=0, rcol=10), 1.e0
   
class inf_traj_4ac(exp_5):
    name = 'inf trajectory'
    desc = "attempting some fancy inf-like traj"
    hz = 10
    # t = np.arange(6.8,7.4,0.2) # for 1st set of pts - 7 s is good!
    # t = np.arange(10.5, 11.5, 0.2)  # for 2nd set of pts - 8 s is good!
    t = [10]
    vref = 12
    dpsi = 0
    p0s = ((75, 40, np.deg2rad(0), np.deg2rad(20), 12),(100, 40, np.deg2rad(0), np.deg2rad(20), 12), (100, -40, np.deg2rad(0), np.deg2rad(20), 12),(75, -40, np.deg2rad(0), np.deg2rad(20), 12))
    p1s = ((75,-40, np.deg2rad(0), np.deg2rad(-39), 12),(100, -40, np.deg2rad(0), np.deg2rad(-39), 12), (100, 40, np.deg2rad(0), np.deg2rad(-39), 12), (75,40, np.deg2rad(0), np.deg2rad(-39), 12))
    # p0s = ((75, 40, np.deg2rad(0), np.deg2rad(20), 12),(75, -40, np.deg2rad(0), np.deg2rad(20), 12))
    # p1s = ((75,-40, np.deg2rad(0), np.deg2rad(-39), 12),(75,40, np.deg2rad(0), np.deg2rad(-39), 12))
    x_constraint = None
    y_constraint = None
    initial_guess = 'tri'
    ncases = len(t)
    # breakpoint()
    def set_case(idx):
        exp_5.t1 = inf_traj_4ac.t[idx]
        cost, obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=10., vsp=inf_traj_4ac.vref, obss=[], obs_kind=0, rcol=10), 1.e0
   
    def label(idx):  return f't_flight_{inf_traj_4ac.t[idx]}'
    
class inf_traj_4ac(exp_5):
    name = 'inf trajectory'
    desc = "attempting some fancy inf-like traj"
    hz = 10
    # t = np.arange(6.8,7.4,0.2) # for 1st set of pts - 7 s is good!
    # t = np.arange(10.5, 11.5, 0.2)  # for 2nd set of pts - 8 s is good!
    t = [10]
    vref = 12
    dpsi = 0
    p0s = ((75, 40, np.deg2rad(0), np.deg2rad(20), 12),(100, 40, np.deg2rad(0), np.deg2rad(20), 12), (100, -40, np.deg2rad(0), np.deg2rad(20), 12),(75, -40, np.deg2rad(0), np.deg2rad(20), 12))
    p1s = ((75,-40, np.deg2rad(0), np.deg2rad(-39), 12),(100, -40, np.deg2rad(0), np.deg2rad(-39), 12), (100, 40, np.deg2rad(0), np.deg2rad(-39), 12), (75,40, np.deg2rad(0), np.deg2rad(-39), 12))
    # p0s = ((75, 40, np.deg2rad(0), np.deg2rad(20), 12),(75, -40, np.deg2rad(0), np.deg2rad(20), 12))
    # p1s = ((75,-40, np.deg2rad(0), np.deg2rad(-39), 12),(75,40, np.deg2rad(0), np.deg2rad(-39), 12))
    x_constraint = None
    y_constraint = None
    initial_guess = 'tri'
    ncases = len(t)
    # breakpoint()
    def set_case(idx):
        exp_5.t1 = inf_traj_4ac.t[idx]
        cost, obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=10., vsp=inf_traj_4ac.vref, obss=[], obs_kind=0, rcol=10), 1.e0
   
    def label(idx):  return f't_flight_{inf_traj_4ac.t[idx]}'
    
class anticoll(exp_0):
    name = 'testing collision'
    desc = "2 ac collision"
    wind = d2ou.WindField()
    tol, max_iter = 1e-5, 5000
    
    hz = 10
    # t = np.arange(6.8,7.4,0.2) # for 1st set of pts - 7 s is good!
    # t = np.arange(10.5, 11.5, 0.2)  # for 2nd set of pts - 8 s is good!
    # t = [4]
    t0 = 0
    t1 = 4
    
    vref = 12
    dpsi = 0
    # p0s = ((0, 40, np.deg2rad(0), np.deg2rad(0), 12),(0,-40, np.deg2rad(0), np.deg2rad(0), 12))
    # p1s = ((0,-40, np.deg2rad(0), np.deg2rad(0), 12),(0, 40, np.deg2rad(0), np.deg2rad(0), 12))
    p0s = (( 0.,  0.,   0.,  0., 12.), ( 50.,  0.,  np.pi-dpsi, 0., 12.))
    p1s = (( 50., 0.,   0.,  0., 12.), (  0.,  0.,  np.pi+dpsi, 0., 12.))
    # p0s = ((75, 40, np.deg2rad(0), np.deg2rad(20), 12),(75, -40, np.deg2rad(0), np.deg2rad(20), 12))
    # p1s = ((75,-40, np.deg2rad(0), np.deg2rad(-39), 12),(75,40, np.deg2rad(0), np.deg2rad(-39), 12))
    x_constraint = None
    y_constraint = None
    initial_guess = 'tri'
    ncases = 2
    # breakpoint()
    def set_case(idx):
        if idx==0:
            # exp_5.t1 = anticoll.t[0]
            anticoll.cost, anticoll.obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=float('NaN'), vsp=anticoll.vref, obss=[], obs_kind=0, rcol=3), 1.e0
        else:
            # exp_5.t1 = anticoll.t[0]
            anticoll.cost, anticoll.obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=10., vsp=anticoll.vref, obss=[], obs_kind=0, rcol=3), 1.e0
   
    def label(idx):  return f't_flight_{anticoll.t1}'
    