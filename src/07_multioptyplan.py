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
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( dx, dy,  np.pi/2,  0., 10.), )

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
   
class exp_0_1(exp_0):  # varying weights
    name = 'exp_0_1'
    desc = 'single aircraft, varying weights'
    t0, p0s =  0., (( 0.,  0., 0.,  0., 10.), )
    t1, p1s = 10., (( 100, 0,  0.,  0., 10.), )
    x_constraint, y_constraint = None, None
    #initial_guess = 'rnd'
    #Ks = [[1., 0.5], [1., 1.],[1., 10.],[1., 20.], [1., 30.], [1., 40.], [1., 50.]]
    Ks = [[1., 1.],[1., 20.], [1., 40.], [1., 60.]]
    ncases = len(Ks)
    def set_case(idx):
        exp_0_1.K = exp_0_1.Ks[idx]
        exp_0_1.cost = d2mou.CostInput(vsp=13., kv=exp_0_1.K[0], kphi=exp_0_1.K[1])
    def label(idx):  return f'kvel, kbank {exp_0_1.K}'

    
class exp_1(exp_0): # 2 aircraft face to face
    name = 'exp_1'
    desc = '2 aicraft face to face'
    t1 = 4.5
    vref = 12.
    dpsi = 0.01
    p0s = (( 0.,  0.,   0.,  0., 12.), ( 50.,  0.,  np.pi-dpsi, 0., 12.))
    p1s = (( 50., 0.,   0.,  0., 12.), (  0.,  0.,  np.pi+dpsi, 0., 12.))
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=5., kphi=1.), 1.e-1
    #cost, obj_scale = SetCostCollision(), 1.
    x_constraint, y_constraint = None, None
    obstacles = []
    #obstacles = ((25, -10, 5), )
    initial_guess = 'rnd'

class exp_1_0(exp_1): # 2 aircraft meeting
    name = 'exp_1_0'
    desc = '2 aicraft meeting'
    t1 = 4.5
    vref = 12.
    p0s = ((  0., -20., np.pi/2,  0., 12.), ( 7.5, -20.,  np.pi/2,  0., 12.))
    p1s = (( 40.,   5.,      0.,  0., 12.), ( 40.,   10.,  0,        0., 12.))
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=1., kphi=1.), 1.e-1
    x_constraint, y_constraint = None, None

class exp_1_1(exp_1): # face to face, wind
    name = 'exp_1_1'
    desc = '2 aicraft face to face, wind'
    #initial_guess = 'rnd'
    initial_guess = 'tri'


    
class exp_2(exp_0): # four aircaft in cross
    name = 'exp_2'
    desc = '4 aicraft'
    t1 = 5.5
    vref = 12.
    overtime=1.5#1.75
    d = t1*vref/2 / overtime
    p0s = ((-d, 0., 0., 0., vref), ( d,  0.,  np.pi, 0., vref), ( 0.,  d, -np.pi/2,  0., vref), ( 0., -d,  np.pi/2, 0., vref))
    p1s = (( d, 0., 0., 0., vref), (-d,  0.,  np.pi, 0., vref), ( 0., -d, -np.pi/2,  0., vref), ( 0.,  d,  np.pi/2, 0., vref))
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=1., kphi=1.), 1.



class exp_3(exp_0): # testing obstacles
    name = 'exp_3'
    desc = 'single obstacle'
    t1 = 6.5 #7.5
    vref = 12.
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( 50., 0.,   0.,  0., 10.), )

    obstacles = ((25, -20, 10), )
    #cost, obj_scale = SetCostInput(vsp=vref, kv=5., kphi=1.), 1.e-1
    cx, cy, r = obstacles[0]
    cost, obj_scale = d2mou.CostObstacle(c=(cx,cy), r=r, kind=0), 1.
    x_constraint, y_constraint = None, None
    #x_constraint, y_constraint = (-100, 100), (-100, 100)
    #v_constraint = (11.99, 12.01)
    v_constraint = (8., 18.)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))


class exp_3_1(exp_3): # testing obstacles
    name = 'exp_3_1'
    desc = 'single obstacle, size/location'
    obstacles = ((25, -20, 10), (25, -10, 10))

    ncases = len(obstacles)
    def set_case(idx):
        cx, cy, r = exp_3_1.obstacles[idx] 
        exp_3_1.cost = d2mou.CostObstacle(c=(cx,cy), r=r, kind=0)
    def label(idx):  return f'obstacle {exp_3_1.obstacles[idx]}'

class exp_4(exp_0): # testing obstacles
    name = 'exp_4'
    desc = 'set of obstacle'
    t1 = 10.5
    vref = 12.
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( 100., 0.,   0.,  0., 10.), )

    obstacles = ((30, -10, 20), (70, 15, 20), )
    #cost, obj_scale = d2mou.CostInput(vsp=vref, kv=1., kphi=1.), 1.
    #cost, obj_scale = d2mou.CostObstacles(obstacles, kind=1), 1.
    cost, obj_scale = d2mou.CostComposit(kvel=1., kbank=1., kobs=1., kcol=float('NaN'), vsp=vref, obss=obstacles, obs_kind=1, rcol=3.), 1.
    x_constraint, y_constraint = None, None
    #x_constraint, y_constraint = (-5., 105.), (-20., 80.)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    #v_constraint = (11.99, 12.01)
    v_constraint = (9., 18.)
    initial_guess = 'rnd'
    #initial_guess = 'tri'
    
class exp_4_1(exp_4): # testing obstacles
    name = 'exp_4_1'
    desc = 'set of obstacles, size'
    v_constraint = (9., 15.)
    _obstacles = (((30, -10, 15), (30, 25, 15)),
                  ((50, -10, 15), (50, 25, 15)),
                  ((70, -10, 15), (70, 25, 15)))
    #obstacles = _obstacles[0]
    #cost, obj_scale = d2mou.CostObstacles(obstacles, kind=1), 1.
    #cost, obj_scale = d2mou.CostComposit(obstacles, vsp=10., kobs=1., kvel=1., kbank=1., obs_kind=1), 1.
    obj_scale = 1e-2
    ncases = len(_obstacles)
    def set_case(idx):
        exp_4_1.obstacles = exp_4_1._obstacles[idx] 
        #exp_4_1.cost = d2mou.CostObstacles(exp_4_1.obstacles, kind=1)
        exp_4_1.cost, exp_4_1.obj_scale = d2mou.CostComposit(kvel=1., kbank=1., kobs=1., kcol=float('NaN'), vsp=14., obss=exp_4_1.obstacles, obs_kind=1, rcol=3.), 1.
        #d2mou.CostComposit(exp_4_1.obstacles, vsp=14., kobs=1., kvel=1., kbank=.1, obs_kind=0)
    def label(idx):  return f'obstacles {exp_4_1._obstacles[idx]}'


class exp_4_2(exp_4): # testing obstacles
    name = 'exp_4_2'
    desc = 'set of obstacles, duration'
    _durations = [9, 10, 11, 12]
    ncases = len(_durations)
    def set_case(idx):
        exp_4_2.t1 = exp_4_2._durations [idx]
    def label(idx):  return f'duration {exp_4_2._durations[idx]} s'
    initial_guess = 'rnd'

    
class exp_5(exp_0): # testing collisions
    name = 'exp_5'
    desc = '2 aicraft face to face'
    t1 = 4.2
    vref = 12.
    dpsi = 0.
    p0s = (( 0.,  0.,   0.,  0., 12.), ( 50.,  0.,  np.pi-dpsi, 0., 12.))
    p1s = (( 50., 0.,   0.,  0., 12.), (  0.,  0.,  np.pi+dpsi, 0., 12.))
    x_constraint, y_constraint = None, None
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
    
class exp_5_1(exp_5): # testing collisions
    name = 'exp_5'
    desc = '2 aicraft next to one another'
    t1 = 8.
    vref = 12.
    if 0:
        p0s = (( 0.,  0.,   0.,  0., 12.), ( 0.,  0.,  0, 0., 12.))
        p1s = (( 50., 50.,   np.pi/2,  0., 12.), (  50.,  50.,  np.pi/2, 0., 12.))
    else:
        p0s = (( 0.,  0.,   0.,  0., 12.), ( 0.,  5.,  0, 0., 12.))
        p1s = (( 50., 50.,   np.pi/2,  0., 12.), (  55.,  50.,  np.pi/2, 0., 12.))
    initial_guess = 'tri'
    #initial_guess = 'rnd'



    

    
scens = [exp_0, exp_0_1, exp_1, exp_1_0, exp_1_1, exp_2,
         exp_3, exp_3_1,
         exp_4, exp_4_1, exp_4_2,
         exp_5, exp_5_1]
def desc_all_scens():
    return '\n'.join([f'{i}: {s.name} {s.desc}' for i, s in enumerate(scens)])
def get_scen(idx): return scens[idx]

def info_scen(idx):
    res = ''
    res += f'{scens[idx].name} {scens[idx].desc}\n'
    res += f't0 {scens[idx].t0} t1 {scens[idx].t1}\n'
    res += f'p0s {scens[idx].p0s}\np1s {scens[idx].p1s}\n'
    res += f'x_constraint {scens[idx].x_constraint}\ny_constraint {scens[idx].y_constraint}\n'
    res += f'v_constraint {scens[idx].v_constraint}\nphi_constraint {np.rad2deg(scens[idx].phi_constraint)}\n'
    res += f'wind {scens[idx].wind}\n'
    res += f'initial guess {scens[idx].initial_guess}\n'
    return res

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
    print(info_scen(int(args.scen)))
    
    f1, a1, f2, a2 = None, None, None, None
    for _case in range(scen.ncases):
        scen.set_case(_case)
        _p = Planner(scen, initialize=True)
        initial_guess = None
        initial_guess = _p.get_initial_guess(scen.initial_guess)
        _p.run(initial_guess=initial_guess, tol=scen.tol, max_iter=scen.max_iter)
        _p.interpret_solution()
        f1, a1 = plot2d(_p, f1, a1, scen.label(_case))
        f2, a2 = plot_chrono(_p, f2, a2)
    for _a in a2: _a.autoscale()
    plt.show()
    

if __name__ == '__main__':
    main()
