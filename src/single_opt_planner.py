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
import d2d.optyplan_scenarios as d2oscen

class Planner:
    def __init__(self, exp, initialize=True):
        self.exp = exp                 # scenario
        self.obj_scale = exp.obj_scale # cost function scaling
        self.num_nodes, self.time_step, self.duration = d2ou.planner_timing(exp.t0, exp.t1, exp.hz)

        self.wind = exp.wind
        self.aircraft = _g = d2ou.Aircraft()
 
        # slice sequence - to extract the different states
        self._slice_x   = slice(0*self.num_nodes, 1*self.num_nodes, 1)
        self._slice_y   = slice(1*self.num_nodes, 2*self.num_nodes, 1)
        self._slice_psi = slice(2*self.num_nodes, 3*self.num_nodes, 1)
        self._slice_phi = slice(3*self.num_nodes, 4*self.num_nodes, 1)
        self._slice_v   = slice(4*self.num_nodes, 5*self.num_nodes, 1)

        # Known system parameters.
        self._par_map = collections.OrderedDict()
        #self._par_map[g] = 9.81
        #self._par_map[self.aircraft._sv] = 12.
        # Symbolic instance constraints, i.e. initial and end conditions.
        t0, (x0, y0, psi0, phi0, v0) = exp.t0, exp.p0
        self._instance_constraints = (_g._sx(t0)-x0, _g._sy(t0)-y0, _g._spsi(t0)-psi0)
        t1, (x1, y1, psi1, phi1, v1) = exp.t1, exp.p1
        self._instance_constraints += (_g._sx(t1)-x1, _g._sy(t1)-y1, _g._spsi(t1)-psi1)
        # nope self._instance_constraints += (_g._sx(_g._st) < 30, )
                                      
        # Bounds
        self._bounds = {}
        self._bounds[_g._sphi(_g._st)] = exp.phi_constraint
        self._bounds[_g._sv(_g._st)] = exp.v_constraint
        if exp.x_constraint is not None: self._bounds[_g._sx(_g._st)] = exp.x_constraint
        if exp.y_constraint is not None: self._bounds[_g._sy(_g._st)] = exp.y_constraint

        self.obstacles = exp.obstacles
        obj = exp.cost
        if initialize: # if initialize is true
            self.prob =  opty.direct_collocation.Problem(lambda _free: obj.cost(_free, self),
                                                         lambda _free: obj.cost_grad(_free, self),
                                                         _g.get_eom(self.wind),
                                                         _g._state_symbols,
                                                         self.num_nodes,
                                                         self.time_step,
                                                         known_parameter_map=self._par_map,
                                                         instance_constraints=self._instance_constraints,
                                                         bounds=self._bounds,
                                                         parallel=False)

           
    def configure(self, tol=1e-8, max_iter=3000):
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000

    def get_initial_guess(self, kind='tri'):
        #initial_guess = np.random.randn(self.prob.num_free)
        initial_guess = np.zeros(self.prob.num_free)
        if kind=='rnd': # random positions
            rng = np.random.default_rng(seed)
            # constraints on x and y, and psi as well
            cx, cy = self.exp.x_constraint, self.exp.y_constraint
            if cx is None: cx = [-100, 100]
            initial_guess[self._slice_x] = rng.uniform(cx[0],cx[1], self.num_nodes)
            if cy is None: cy = [-100, 100]
            initial_guess[self._slice_y] = rng.uniform(cy[0],cy[1],self.num_nodes)
            initial_guess[self._slice_psi] = rng.uniform(-np.pi, np.pi,self.num_nodes)    
        elif kind == 'tri': # equilateral triangle for arriving on time
            p0, p1 = np.array(self.exp.p0[:2]), np.array(self.exp.p1[:2]) # start and end
            p0p1 = p1-p0
            d = np.linalg.norm(p0p1)        # distance between start and end
            u = p0p1/d; v = np.array([-u[1], u[0]]) # unit and normal vectors
            D = self.exp.vref*self.duration # distance to be traveled during scenario
            p3 = p0 + p0p1/2
            if D > d: # We have time to spare, let make an isoceles triangle
                p3 -= np.sqrt(D**2-d**2)/2*v
            n1 = int(self.num_nodes/2); n2 = self.num_nodes - n1
            _p0p3 = np.linspace(p0, p3, n1)
            _p3p1 = np.linspace(p3, p1, n2)
            _p0p1 = np.vstack((_p0p3, _p3p1))
            initial_guess[self._slice_x] = _p0p1[:,0]
            initial_guess[self._slice_y] = _p0p1[:,1]
            p0p3 = p3-p0; psi0 = np.arctan2(p0p3[1], p0p3[0])
            p3p1 = p1-p3; psi1 = np.arctan2(p3p1[1], p3p1[0])
            initial_guess[self._slice_psi] = np.hstack((psi0*np.ones(n1), psi1*np.ones(n2)))
            initial_guess[self._slice_v] = self.exp.vref
            initial_guess[self._slice_phi] = np.zeros(self.num_nodes)
        else: # straight line
            p0, p1 = np.array(self.exp.p0[:2]), np.array(self.exp.p1[:2])
            initial_guess[self._slice_x] = np.linspace(p0[0], p1[0], self.num_nodes)
            initial_guess[self._slice_y] = np.linspace(p0[1], p1[1], self.num_nodes)
        return initial_guess
        
    def run(self, initial_guess=None):
        # Use a random positive initial guess.
        if initial_guess is None:
            initial_guess = np.random.randn(self.prob.num_free)
        else:
            initial_guess
        # Find the optimal solution.
        self.solution, info = self.prob.solve(initial_guess)   
        self.interpret_solution()

    def interpret_solution(self):
        self.sol_time = np.linspace(0.0, self.duration, num=self.num_nodes)
        self.sol_x   = self.solution[self._slice_x]
        self.sol_y   = self.solution[self._slice_y]
        self.sol_psi = self.solution[self._slice_psi]
        self.sol_phi = self.solution[self._slice_phi]
        self.sol_v   = self.solution[self._slice_v]
        
    def save_solution(self, filename):
        wind = np.array([self.wind.sample_num(_t, _x, _y) for _t, _x, _y in zip(self.sol_time, self.sol_x, self.sol_y)])
        np.savez(filename, sol_time=self.sol_time, sol_x=self.sol_x, sol_y=self.sol_y,
                 sol_psi=self.sol_psi, sol_phi=self.sol_phi, sol_v=self.sol_v, wind=wind)
        print('saved {}'.format(filename))

    def load_solution(self, filename):
        _data =  np.load(filename)
        labels = ['sol_time', 'sol_x', 'sol_y', 'sol_psi', 'sol_phi', 'sol_v']
        self.sol_time, self.sol_x, self.sol_y, self.sol_psi, self.sol_phi, self.sol_v = [_data[k] for k in labels]
        print(f'loaded {filename}')



        


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
        if scen.ncases > 1:
            cache_filename = f'./cache/optyplan_{scen.name}_{_case}.npz'
        else:
            cache_filename = f'./cache/optyplan_{scen.name}.npz'
        directory = os.path.dirname(cache_filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        _p = Planner(scen, initialize=args.force or not os.path.exists(cache_filename))
        print('Planner initialized')
        compute_or_load(_p, args.force, cache_filename, tol=scen.tol, max_iter=scen.max_iter, initial_guess=None)
        print('Planner ran')
        f1, a1 = d2ou.plot2d(_p, args.save, f1, a1, scen.label(_case))
        f2, a2 = d2ou.plot_chrono(_p, args.save, f2, a2)
    plt.show()

class exp_0:
    ncases = 1
    tol, max_iter = 1e-5, 1500
    vref = 12.
    cost = d2ou.CostAirVel(vref)
    #cost = d2ou.CostBank()
    obj_scale = 1.
    wind = d2ou.WindField(w=[0.,0.])
    obstacles = ( )
    t0, p0 = 0.,  ( 0.,  0.,    0.,    0., 10.)    # initial position: t0, ( x0, y0, psi0, phi0, v0)
    t1, p1 = 10., ( 0., 30., np.pi,    0., 10.)    # final position
    x_constraint, y_constraint = None, None
    #x_constraint, y_constraint = (-5., 100.), (-50., 50.) 
    phi_constraint = (-np.deg2rad(30.), np.deg2rad(30.))
    v_constraint = (9., 14.)
    hz = 10.
    name = 'exp0'
    desc = 'Turn around - 12m/s objective'
    def set_case(idx): pass
    def label(idx): return ''

class exp_1():
    name = "exp 1 - joining 2 points"
    desc = "single ac traj computation for test case 2 of full sim"
    ncases = 1
    tol, max_iter = 1e-5, 1500
    vref = 12
    cost = d2ou.CostAirVel(vref)
    obj_scale = 1
    wind = d2ou.WindField(w=[0,0])
    obstacles = ()
    # t0, p0 = 0, (-49.98,-58.14,2.22,-0.35,15.  )
    t0 = 0
    t1, p1 = 12, (75, 40, 0, 0, 12)
    x_constraint = (-150, 150)
    y_constraint = (-150, 150)
    v_constraint = (9., 15.)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    initial_guess = 'tri'
    
    hz = 10
    def set_case(idx): pass
    def label(idx): return ''
    
# if __name__ == '__main__':
#     main()
