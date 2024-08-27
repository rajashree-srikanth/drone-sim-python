#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Trajectory planning with Opty (direct collocation)
 Single vehicle planner
"""
import numpy as np, sympy as sym
import collections
import opty.direct_collocation

import d2d.opty_utils as d2ou

class Planner:
    def __init__(self, exp, initialize=True):
        self.exp = exp                 # scenario
        self.obj_scale = exp.obj_scale # cost function scaling
        self.num_nodes, self.time_step, self.duration = d2ou.planner_timing(exp.t0, exp.t1, exp.hz)

        self.wind = exp.wind
        self.aircraft = _g = d2ou.Aircraft5d()
 
        self._slice_x      = slice(0*self.num_nodes, 1*self.num_nodes, 1)
        self._slice_y      = slice(1*self.num_nodes, 2*self.num_nodes, 1)
        self._slice_psi    = slice(2*self.num_nodes, 3*self.num_nodes, 1)
        self._slice_v      = slice(3*self.num_nodes, 4*self.num_nodes, 1)
        self._slice_phi    = slice(4*self.num_nodes, 5*self.num_nodes, 1)
        self._slice_v_sp   = slice(5*self.num_nodes, 6*self.num_nodes, 1)
        self._slice_phi_sp = slice(6*self.num_nodes, 7*self.num_nodes, 1)
        
        # Known system parameters.
        self._par_map = collections.OrderedDict()
        #self._par_map[g] = 9.81
        #self._par_map[self.aircraft._sv] = 12.
        # Symbolic instance constraints, i.e. initial and end conditions.
        t0, (x0, y0, psi0, phi0, v0) = exp.t0, exp.p0
        self._instance_constraints = (_g._sx(t0)-x0, _g._sy(t0)-y0, _g._spsi(t0)-psi0, _g._sv(t0)-v0, _g._sphi(t0)-phi0)
        t1, (x1, y1, psi1, phi1, v1) = exp.t1, exp.p1
        self._instance_constraints += (_g._sx(t1)-x1, _g._sy(t1)-y1, _g._spsi(t1)-psi1, _g._sv(t1)-v1, _g._sphi(t1)-phi1)
                                      
        # Bounds
        self._bounds = {}
        self._bounds[_g._sphi_sp(_g._st)] = exp.phi_constraint
        self._bounds[_g._sv_sp(_g._st)] = exp.v_constraint
        if exp.x_constraint is not None: self._bounds[_g._sx(_g._st)] = exp.x_constraint
        if exp.y_constraint is not None: self._bounds[_g._sy(_g._st)] = exp.y_constraint

        self.obstacles = exp.obstacles
        obj = exp.cost
        if initialize:
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
            if D > d: # We have time to spare, let make an isocele triangle
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
        initial_guess = np.random.randn(self.prob.num_free) if initial_guess is None else initial_guess
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


