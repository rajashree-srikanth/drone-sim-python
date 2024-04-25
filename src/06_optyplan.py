#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Trajectory planning with Opty (direct collocation)
"""

import sys, os, argparse
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation

import d2d.ploting as d2p


class WindField:
    def __init__(self, w=[0.,0.]):
        self.w = w
        
    def sample_sym(self, _t, _x, _y):
        return self.w
    
    def sample_num(self, _t, _x, _y):
        return self.w


class CostAirVel: # constant air velocity
    def __init__(self, vsp=10.):
        self.vsp = vsp
        
    def cost(self, free, _p):
        dvs = free[_p._slice_v]-self.vsp
        return _p.obj_scale * np.sum(np.square(dvs))/_p.num_nodes
    def cost_grad(self, free, _p):
        dvs = free[_p._slice_v]-self.vsp
        grad = np.zeros_like(free)
        grad[_p._slice_v] = _p.obj_scale/_p.num_nodes*2*dvs
        return grad

class CostBank:  # max bank or mean squared bank 

    def cost(self, free, _p):
        phis = free[_p._slice_phi]
        #return _p.obj_scale * np.sum(np.square(phis))/_p.num_nodes
        return _p.obj_scale * np.max(np.square(phis))
    
    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        phis = free[_p._slice_phi]
        #grad[_p._slice_phi] = _p.obj_scale/_p.num_nodes*2*phis
        _i = np.argmax(np.square(phis))
        grad[_p._slice_phi][_i] = _p.obj_scale*2*phis[_i]
        return grad

class CostObstacle: # penalyze circular obstacle
    def __init__(self, c=(30,0), r=15.):
        self.c, self.r = c, r

    def cost(self, free, _p):
        xs, ys = free[_p._slice_x], free[_p._slice_y]
        dxs, dys = xs-self.c[0], ys-self.c[1]
        ds = np.square(dxs)+np.square(dys)
        es = np.exp(self.r**2-ds)
        es = np.clip(es, 0., 1e3)
        return _p.obj_scale/_p.num_nodes * np.sum(es)

    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        xs, ys = free[_p._slice_x], free[_p._slice_y]
        dxs, dys = xs-self.c[0], ys-self.c[1]
        ds = np.square(dxs)+np.square(dys)
        es = np.exp(self.r**2-ds)
        es = np.clip(es, 0., 1e3)#breakpoint()
        grad[_p._slice_x] =  _p.obj_scale/_p.num_nodes*-2.*dxs*es
        grad[_p._slice_y] =  _p.obj_scale/_p.num_nodes*-2.*dys*es
        return grad

class CostObstacles: # penalyze set of circular obstacles
    def __init__(self, obss):
        self.obss = [CostObstacle(c=(_o[0], _o[1]), r=_o[2]) for _o in obss]
        
    def cost(self, free, _p):
        return np.sum([_c.cost(free, _p) for _c in self.obss])

    def cost_grad(self, free, _p):
        return np.sum([_c.cost_grad(free, _p) for _c in self.obss], axis=0)


class CostComposit: # a mix of the above
    def __init__(self, obss, vsp=10., kobs=1., kvel=1., kbank=1.):
        self.kobs, self.kvel, self.kbank = kobs, kvel, kbank
        self.cobs = CostObstacles(obss)
        self.cvel = CostAirVel(vsp)
        self.cbank = CostBank()

    def cost(self, free, _p):
        return self.kobs*self.cobs.cost(free, _p) + self.kvel*self.cvel.cost(free, _p) + self.kbank*self.cbank.cost(free, _p)

    def cost_grad(self, free, _p):
        return self.kobs*self.cobs.cost_grad(free, _p) + self.kvel*self.cvel.cost_grad(free, _p) + self.kbank*self.cbank.cost_grad(free, _p)
        


    
class SymAircraft:
    def __init__(self):
        # symbols
        self._st = sym.symbols('t')
        self._sx, self._sy, self._sv, self._sphi, self._spsi = sym.symbols('x, y, v, phi, psi', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sy(self._st), self._spsi(self._st))
        self._input_symbols = (self._sv, self._sphi)

    def get_eom(self, atm, g=9.81):
        wx, wy = atm.sample_sym(self._st, self._sx(self._st), self._sy(self._st))
        eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st)) + wx
        eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st)) + wy
        eq3 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3])

    
class Planner:
    def __init__(self, exp, initialize=True):
        self.exp = exp
        self.obj_scale = exp.obj_scale
        duration  = exp.t1 - exp.t0
        self.num_nodes = int(duration*exp.hz)+1
        self.interval_value = 1./exp.hz
        self.duration  = (self.num_nodes-1)*self.interval_value#duration
        print(f'solver: interval_value: {self.interval_value:.3f}s ({1./self.interval_value:.1f}hz)')
        self.wind = exp.wind
        self.aircraft = _g = SymAircraft()
 
        self._slice_x   = slice(0*self.num_nodes, 1*self.num_nodes, 1)
        self._slice_y   = slice(1*self.num_nodes, 2*self.num_nodes, 1)
        self._slice_psi = slice(2*self.num_nodes, 3*self.num_nodes, 1)
        self._slice_phi = slice(3*self.num_nodes, 4*self.num_nodes, 1)
        self._slice_v   = slice(4*self.num_nodes, 5*self.num_nodes, 1)

        # Known system parameters.
        self._par_map = collections.OrderedDict()
        #self._par_map[g] = 9.81
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
        #for _o in self.obstacles:
        #    cx, cy, rm = _o
        #    d2s = np.square(_g._sx(_g._st)-cx)+np.square(_g._sy(_g._st)-cy)
        #    #self._bounds[d2s] = (np.square(rm), np.square(rM))
        
        obj = exp.cost
        if initialize:
            self.prob =  opty.direct_collocation.Problem(lambda _free: obj.cost(_free, self),
                                                         lambda _free: obj.cost_grad(_free, self),
                                                         _g.get_eom(self.wind),
                                                         _g._state_symbols,
                                                         self.num_nodes,
                                                         self.interval_value,
                                                         known_parameter_map=self._par_map,
                                                         instance_constraints=self._instance_constraints,
                                                         bounds=self._bounds,
                                                         parallel=False)

           
    def configure(self, tol=1e-8, max_iter=3000):
        # https://coin-or.github.io/Ipopt/OPTIONS.html
        self.prob.addOption('tol', tol)            # default 1e-8
        self.prob.addOption('max_iter', max_iter)  # default 3000

    def get_initial_guess(self):
        initial_guess = np.random.randn(self.prob.num_free)
        if 1: # random positions
            cx, cy = self.exp.x_constraint, self.exp.y_constraint
            if cx is not None:
                initial_guess[self._slice_x] = np.random.default_rng().uniform(cx[0],cx[1],self.num_nodes)
            if cy is not None:
                initial_guess[self._slice_y] = np.random.default_rng().uniform(cy[0],cy[1],self.num_nodes)
        if 0:
            p0, p1 = self.exp.p0[:2], self.exp.p1[:2] 
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



        
def compute_or_load(_p, force_recompute=False, filename='/tmp/optyplan.npz', tol=1e-5, max_iter=1500, initial_guess=None):
    if force_recompute or not os.path.exists(filename):
        print(f'{filename} { os.path.exists(filename)}')
        _p.configure(tol, max_iter)
        initial_guess = _p.get_initial_guess()
        _p.run(initial_guess)
        if 1:
            _p.prob.plot_objective_value()
            _p.prob.plot_trajectories(_p.solution)
            #_p.prob.plot_constraint_violations(_p.solution)
        _p.save_solution(filename)
    else:
        _p.load_solution(filename)



def plot(_p, save):
    fig, axes = plt.subplots(5, 1)
    axes[0].plot(_p.sol_time, _p.sol_x)
    if _p.exp.x_constraint is not None:
        p0, dx, dy = (_p.exp.t0, _p.exp.x_constraint[0]), _p.exp.t1-_p.exp.t0, _p.exp.x_constraint[1]-_p.exp.x_constraint[0]
        axes[0].add_patch(plt.Rectangle(p0, dx, dy, color='g', alpha=0.1))
    d2p.decorate(axes[0], title='x', xlab='t in s', ylab='m', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[1].plot(_p.sol_time, _p.sol_y)
    if _p.exp.x_constraint is not None:
        p0, dx, dy = (_p.exp.t0, _p.exp.y_constraint[0]), _p.exp.t1-_p.exp.t0, _p.exp.y_constraint[1]-_p.exp.y_constraint[0]
        axes[1].add_patch(plt.Rectangle(p0, dx, dy, color='g', alpha=0.1))
    d2p.decorate(axes[1], title='y', xlab='t in s', ylab='m', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[2].plot(_p.sol_time, np.rad2deg(_p.sol_psi))
    d2p.decorate(axes[2], title='$\\psi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[3].plot(_p.sol_time, np.rad2deg(_p.sol_phi))
    if _p.exp.phi_constraint is not None:
        p0, dx, dy = (_p.exp.t0, np.rad2deg(_p.exp.phi_constraint[0])), _p.exp.t1-_p.exp.t0, np.rad2deg(_p.exp.phi_constraint[1]-_p.exp.phi_constraint[0])
        axes[3].add_patch(plt.Rectangle(p0, dx, dy, color='g', alpha=0.1))
    d2p.decorate(axes[3], title='$\\phi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[4].plot(_p.sol_time, _p.sol_v)
    if _p.exp.v_constraint is not None:
        p0, dx, dy = (_p.exp.t0, _p.exp.v_constraint[0]), _p.exp.t1-_p.exp.t0, _p.exp.v_constraint[1]-_p.exp.v_constraint[0]
        axes[4].add_patch(plt.Rectangle(p0, dx, dy, color='g', alpha=0.1))
    d2p.decorate(axes[4], title='$v$', xlab='t in s', ylab='m/s', legend=None, xlim=None, ylim=None, min_yspan=0.1)
    if save is not None:
        fn = f'{save}_chrono.png'
        print(f'saved {fn}'); plt.savefig(fn)
    
def plot2d(_p, save):
    _f = plt.figure()
    _a = plt.gca()
    _a.plot(_p.sol_x, _p.sol_y, solid_capstyle='butt')
    #p0, p1 = (_p.sol_x[0], _p.sol_y[0]), (_p.sol_x[-1], _p.sol_y[-1])
    dx, dy = _p.sol_x[-1]-_p.sol_x[-2], _p.sol_y[-1]-_p.sol_y[-2]
    #dx *= 10; dy *= 10
    _a.arrow(_p.sol_x[-1], _p.sol_y[-1], dx, dy, head_width=0.5,head_length=1)
    _a.add_patch(plt.Circle((_p.sol_x[0], _p.sol_y[0]), 0.5))
    for _o in _p.obstacles:
        cx, cy, rm = _o
        _a.add_patch(plt.Circle((cx,cy),rm, color='r', alpha=0.1))
    if _p.exp.y_constraint is not None:
        if _p.exp.x_constraint is not None:
            p0 = (_p.exp.x_constraint[0], _p.exp.y_constraint[0])
            dx = _p.exp.x_constraint[1]-_p.exp.x_constraint[0]
            dy = _p.exp.y_constraint[1]-_p.exp.y_constraint[0]
            _a.add_patch(plt.Rectangle(p0, dx, dy, color='g', alpha=0.1))
    _a.axis('equal')
    if save is not None:
        fn = f'{save}_2d.png'
        print(f'saved {fn}'); plt.savefig(fn)

class exp_0:
    tol, max_iter = 1e-5, 1500
    wind = WindField(w=[0.,0.])
    cost = CostAirVel(12.)
    obstacles = ( )
    obj_scale = 1.
    t0, p0 = 0.,  ( 0.,  0.,    0.,    0., 10.)    # initial position: t0, ( x0, y0, psi0, phi0, v0)
    t1, p1 = 10., ( 0., 30., np.pi,    0., 10.)    # final position
    x_constraint, y_constraint = None, None
    phi_constraint = (-np.deg2rad(30.), np.deg2rad(30.))
    v_constraint = (9., 12.)
    hz = 50.
    name = 'exp0'
    
class exp_1(exp_0):
    #wind = WindField(w=[2.,0.])
    obstacles = ((30, 0, 15), )
    cost = CostObstacle((obstacles[0][0], obstacles[0][1]), obstacles[0][2])
    #cost = CostBank()
    obj_scale = 1.e1
    #t1 = 4.
    name = 'exp1'

class exp_2(exp_0):
    obstacles = ((33, 0, 15), (23, 30, 12))
    #cost = CostObstacles(obstacles)
    cost = CostComposit(obstacles, 12., kobs=0.5, kvel=1e-6, kbank=10.)
    #obj_scale = 1.
    phi_constraint = (-np.deg2rad(60.), np.deg2rad(60.))
    t1 = 11.
    name = 'exp2'

class exp_3(exp_0):
    tol, max_iter = 1e-5, 3000
    obstacles = ((33, 0, 15), (23, 30, 12))
    cost = CostComposit(obstacles, vsp=14., kobs=0.1, kvel=2., kbank=2.)
    #cost = CostComposit(obstacles, vsp=12., kobs=1., kvel=1., kbank=0.5)
    #cost = CostComposit(obstacles, vsp=14., kobs=0.2, kvel=0.1, kbank=2.)
    obj_scale = 1.e-2
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    v_constraint = (9., 15.)
    #t1 = 16.
    x_constraint, y_constraint = (-5., 60.), (-1., 51.)
    name = 'exp3'

class exp_4(exp_0):
    t1, p1 = 10., ( 100., 0., 0,    0., 10.)    # final position
    obstacles = ((25, 0, 15), (55, 7.5, 12), (80, -10, 12))
    cost = CostComposit(obstacles, vsp=15., kobs=0.5, kvel=0.5, kbank=1.)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    obj_scale = 1.e-2
    x_constraint = (-5., 105.)
    y_constraint = (-15., 25.) #(-20., 20.)
    name = 'exp4'

class exp_5(exp_0):
    t0, p0 =  0., (   0., 40., 0,   0., 10.)    # start position 
    t1, p1 = 12., ( 100., 40., 0,   0., 10.)    # end position
    obstacles = []
    for i in range(5):
        for j in range(5):
            if (i+j)%2:
                obstacles.append((i*20., j*20., 10.))
    if 1:
        cost = CostComposit(obstacles, vsp=15., kobs=0.5, kvel=10., kbank=1.)
    else:
        cost = CostComposit(obstacles, vsp=15., kobs=0.5, kvel=10., kbank=1.)
        obj_scale = 1.e-4
        t1 = 15.
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    name = 'exp5'

class exp_6_0(exp_0):
    t0, p0 =  0., (   0.,  0., np.pi/2, 0., 10.)    # start position 
    t1, p1 = 12., ( 100., 40., 0,       0., 10.)    # end position
    name = 'exp6_0'
class exp_6_1(exp_0):
    t0, p0 =  0., (  10., 10., np.pi/2, 0., 10.)    # start position 
    t1, p1 = 12., ( 100., 42., 0,   0., 10.)    # end position
    name = 'exp6_1'
class exp_6_2(exp_0):
    t0, p0 =  0., (  20., 20., np.pi/2, 0., 10.)    # start position 
    t1, p1 = 12., ( 100., 44., 0,   0., 10.)    # end position
    name = 'exp6_2'

class exp_7_0(exp_0):
    obstacles = ((10, 20, 5), (40, 30, 5), (70, 20, 5))
    cost = CostComposit(obstacles, vsp=14., kobs=0.1, kvel=2., kbank=0.)
    obj_scale = 1.e-3
    x_constraint, y_constraint = (-5., 105.), (-1., 51.)
    t0, t1 = 0., 16.
    p0 = (   0.,  0., np.pi/2, 0., 10.)    # start position 
    p1 = ( 100., 40., 0,       0., 10.)    # end position
    name = 'exp7_0'
class exp_7_1(exp_7_0):
    p0 = (  10.,  0., np.pi/2, 0., 10.)    # start position 
    p1 = ( 100., 42., 0,       0., 10.)    # end position
    name = 'exp7_1'
class exp_7_2(exp_7_0):
    p0 = (  20.,  0., np.pi/2, 0., 10.)    # start position 
    p1 = ( 100., 44., 0,       0., 10.)    # end position
    name = 'exp7_2'    
class exp_7_3(exp_7_0):
    p0 = (  30.,  0., np.pi/2, 0., 10.)    # start position 
    p1 = ( 100., 46., 0,       0., 10.)    # end position
    name = 'exp7_3'    
class exp_7_4(exp_7_0):
    p0 = (  40.,  0., np.pi/2, 0., 10.)    # start position 
    p1 = ( 100., 48., 0,       0., 10.)    # end position
    name = 'exp7_4'    
class exp_8(exp_0):
    obstacles = ((10, 20, 5), (40, 30, 5), (70, 20, 5))
    cost = CostComposit(obstacles, vsp=14., kobs=0.1, kvel=2., kbank=0.)
    obj_scale = 1.e-3
    x_constraint, y_constraint = (-5., 105.), (-1., 51.)
    t0, t1 = 0., 30.
    p0 = (   0.,  0., np.pi/2, 0., 10.)    # start position 
    p1 = ( 100., 40., 0,       0., 10.)    # end position
    name = 'exp_8'
    
exps = [exp_0, exp_1, exp_2, exp_3, exp_4, exp_5,
        exp_6_0, exp_6_1, exp_6_2,
        exp_7_0, exp_7_1, exp_7_2, exp_7_3, exp_7_4,
        exp_8]
    
def parse_command_line():
    parser = argparse.ArgumentParser(description='Trajectory planning.')
    parser.add_argument('--traj', help='the name of the trajectory', default=None)
    parser.add_argument('--force', help='force recompute', action='store_true', default=False)
    parser.add_argument('--list', help='list all known trajectories', action='store_true', default=False)
    parser.add_argument('--save', help='save plot', action='store', default=None)
    args = parser.parse_args()
    return args
    
def main():
    args = parse_command_line()
    if args.list or not args.traj:
        print(exps)
        return
    try:
        traj_idx = int(args.traj)
        exp = exps[traj_idx]
    except ValueError:
        pass

    _p = Planner(exp, initialize = args.force)
    print('Planner initialized')
    compute_or_load(_p, args.force, f'./optyplan_{exp.name}.npz', tol=exp.tol, max_iter=exp.max_iter, initial_guess=None) # 1e-5
    print('Planner ran')
    plot(_p, args.save)
    plot2d(_p, args.save)
    plt.show()

    
if __name__ == '__main__':
    main()
