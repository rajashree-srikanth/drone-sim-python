#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import sys, os, numpy as np, sympy as sym
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



# mean bank
def obj_meanbank(free, _p):
    phis = free[_p._slice_phi]
    return _p.obj_scale/_p.num_nodes * np.sum(np.square(phis))
def obj_grad_meanbank(free, _p):
    grad = np.zeros_like(free)
    phis = free[_p._slice_phi]
    grad[_p._slice_phi] = _p.obj_scale/_p.num_nodes*2*phis
    return grad

# min bank
def obj_minbank(free, _p):
    phis = free[_p._slice_phi]
    return _p.obj_scale * np.max(np.square(phis))
def obj_grad_minbank(free, _p):
    grad = np.zeros_like(free)
    phis = free[_p._slice_phi]
    _i = np.argmax(np.square(phis))
    grad[_i] = _p.obj_scale*2*phis[_i]
    return grad


class CostAirVel:
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

class CostObstacle:
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

class CostObstacles:
    def __init__(self, obss):
        self.obss = [CostObstacle(c=(_o[0], _o[1]), r=_o[2]) for _o in obss]
        
    def cost(self, free, _p):
        return np.sum([_c.cost(free, _p) for _c in self.obss])

    def cost_grad(self, free, _p):
        return np.sum([_c.cost_grad(free, _p) for _c in self.obss], axis=0)


class CostVelObstacles:
    def __init__(self, obss, vsp=10., kobs=1., kvel=1.):
        self.kobs, self.kvel = kobs, kvel
        self.cobs = CostObstacles(obss)
        self.cvel = CostAirVel(vsp)

    def cost(self, free, _p):
        return self.kobs*self.cobs.cost(free, _p) + self.kvel*self.cvel.cost(free, _p)

    def cost_grad(self, free, _p):
        return self.kobs*self.cobs.cost_grad(free, _p) + self.kvel*self.cvel.cost_grad(free, _p)
        
k_vel = 0.5
k_phi = 0.
k_obs = 1.e-3
def obj_compo2(free, _p):
    return k_vel * obj_meanvel(free, _p) + k_phi * obj_meanbank(free, _p) + k_obs * obj_obs(free, _p)

def obj_grad_compo2(free, _p):
    return k_vel * obj_grad_meanvel(free, _p) + k_phi * obj_grad_meanbank(free, _p) + k_obs * obj_grad_obs(free, _p)




    
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
    def __init__(self,
                 exp,
#                 _x_constraint = (-100, 100),
#                 _y_constraint = (-100, 100),
                 initialize=True
                 ):
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
        if 0:
            self._bounds = {_g._sphi(_g._st): exp.phi_constraint, #(_min_bank, _max_bank),
                        _g._sv(_g._st): exp.v_constraint}     #(_min_v, _max_v),
        else:
            self._bounds = {}
            self._bounds[_g._sphi(_g._st)] = exp.phi_constraint
            self._bounds[_g._sv(_g._st)] = exp.v_constraint
            #if 0:
            #    self._bounds[_g._sy(_g._st)] = (-1., 35.)
            #else:
            #    self._bounds[_g._sy(_g._st)-12.] = (-13., 23.)

        #if _x_constraint is not None: self._bounds[_g._sx(_g._st)] = _x_constraint
        #if _y_constraint is not None: self._bounds[_g._sy(_g._st)] = _y_constraint

        self.obstacles = exp.obstacles
        for _o in self.obstacles:
            cx, cy, rm = _o
            d2s = np.square(_g._sx(_g._st)-cx)+np.square(_g._sy(_g._st)-cy)
            #self._bounds[d2s] = (np.square(rm), np.square(rM))
        
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
        #breakpoint()
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
        _p.run(initial_guess)
        if 1:
            _p.prob.plot_objective_value()
            _p.prob.plot_trajectories(_p.solution)
            #_p.prob.plot_constraint_violations(_p.solution)
        _p.save_solution(filename)
    else:
        _p.load_solution(filename)



def plot(_p):
    fig, axes = plt.subplots(5, 1)
    axes[0].plot(_p.sol_time, _p.sol_x)
    d2p.decorate(axes[0], title='x', xlab='t in s', ylab='m', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[1].plot(_p.sol_time, _p.sol_y)
    d2p.decorate(axes[1], title='y', xlab='t in s', ylab='m', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[2].plot(_p.sol_time, np.rad2deg(_p.sol_psi))
    d2p.decorate(axes[2], title='$\\psi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[3].plot(_p.sol_time, np.rad2deg(_p.sol_phi))
    d2p.decorate(axes[3], title='$\\phi$', xlab='t in s', ylab='deg', legend=None, xlim=None, ylim=None, min_yspan=None)
    axes[4].plot(_p.sol_time, _p.sol_v)
    d2p.decorate(axes[4], title='$v$', xlab='t in s', ylab='m/s', legend=None, xlim=None, ylim=None, min_yspan=0.1)
    
def plot2d(_p):
     
    _f = plt.figure()
    _a = plt.gca()
    _a.plot(_p.sol_x, _p.sol_y)
    for _o in _p.obstacles:
        cx, cy, rm = _o
        _a.add_patch(plt.Circle((cx,cy),rm, alpha=0.25))
    _a.axis('equal')

class exp_0:
    cost = CostAirVel(12.)
    wind = WindField(w=[0.,0.])
    obstacles = ( )
    obj_scale = 1.
    t0, p0 = 0.,  ( 0.,  0.,    0.,    0., 10.)    # initial position: t0, ( x0, y0, psi0, phi0, v0)
    t1, p1 = 10., ( 0., 30., np.pi,    0., 10.)    # final position
    phi_constraint = (-np.deg2rad(30.), np.deg2rad(30.))
    v_constraint = (7., 20.)
    hz = 50.
    name = 'exp0'
    
class exp_1(exp_0):
    #wind = WindField(w=[2.,0.])
    obstacles = ((30, 0, 15), )
    cost = CostObstacle((obstacles[0][0], obstacles[0][1]), obstacles[0][2])
    #obj_scale = 1.e1
    #t1 = 4.
    name = 'exp1'

class exp_2(exp_0):
    #wind = WindField(w=[0.,0.])
    obstacles = ((33, 0, 15), (23, 30, 12))
    #cost = CostObstacles(obstacles)
    cost = CostVelObstacles(obstacles, kobs=1., kvel=0.1)
    #obj_scale = 1.
    t1 = 11.
    name = 'exp2'

class exp_3(exp_0):
    #cost = (obj_minvel, obj_grad_minvel)
    #obstacles = ((15, 10, 11, 1000), (35, -10, 11, 1000))
    #obj_scale = 1e-5
    pass

def main(force_recompute=False, exp=exp_2):
    _p = Planner(exp, initialize = force_recompute)
    print('Planner initialized')
    #filename = './optyplan.npz'
    compute_or_load(_p, force_recompute, f'./optyplan_{exp.name}.npz', tol=1e-9, max_iter=3000, initial_guess=None) # 1e-5
    print('Planner ran')
    plot(_p)
    plot2d(_p)
    plt.show()


    



    
if __name__ == '__main__':
    main(force_recompute='--force' in sys.argv)
