import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation

def planner_timing(t0, t1, hz):
   duration  = t1 - t0
   num_nodes = int(duration*hz)+1
   time_step = 1./hz
   duration  = (num_nodes-1)*time_step#duration
   print(f'time_step: {time_step:.3f}s ({hz:.1f}hz), duration {duration:.1f}s -> {num_nodes} nodes') 
   return num_nodes, time_step, duration


# Wind
class WindField:
    def __init__(self, w=[0.,0.]):
        self.w = w
        
    def sample_sym(self, _t, _x, _y):
        return self.w
    
    def sample_num(self, _t, _x, _y):
        return self.w

# Symbols for one aircraft (time, state, input)
class Aircraft:
    def __init__(self, st=None):
        self._st = st or sym.symbols('t')
        self._sx, self._sy, self._sv, self._sphi, self._spsi = sym.symbols('x, y, v, phi, psi', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sy(self._st), self._spsi(self._st))
        #self._input_symbols = (self._sphi) #self._sv, self._sphi)
        self._input_symbols = (self._sv, self._sphi)

    def get_eom(self, atm, g=9.81):
        wx, wy = atm.sample_sym(self._st, self._sx(self._st), self._sy(self._st))
        if 1:
            eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st)) + wx
            eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st)) + wy
            eq3 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        else:
            eq1 = self._sx(self._st).diff() - self._sv * sym.cos(self._spsi(self._st)) + wx
            eq2 = self._sy(self._st).diff() - self._sv * sym.sin(self._spsi(self._st)) + wy
            eq3 = self._spsi(self._st).diff() - g / self._sv * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3])



# Cost functions
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
    use_mean = True
    def cost(self, free, _p):
        phis = free[_p._slice_phi]
        if self.use_mean: return _p.obj_scale * np.sum(np.square(phis))/_p.num_nodes
        else: return _p.obj_scale * np.max(np.square(phis))
    
    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        phis = free[_p._slice_phi]
        if self.use_mean: grad[_p._slice_phi] = _p.obj_scale/_p.num_nodes*2*phis
        else:
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
        try: self.cobs = CostObstacles(obss)
        except: pass
        self.cvel = CostAirVel(vsp)
        self.cbank = CostBank()

    def cost(self, free, _p):
        try:
            return self.kobs*self.cobs.cost(free, _p) + self.kvel*self.cvel.cost(free, _p) + self.kbank*self.cbank.cost(free, _p)
        except AttributeError:  # no obstacles
            return self.kvel*self.cvel.cost(free, _p) + self.kbank*self.cbank.cost(free, _p)

    def cost_grad(self, free, _p):
        try:
            return self.kobs*self.cobs.cost_grad(free, _p) + self.kvel*self.cvel.cost_grad(free, _p) + self.kbank*self.cbank.cost_grad(free, _p)
        except AttributeError:  # no obstacles
            return self.kvel*self.cvel.cost_grad(free, _p) + self.kbank*self.cbank.cost_grad(free, _p)
