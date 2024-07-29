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


class CostNull:
    def cost(self, free, _p): return 0.
    def cost_grad(self, free, _p): return np.zeros_like(free)

class CostAirvel:
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

class CostBank:
    def cost(self, free, _p):
        sum_phi_squared = np.sum([np.sum(np.square(free[slice_phi])) for slice_phi in _p._slice_phi])
        return _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * sum_phi_squared
    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        for _s in _p._slice_phi:
            grad[_s] = _p.obj_scale/_p.num_nodes/_p.acs.nb_aicraft * 2*free[_s]
        return grad

class CostInput:
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


class CostObstacle: # circular obstacle - FIXME, only for plane 0!!!
    def __init__(self, c=(0,0), r=10., kind=0):
           self.c, self.r = c, r
           self.kind = kind
           self. k = 2
           
    def cost(self, free, _p):
        xs, ys = free[_p._slice_x[0]], free[_p._slice_y[0]]
        dxs, dys = xs-self.c[0], ys-self.c[1]
        if self.kind==0:
            ds = np.square(dxs)+np.square(dys)
            es = np.exp(self.r**2-ds)
            es = np.clip(es, 0., 1e3)
        else:
            d2 = np.square(dxs/self.r*self.k)+np.square(dys/self.r*self.k)
            es = np.exp(-d2)
        return _p.obj_scale/_p.num_nodes * np.sum(es)

    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        xs, ys = free[_p._slice_x[0]], free[_p._slice_y[0]]
        dxs, dys = xs-self.c[0], ys-self.c[1]
        if self.kind==0:
            ds = np.square(dxs)+np.square(dys)
            es = np.exp(self.r**2-ds)
            es = np.clip(es, 0., 1e3)#breakpoint()
        else:
            d2 = np.square(dxs/self.r*self.k)+np.square(dys/self.r*self.k)
            es = np.exp(-d2)
            
        grad[_p._slice_x[0]] =  _p.obj_scale/_p.num_nodes*-2.*dxs*es
        grad[_p._slice_y[0]] =  _p.obj_scale/_p.num_nodes*-2.*dys*es
        return grad
    
class CostObstacles: # penalyze set of circular obstacles
    def __init__(self, obss, kind=0):
        self.obss = [CostObstacle(c=(_o[0], _o[1]), r=_o[2], kind=kind) for _o in obss]
        
    def cost(self, free, _p):
        return np.sum([_c.cost(free, _p) for _c in self.obss])

    def cost_grad(self, free, _p):
        return np.sum([_c.cost_grad(free, _p) for _c in self.obss], axis=0)



class CostCollision: #
    def __init__(self, r=3., k=2.):
        self.r, self.k = r, k
    def cost(self, free, _p):
        ac1x, ac1y = free[_p._slice_x[1]], free[_p._slice_y[1]]
        ac0x, ac0y = free[_p._slice_x[0]], free[_p._slice_y[0]]
        dx, dy = ac0x-ac1x, ac0y-ac1y
        if 0: #self.kind==0:
            ds = np.square(dxs)+np.square(dys)
            es = np.exp(self.r**2-ds)
            es = np.clip(es, 0., 1e3)
        else:
            d2 = np.square(dx/self.r*self.k)+np.square(dy/self.r*self.k)
            es = np.exp(-d2)
        return _p.obj_scale/_p.num_nodes * np.sum(es)

    def cost_grad(self, free, _p):
        grad = np.zeros_like(free)
        ac1x, ac1y = free[_p._slice_x[1]], free[_p._slice_y[1]]
        ac0x, ac0y = free[_p._slice_x[0]], free[_p._slice_y[0]]
        dx, dy = ac0x-ac1x, ac0y-ac1y
        if 0: #self.kind==0:
            ds = np.square(dxs)+np.square(dys)
            es = np.exp(self.r**2-ds)
            es = np.clip(es, 0., 1e3)#breakpoint()
        else:
            d2 = np.square(dx/self.r*self.k)+np.square(dy/self.r*self.k)
            es = np.exp(-d2)
            
        grad[_p._slice_x[0]] =  _p.obj_scale/_p.num_nodes*-2.*dx*es
        grad[_p._slice_y[0]] =  _p.obj_scale/_p.num_nodes*-2.*dy*es
        grad[_p._slice_x[1]] =  _p.obj_scale/_p.num_nodes* 2.*dx*es
        grad[_p._slice_y[1]] =  _p.obj_scale/_p.num_nodes* 2.*dy*es
        return grad


class CostCollision2: #
    def __init__(self, r=3., k=2.):
        self.r, self.k = r, k
    def cost(self, free, _p):
        nac = _p.acs.nb_aicraft
        if nac < 2: return 0.
        cs = np.zeros((nac, nac))
        for ac0 in range(nac-1):
            for ac1 in range(ac0+1, nac):
                ac1x, ac1y = free[_p._slice_x[ac1]], free[_p._slice_y[ac1]]
                ac0x, ac0y = free[_p._slice_x[ac0]], free[_p._slice_y[ac0]]
                dx, dy = ac0x-ac1x, ac0y-ac1y
                d2 = np.square(dx/self.r*self.k)+np.square(dy/self.r*self.k)
                cs[ac0, ac1] = np.sum(np.exp(-d2))
        #breakpoint()
        return _p.obj_scale/_p.num_nodes * np.sum(cs)

    def cost_grad(self, free, _p):
        nac = _p.acs.nb_aicraft
        grad = np.zeros_like(free)
        if nac < 2: return grad
        for ac0 in range(nac-1):
            for ac1 in range(ac0+1, nac):
                ac1x, ac1y = free[_p._slice_x[ac1]], free[_p._slice_y[ac1]]
                ac0x, ac0y = free[_p._slice_x[ac0]], free[_p._slice_y[ac0]]
                dx, dy = ac0x-ac1x, ac0y-ac1y
                d2 = np.square(dx/self.r*self.k)+np.square(dy/self.r*self.k)
                es = np.exp(-d2)
                grad[_p._slice_x[ac0]] -= 2.*dx*es 
                grad[_p._slice_y[ac0]] -= 2.*dy*es
                grad[_p._slice_x[ac1]] += 2.*dx*es 
                grad[_p._slice_y[ac1]] += 2.*dy*es
        grad *= (_p.obj_scale/_p.num_nodes)
        return grad

    
    
class CostComposit:
    def __init__(self, kvel=1., kbank=1., kobs=float('Nan'), kcol=float('NaN'), vsp=10., obss=[], obs_kind=0, rcol=3.):
        self.kvel, self.kbank, self.kobs, self.kcol = kvel, kbank, kobs, kcol
        self.ci = CostInput(vsp, kvel, kbank)
        if kobs is not float('NaN'): self.cobs = CostObstacles(obss, obs_kind)
        if kcol is not float('NaN'): self.ccol = CostCollision2(r=rcol)
        
        
    def cost(self, free, _p):
        res = self.ci.cost(free, _p)
        if not np.isnan(self.kobs): res += self.kobs*self.cobs.cost(free, _p)
        if not np.isnan(self.kcol): res += self.kcol*self.ccol.cost(free, _p)
        return res
    
    def cost_grad(self, free, _p):
        res = self.ci.cost_grad(free, _p)
        if not np.isnan(self.kobs): res += self.kobs*self.cobs.cost_grad(free, _p)
        if not np.isnan(self.kcol): res += self.kcol*self.ccol.cost_grad(free, _p)
        return res

