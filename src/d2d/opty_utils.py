import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation
import d2d.ploting as d2p

#
#  Misc stuff
#

def planner_timing(t0, t1, hz): # discretization of the time for initial trajectory
   duration  = t1 - t0
   num_nodes = int(duration*hz)+1
   time_step = 1./hz
   duration  = (num_nodes-1)*time_step
   print(f'time_step: {time_step:.3f}s ({hz:.1f}hz), duration {duration:.1f}s -> {num_nodes} nodes') 
   return num_nodes, time_step, duration


def norm_mpi_pi(v): return ( v + np.pi) % (2 * np.pi ) - np.pi

#
# formation flight
#
def line_formation(nac, p0, psi, dx, dy, v):
   return [(p0[0]+i*dx, p0[1]+i*dy, psi, 0., v) for i in range(nac)]

def arrow_formation(nac, p0, psi, dx, dy, v):
   w = int(nac/2)
   b1 = [(p0[0]-i*dx, p0[1]+i*dy, psi, 0., v) for i in range(w)]
   b2 = [(p0[0]+(i-nac+1)*dx, p0[1]+i*dy, psi, 0., v) for i in range(w, nac)]
   return b1 + b2

def diamond_formation(nac, p0, psi, dx, dy, v):
   w = int(np.ceil(nac/4))
   xs, ys = np.zeros((nac, 1)), np.zeros((nac, 1))
   for i in range(nac):
      if i < w:
         xs[i], ys[i] = p0[0] - i*dx,         p0[1] + i*dy
      elif i < 2*w:
         xs[i], ys[i] = p0[0] + (i-2*w)*dx, p0[1] + i*dy
      elif i < 3*w:
         xs[i], ys[i] = p0[0] + (i-2*w)*dx, p0[1] + (4*w-i)*dy
      else:
         xs[i], ys[i] = p0[0] + (4*w-i)*dx,  p0[1] + (4*w-i)*dy
   #plt.plot(xs, ys, '*')
   #plt.show()
   psis, vs, phis = psi * np.ones((nac, 1)), v*np.ones((nac, 1)), np.zeros((nac, 1))
   return np.hstack([xs, ys, psis, phis, vs]) 


def circle_formation(nac, c, r, alpha0, dalpha, v, dpsi=np.pi/2):
   alphas = np.arange(alpha0, alpha0+nac*dalpha, dalpha) #np.linspace(alpha0, alpha0+nac*dalpha, nac) 
   xs = (c[0]+r*np.cos(alphas)).reshape(nac, 1)
   ys = (c[1]+r*np.sin(alphas)).reshape(nac, 1)
   psis = norm_mpi_pi((alphas + dpsi)).reshape(nac, 1)
   #breakpoint()
   phis = np.arctan([v**2/r/9.81 for _ in range(nac)]).reshape(nac, 1) # FIXME - use wind
   vs = v*np.ones((nac, 1))
   return np.hstack([xs, ys, psis, phis, vs]) 
   

def random_states(rng, nac, xlim=(-50, 50), ylim=(-50, 50), v=12.):
   _xs = rng.uniform(xlim[0], xlim[1], nac).reshape(nac, 1)
   _ys = rng.uniform(ylim[0], ylim[1], nac).reshape(nac, 1)
   _psis = rng.uniform(-np.pi, np.pi, nac).reshape(nac,1)
   _vs, _phis = v*np.ones((nac,1)), np.zeros((nac,1))
   return np.hstack([_xs, _ys, _psis, _phis, _vs]) 



# Wind
class WindField:
   def __init__(self, w=[0.,0.]):
      self.w = w
        
   def sample_sym(self, _t, _x, _y):
      return self.w
    
   def sample_num(self, _t, _x, _y):
      return self.w
   def __str__(self): return f'{self.w} m/s'

# Symbols for one aircraft (time, state, input)
# returns equations of motion in symbolic notation
class Aircraft:
    def __init__(self, st=None, id=''):
        self._st = st or sym.symbols('t')
        self._sx, self._sy, self._sv, self._sphi, self._spsi = sym.symbols(f'x{id}, y{id}, v{id}, phi{id}, psi{id}', cls=sym.Function)
        self._state_symbols = (self._sx(self._st), self._sy(self._st), self._spsi(self._st))
        self._input_symbols = (self._sv, self._sphi)

    def get_eom(self, atm, g=9.81): # symbolic form of equations of motion
    # atm - instance of WindField class    
        wx, wy = atm.sample_sym(self._st, self._sx(self._st), self._sy(self._st))
        if 1:
            eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st)) + wx
            eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st)) + wy
            eq3 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
        else: # constant speed attempt
            # YET TO BE COMPLETED - THIS IS WHAT WE REQUIRE IN THE END
            eq1 = self._sx(self._st).diff() - self._sv * sym.cos(self._spsi(self._st)) + wx
            eq2 = self._sy(self._st).diff() - self._sv * sym.sin(self._spsi(self._st)) + wy
            eq3 = self._spsi(self._st).diff() - g / self._sv * sym.tan(self._sphi(self._st))
        return sym.Matrix([eq1, eq2, eq3])

# Symbols for one dimension 5 aircraft (time, state, input)
class Aircraft5d:
   def __init__(self, st=None, id=''):
      self._st = st or sym.symbols('t')
      self._sx, self._sy, self._spsi, self._sv, self._sphi, self._sv_sp, self._sphi_sp = sym.symbols(f'x{id}, y{id}, psi{id}, v{id}, phi{id}, vsp{id}, phisp{id}', cls=sym.Function)
      self._state_symbols = (self._sx(self._st), self._sy(self._st), self._spsi(self._st), self._sv(self._st), self._sphi(self._st))
      self._input_symbols = (self._sv_sp, self._sphi_sp)

   def get_eom(self, atm, g=9.81, tau_v=3., tau_phi=1.):
      wx, wy = atm.sample_sym(self._st, self._sx(self._st), self._sy(self._st))
      eq1 = self._sx(self._st).diff() - self._sv(self._st) * sym.cos(self._spsi(self._st)) + wx
      eq2 = self._sy(self._st).diff() - self._sv(self._st) * sym.sin(self._spsi(self._st)) + wy
      eq3 = self._spsi(self._st).diff() - g / self._sv(self._st) * sym.tan(self._sphi(self._st))
      eq4 = self._sv(self._st).diff() + 1./tau_v*(self._sv(self._st)-self._sv_sp(self._st))
      eq5 = self._sphi(self._st).diff() + 1./tau_phi*(self._sphi(self._st)-self._sphi_sp(self._st))
      return sym.Matrix([eq1, eq2, eq3, eq4, eq5])

   


# Cost functions
class CostAirVel: # constant air velocity
    def __init__(self, vsp=10.): # assumes flight velocity = 10 m/s  unless specified otherwise
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


class CostInput:
   def __init__(self, vsp=10., kvel=1., kbank=1.):
      self.vsp, self.kv, self.kphi = vsp, kvel, kbank
   def cost(self, free, _p):
      sum_phi_squared = np.sum(np.square(free[_p._slice_phi]))
      sum_err_vel_squared = np.sum(np.square(free[_p._slice_v]-self.vsp))
      return _p.obj_scale/_p.num_nodes* (self.kv*sum_err_vel_squared + self.kphi*sum_phi_squared)
   def cost_grad(self, free, _p):
      grad = np.zeros_like(free)
      grad[_p._slice_phi] = self.kphi*2*free[_p._slice_phi]
      grad[_p._slice_v] =  self.kv*2*(free[_p._slice_v]-self.vsp)
      grad *= _p.obj_scale/_p.num_nodes
      return grad
     
class CostObstacle: # penalyze circular obstacle
   def __init__(self, c=(30,0), r=15., kind=0):
      self.c, self.r = c, r
      self.kind = kind
      self.k=2.

   def cost1(self, free, _p):
      xs, ys = free[_p._slice_x], free[_p._slice_y]
      dxs, dys = xs-self.c[0], ys-self.c[1]
      if self.kind==0:
         ds = np.square(dxs)+np.square(dys)
         errs = np.exp(self.r**2-ds)
         es = np.clip(errs, 0., 1e3)
      else:
         d2 = np.square(dxs/self.r*self.k)+np.square(dys/self.r*self.k)
         es = np.exp(-d2) 
      return es
   
   def cost(self, free, _p):
      errs = self.cost1(free, _p)
      return _p.obj_scale/_p.num_nodes * np.sum(errs)

   def cost_grad(self, free, _p):
      grad = np.zeros_like(free)
      xs, ys = free[_p._slice_x], free[_p._slice_y]
      dxs, dys = xs-self.c[0], ys-self.c[1]
      if self.kind==0:
         ds = np.square(dxs)+np.square(dys)
         es = np.exp(self.r**2-ds)
         es = np.clip(es, 0., 1e3)#breakpoint()
      else:
         d2 = np.square(dxs/self.r*self.k)+np.square(dys/self.r*self.k)
         es = np.exp(-d2)
      grad[_p._slice_x] =  _p.obj_scale/_p.num_nodes*-2.*dxs*es
      grad[_p._slice_y] =  _p.obj_scale/_p.num_nodes*-2.*dys*es
      return grad

class CostObstacles: # penalyze set of circular obstacles
    def __init__(self, obss, kind=0):
        self.obss = [CostObstacle(c=(_o[0], _o[1]), r=_o[2], kind=kind) for _o in obss]
        
    def cost(self, free, _p):
        return np.sum([_c.cost(free, _p) for _c in self.obss])

    def cost_grad(self, free, _p):
        return np.sum([_c.cost_grad(free, _p) for _c in self.obss], axis=0)


class CostComposit: # a mix of the above
    def __init__(self, obss, vsp=10., kobs=1., kvel=1., kbank=1., obs_kind=0):
        self.kobs, self.kvel, self.kbank = kobs, kvel, kbank
        try: self.cobs = CostObstacles(obss, obs_kind)
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


#
#
#
def triangle(p0, p1, va, duration, num_nodes, go_left=1.):
    p0p1 = p1-p0
    d = np.linalg.norm(p0p1)                # distance between start and end
    u = p0p1/d; v = np.array([-u[1], u[0]]) # unit and normal vectors
    D = va*duration                         # distance to be traveled
    p2 = p0 + p0p1/2                        # center of the direct start-end leg
    if D > d:                               # We have time to spare, let make an isocele triangle
        p2 += np.sign(go_left) * np.sqrt(D**2-d**2)/2 * v

    n1 = int(num_nodes/2); n2 = num_nodes - n1
    _p0p2, _p2p1 = np.linspace(p0, p2, n1), np.linspace(p2, p1, n2)
    _p0p1 = np.vstack((_p0p2, _p2p1))
    p0p2 = p2-p0; psi0 = np.arctan2(p0p2[1], p0p2[0])
    p2p1 = p1-p2; psi1 = np.arctan2(p2p1[1], p2p1[0])
    psis = np.hstack((psi0*np.ones(n1), psi1*np.ones(n2)))
    phis, vs = np.zeros(num_nodes), va*np.ones(num_nodes)
    return  _p0p1[:,0], _p0p1[:,1], psis, phis, vs


def pts_on_circle(c=(0,0), r=20, alpha0=0., dalpha=np.deg2rad(20), n=3):
   alphas = alpha0 +np.arange(0, dalpha*n, dalpha)
   v, phi = 10, 0
   ps = [(c[0] + r*np.cos(alpha), c[1] + r*np.sin(alpha), alpha+np.pi/2, v, phi) for alpha in alphas]
   return ps
#

def plot_chrono(_p, save, _f=None, _a=None):
    if _f is None: fig, axes = plt.subplots(5, 1)
    else: fig, axes = _f, _a

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
    d2p.decorate(axes[4], title='$v_a$', xlab='t in s', ylab='m/s', legend=None, xlim=None, ylim=None, min_yspan=0.1)
    if save is not None:
        fn = f'{save}_chrono.png'
        print(f'saved {fn}'); plt.savefig(fn)
    return fig, axes
    
def plot2d(_p, save, _f=None, _a=None, label=''):
    _f = _f or plt.figure()
    _a = _a or plt.gca()
    _a.plot(_p.sol_x, _p.sol_y, solid_capstyle='butt', label=label)
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
    d2p.decorate(_a, title='$2D$', xlab='x in m', ylab='y in m', legend=True, xlim=None, ylim=None, min_yspan=0.1)  
    _a.axis('equal')
    if save is not None:
        fn = f'{save}_2d.png'
        print(f'saved {fn}'); plt.savefig(fn)
    return _f, _a
        
