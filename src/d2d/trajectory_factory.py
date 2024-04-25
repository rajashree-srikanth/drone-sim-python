import numpy as np

import d2d.trajectory as ddt
import d2d.guidance as d2guid
import d2d.utils as d2u

#
#
# A set of commonly used trajectories
#
#


trajectories = {}
def register(T): trajectories[T.name] = (T.desc, T)
def list_available():
    return ['{}: {}'.format(k,v[0]) for k,v in sorted(trajectories.items())]

class TrajCircle(ddt.TrajectoryCircle):
    name = "circle"
    desc = "30m radius (30,30) centered circle"
    extends = (-10, 70, -10, 70) # _xmin, _xmax, _ymin, _ymax 
    def __init__(self):
        ddt.TrajectoryCircle.__init__(self, c=[30., 30.],  r=30., v=10., t0=0., alpha0=0, dalpha=2*np.pi)
register(TrajCircle)


class TrajTwoLines(ddt.CompositeTraj):
    name = "two_lines"
    desc = "example of composite trajectory"
    extends = (-20, 120, -20, 60)  # _xmin, _xmax, _ymin, _ymax
    def __init__(self):
        Y0, Y1, Y2 = [0,0], [50, 50], [100, 0]
        s1 = ddt.TrajectoryLine(Y0, Y1, v=10., t0=0.)
        s2 = ddt.TrajectoryLine(Y1, Y2, v=10., t0=s1.duration)
        ddt.CompositeTraj.__init__(self, [s1, s2])
register(TrajTwoLines)       

class TrajSquare(ddt.CompositeTraj):
    name = "square"
    desc = "example of composite trajectory"
    extends = (-10, 60, -10, 60)  # _xmin, _xmax, _ymin, _ymax
    def __init__(self):
        Y0, Y1, Y2, Y3 = [0,0], [50, 0], [50, 50], [0, 50]
        s1 = ddt.TrajectoryLine(Y0, Y1, v=10., t0=0.)
        s2 = ddt.TrajectoryLine(Y1, Y2, v=10., t0=s1.duration)
        s3 = ddt.TrajectoryLine(Y2, Y3, v=10., t0=s1.duration+s2.duration)
        s4 = ddt.TrajectoryLine(Y3, Y0, v=10., t0=s1.duration+s2.duration+s3.duration)
        ddt.CompositeTraj.__init__(self, [s1, s2, s3, s4])
register(TrajSquare)       

class TrajLineWithIntro(ddt.CompositeTraj):
    name = "line_with_intro"
    desc = "line with circle_intro"
    def __init__(self, Y0=[0, 0], Y1=[0, 50], Y2=[100, 50], r=-25.):
        Yc = (np.asarray(Y0)+Y1)/2
        s1 = ddt.TrajectoryCircle(c=Yc,  r=r, v=10., alpha0=np.pi/2, dalpha = np.pi)
        s2 = ddt.TrajectoryLine(Y1, Y2, v=10., t0=s1.duration)
        ddt.CompositeTraj.__init__(self, [s1, s2])
        self.extends = (-50, 130, -30, 130)  # _xmin, _xmax, _ymin, _ymax
register(TrajLineWithIntro) 


class TrajWithIntro(ddt.CompositeTraj):
    name = "traj_with_intro"
    desc = "traj with circle_intro"
    def __init__(self, Y0, traj, v=10, duration=8.):
        Y1 = traj.get(0)[0] # start position of next trajectory step
        Y2 = (Y0+Y1)/2
        Y0Y1 = Y1-Y0
        y0y1 = np.linalg.norm(Y0Y1)
        n = 1/y0y1*Y0Y1
        p = np.array(-n[1], n[0])
        z = 0.
        beta = np.arctan2(z, y0y1)
        alpha = np.pi - 2*beta
        dist = v*duration
        R = y0y1/np.cos(beta)
        #R = dist/2/np.pi/alpha
        #breakpoint()
        #Yc = (np.asarray(Y0)+Y1)/2
        #s1 = ddt.TrajectoryCircle(c=Yc,  r=r, v=10., alpha0=np.pi/2, dalpha = np.pi)
        v_line = y0y1/duration # m/s in straight line
        intro = ddt.TrajectoryLine(Y0, Y1, v=v_line)
        #self.extends = (-50, 130, -30, 130)  # _xmin, _xmax, _ymin, _ymax
        ddt.CompositeTraj.__init__(self, [intro, traj])
# we don't register(TrajWithIntro) as it's not meant to be standalone, it needs to be in a scenario


# TODO not sure what it is supposed to be
class CircleWithIntro(ddt.CompositeTraj):
    name = "Circle_intro"
    desc = "Circle with circle_intro"
    def __init__(self, c=[0, 0, -0.5], r=1., v=3., dt_intro=1.8, dt_stay=0.):
        eps = np.deg2rad(2)
        circle = Circle(c,  r,  v, np.pi/2-eps, 2*np.pi+2*eps)
        Y0 = [0, 0, 0, 0]
        Y1 = circle.get(0.)
        Y2 = circle.get(circle.duration)
        steps = [SmoothLine(Y0, Y1, duration=dt_intro),
                 circle,
                 SmoothLine(Y2, Y0, duration=dt_intro),
                 Cst(Y0, dt_stay)]
        CompositeTraj.__init__(self, steps)




class TrajMinSnapDemo(ddt.MinSnapPoly):
    name = "demo_minsnap"
    desc = "demo_minsnap"
    extends = (-10, 210, -10, 210)  # _xmin, _xmax, _ymin, _ymax
    def __init__(self):
        Y0 = [[0, 10, 0, 0],  [0, 0, 0, 0]]
        Y1 = [[200, 0, 0, 0], [200, 10, 0, 0]]
        ddt.MinSnapPoly.__init__(self, Y0,  Y1, duration=33.65) # duration hand optimized to keep ground velocity constant
register(TrajMinSnapDemo) 


class TrajSlalom(ddt.Trajectory):
    name = "slalom"
    desc = "slalom"
    extends = (-10, 100, -10, 50)  # _xmin, _xmax, _ymin, _ymax
    def __init__(self, p1=[0,20], p2=[100,20], v=10., t0=0., phi=0.):
        self.p1, self.p2, self.v, self.t0 = np.asarray(p1), np.asarray(p2), v, t0 # ends, velocity, initial time
        self.phi = phi # sine phase
        dep = self.p2-self.p1
        self.length = np.linalg.norm(dep)   # length
        self.un = dep/self.length           # unit vector
        self.duration = self.length/self.v  # duration

    def get(self, t):
        Yc = np.zeros((self.nder+1, self.ncomp))
        Yc[0,:] = self.p1 + self.un*self.v*(t-self.t0)
        Yc[1,:] =           self.un*self.v

        a, om = 10., 1.; alpha = om*(t-self.t0+self.phi)
        s, c = np.sin(alpha), np.cos(alpha)
        y, yd, ydd, yddd = a*s, a*om*c, -a*om**2*s, -a*om**3*c
        Yc[0,1] += y
        Yc[1,1] += yd
        Yc[2,1] += ydd
        Yc[3,1] += yddd
        return Yc#Yc.T
register(TrajSlalom) 


class TrajTabulated(ddt.Trajectory):
    name = "tabulated"
    desc = "tabulated"
    extends = (-5, 25, -10, 20)  # _xmin, _xmax, _ymin, _ymax
    def __init__(self, filename='./optyplan_exp0.npz'):
        _data =  np.load(filename)
        labels = ['sol_time', 'sol_x', 'sol_y', 'sol_psi', 'sol_phi', 'sol_v', 'wind']
        self.sol_time, self.sol_x, self.sol_y, self.sol_psi, self.sol_phi, self.sol_v, self.wind = [_data[k] for k in labels]
        print(f'loaded {filename}')
        self.t0 = 0.
        self.duration = self.sol_time[-1]
        self.compute_extends()

    def get(self, t):
        Yc = np.zeros((self.nder+1, self.ncomp))
        idx = np.argmin(t>self.sol_time)
        Yc[0,0] = self.sol_x[idx] 
        Yc[0,1] = self.sol_y[idx]
        (wx, wy), v, psi = self.wind[idx], self.sol_v[idx], self.sol_psi[idx]
        Yc[1,0] = v * np.cos(psi) + wx
        Yc[1,1] = v * np.sin(psi) + wy
        #breakpoint()
        return Yc

register(TrajTabulated) 



class TrajSiDemo(ddt.SpaceIndexedTraj):
    name = "sidemo"
    desc = "space indexed trajectory demo"
    extends = (-10, 100, -10, 50)  # _xmin, _xmax, _ymin, _ymax
    def __init__(self, p1=[0,20], p2=[100,20], duration=10., t0=0.):
        #dynamic = ddt.AffineOne(1./duration, 0., duration=duration)
        dynamic = ddt.PolynomialOne([0, 0.05, 0, 0], [1, 0.05, 0, 0], duration=duration)
        geometry = ddt.TrajectoryLine([0, 20], [100, 20], v=100)
        ddt.SpaceIndexedTraj.__init__(self, geometry, dynamic)
        
register(TrajSiDemo)

import scipy.interpolate as interpolate
class TrajSpline(ddt.Trajectory):
    name = "spline"
    desc = "spline dev"
    def __init__(self, waypoints=None, duration=None):
        self.waypoints = waypoints or np.array([[0., 0.], [50, 50], [100, 0], [150, 50], [200, 0]])
        if duration is None:
            v = 10.
            duration = np.sum(np.linalg.norm(self.waypoints[1:]-self.waypoints[:-1], axis=1))/v
        self.duration = duration
       
        l = np.linspace(0, self.duration, len(self.waypoints))
        self.splines = [interpolate.InterpolatedUnivariateSpline(l, self.waypoints[:,i], k=4) for i in range(2)]
        #self.splines = [interpolate.InterpolatedUnivariateSpline(l, self.waypoints[:,i], k=3) for i in range(2)]
        self.extends= [0, 200, -20, 80]

    def get(self, t):
        #l = np.fmod(t, self.duration)/self.duration
        t = np.fmod(t, self.duration)
        #Yl = np.zeros((self.nder+1, self.ncomp))
        #Yl[0] = [self.splines[i](l) for i in range(2)]
        Yl = np.array([self.splines[i].derivatives(t) for i in range(self.ncomp)])[:,:self.nder+1].T
        return Yl

        
register(TrajSpline)

class SplineOne:
    def __init__(self, xs, ys):
        self.nder=3
        self.dyn = interpolate.InterpolatedUnivariateSpline(xs, ys, k=4)
        #self.dyn = interpolate.UnivariateSpline(xs, ys, k=4, s=1.)
        self.duration = xs[-1]
    def get(self, t):
        
        return np.array(self.dyn.derivatives(t))[:self.nder+1].T


class FooOne:
    def __init__(self, xs, ys):
        self.xs, self.ys = np.asarray(xs), np.asarray(ys)
        self.dxs, self.dys = self.xs[1:]-self.xs[:-1], self.ys[1:]-self.ys[:-1]
        self.ds = self.dys/self.dxs
        self.duration = xs[-1]

    def get(self, t):
        _i = np.where(t >= self.xs)[0][-1]
        return np.array([self.ys[_i] + (t-self.xs[_i])*self.ds[_i], self.ds[_i], 0, 0])

    
import matplotlib.pyplot as plt
import d2d.ploting as d2plot
class TrajSiSpline(ddt.SpaceIndexedTraj):
    name = "sispline"
    desc = "spline dev"
    extends = (-10, 100, -10, 50)
    def __init__(self, duration=30.):
        print(duration)
        #geometry = TrajSpline(waypoints=None, duration=1.)
        geometry = ddt.TrajectoryCircle( c=[30., 30.],  r=30., v=2*np.pi*30., t0=0., alpha0=0, dalpha=3*np.pi/2)
        dynamic = ddt.AffineOne(1./duration, 0., duration=duration)
        ddt.SpaceIndexedTraj.__init__(self, geometry, dynamic)
        print(self.duration)
        self._dyn1 = self._dyn

        windfield = d2guid.WindField([5, 0])
        
        self.ts = np.arange(0, self._dyn.duration, 0.5)
        self.lambdas = [self._dyn.get(_t) for _t in self.ts]
        self.lambdas = np.array(self.lambdas)[:, 0]
        #self.lambdas = np.sin(ts/self.duration*np.pi/2)
        self._dyn2 = SplineOne(self.ts, self.lambdas)
        self.lambdas_dyn2 = np.array([self._dyn2.get(_t) for _t in self.ts])
        npts = 10
        xs = np.linspace(0, 30, npts)
        vtarget = 10.
        def err_fun(p):
            ys = np.concatenate(([0.],np.cumsum(p)))
            self.set_dyn(FooOne(xs, ys))
            flat_out = np.array([self.get(_t) for _t in self.ts]) 
            pos = flat_out[:,0]
            vel_gnd = flat_out[:,1]
            wind = np.array([windfield.sample(_t, _p) for _t, _p in zip(self.ts, pos)])
            vel_air = vel_gnd - wind
            vair = np.linalg.norm(vel_air, axis=1)
            err = np.mean(np.square(vair-vtarget))
            #err = np.mean(np.square(np.linalg.norm(vel_gnd, axis=1)-vtarget))
            return err
        import scipy.optimize
        res = scipy.optimize.minimize(err_fun, [1./npts]*(npts-1))
        ys = np.concatenate(([0.],np.cumsum(res.x)))
        #self._dyn4 = FooOne(xs, ys)
        self._dyn4 = SplineOne(xs, ys)
                    
        #self.plot()
        plt.show()
        

    def plot(self):
        _a = plt.gcf().subplots(2, 2)
        self.set_dyn(self._dyn1)
        dyn1 = [self._dyn.get(_t) for _t in self.ts]
        _a[0,0].plot(self.ts, dyn1)
        d2plot.decorate(_a[0,0], 'lambda')
        v1 = [np.linalg.norm(self.get(_t)[1]) for _t in self.ts]
        _a[1,0].plot(self.ts, v1)
        d2plot.decorate(_a[1,0], 'v')

        self.set_dyn(self._dyn4)
        dyn3 = [self._dyn.get(_t) for _t in self.ts]
        #_a[0,1].plot(ts, self.lambdas2)
        _a[0,1].plot(self.ts, dyn3)
        d2plot.decorate(_a[0,1], 'lambda2')
        v3 = [np.linalg.norm(self.get(_t)[1]) for _t in self.ts]
        _a[1,1].plot(self.ts, v3)
        d2plot.decorate(_a[1,1], 'v2', min_yspan=0.1)
        
        
register(TrajSiSpline)


def print_available():
    print('Available trajectories:')
    for i, n in enumerate(list_available()):
        print(f'{i} -> {n}')

def get(traj_name):
    return trajectories[traj_name][1](), trajectories[traj_name][0]

