import math, numpy as np


#
# 2D+t trajectories
#


class Trajectory:
    cx, cy, ncomp = np.arange(3)
    nder = 3


class TrajectoryLine(Trajectory):

    def __init__(self, p1, p2, v=10., t0=0.):
        self.p1, self.p2, self.v, self.t0 = np.asarray(p1), np.asarray(p2), v, t0 # ends, velocity, initial time
        dep = self.p2-self.p1
        self.length = np.linalg.norm(dep)   # length
        self.un = dep/self.length           # unit vector
        self.duration = self.length/self.v  # duration

    def reset(self, t0): self.t0 = t0

    def get(self, t):
        Yc = np.zeros((Trajectory.nder+1, Trajectory.ncomp))
        Yc[0,:3] = self.p1 + self.un*self.v*(t-self.t0)
        Yc[1,:3] =           self.un*self.v
        return Yc#Yc.T
    

class TrajectoryCircle(Trajectory):

    def __init__(self, c=[30., 30.],  r=30., v=10., t0=0., alpha0=0, dalpha=2*np.pi):
        self.c, self.r, self.v, self.t0 = np.asarray(c), r, v, t0 # mxm, m, m/s
        self.alpha0, self.dalpha = alpha0, dalpha # rad
        self.omega = self.v/self.r                # rad/s
        self.duration = r*dalpha/v

    def reset(self, t0): self.t0 = t0
       
    def get(self, t):
        alpha = t * self.omega + self.alpha0
        ca, sa = np.cos(alpha), np.sin(alpha)
        p  = self.c+self.r*np.array([ca, sa])
        p1 = self.omega*self.r*np.array([-sa, ca])
        p2 = self.omega**2*self.r*np.array([-ca, -sa])
        p3 = self.omega**3*self.r*np.array([ sa, -ca])
        return np.array((p, p1, p2, p3))




#
# Min snap polynomial trajectories
#
def arr(k,n): # arangements a(k,n) = n!/k!
    a,i = 1,n
    while i>n-k:
        a *= i; i -= 1
    return a

class PolynomialOne:
    def __init__(self, Y0, Y1, duration):
        self.duration = duration
        _der = len(Y0)    # number of time derivatives
        _order = 2*_der   # we need twice as many coefficients
        self._der, self._order = _der, _order
        # compute polynomial coefficients for time derivative zeros
        self.coefs = np.zeros((_der, _order))
        M1 = np.zeros((_der, _der))
        for i in range(_der):
            M1[i,i] = arr(i,i)
        self.coefs[0, 0:_der] = np.dot(np.linalg.inv(M1), Y0)
        M3 = np.zeros((_der, _der))
        for i in range(_der):
            for j in range(i, _der):
                M3[i,j] = arr(i,j) * duration**(j-i)
        M4 = np.zeros((_der, _der))
        for i in range(_der):
            for j in range(_der):
                M4[i,j] = arr(i, j+_der) * duration**(j-i+_der)
        M3a0k = np.dot(M3, self.coefs[0, 0:_der])
        self.coefs[0, _der:_order] = np.dot(np.linalg.inv(M4), Y1 - M3a0k)
        # fill in coefficients for the subsequent time derivatives  
        for d in range(1,_der):
            for pow in range(0,2*_der-d):
                self.coefs[d, pow] = arr(d, pow+d)*self.coefs[0, pow+d]

    def get(self, t):
        # Horner method for computing polynomial value
        Y = np.zeros(self._der)
        for d in range(0, self._der):
            v = self.coefs[d,-1]
            for j in range(self._order-2, -1, -1):
                v *= t
                v += self.coefs[d,j]
                Y[d] = v
        return Y

class MinSnapPoly(Trajectory):
    def __init__(self, Y00=[0, 0], Y10=[1, 0], duration=1.):
        self.duration = duration
        Y0 = np.zeros((Trajectory.ncomp, Trajectory.nder+1))# [_x, _y, _z, _psi, _ylen]
        if len(np.asarray(Y00).shape) == 1: # we only got zero order derivatives
            Y0[:,0] = Y00
        else:
            Y0 = Y00
        Y1 = np.zeros((Trajectory.ncomp, Trajectory.nder+1))
        if len(np.asarray(Y10).shape) == 1: # we only got zero order derivatives
            Y1[:,0] = Y10
        else:
            Y1 = Y10

        self._polys = [PolynomialOne(Y0[i], Y1[i], self.duration) for i in range(Trajectory.ncomp)]
        self.t0 = 0


    def reset(self, t0):
        self.t0 = t0

    def get(self, t):
        Yc = np.array([p.get(t-self.t0) for p in self._polys])
        return Yc.T


class CompositeTraj:
    def __init__(self, steps):
        self.steps = steps
        self.steps_dur = [s.duration for s in self.steps]
        self.steps_end = np.cumsum(self.steps_dur)
        self.duration = np.sum(self.steps_dur)
        for s, st in zip(self.steps[1:], self.steps_end):
            s.reset(st)
        self.t0 = 0.

    def reset(self, t0):
        #print('reset at {}'.format(t0))
        self.t0 = t0

    def get(self, t):
        dt = t - self.t0
        Yc = np.zeros((5,4))
        dt_lapse = math.fmod(dt, self.duration)
        cur_step = np.argmax(self.steps_end > dt_lapse)
        #print('get t {} dt {} cur_step {}'.format(t, dt, cur_step))
        Yc = self.steps[cur_step].get(dt_lapse)
        return Yc


class CircleWithIntro(CompositeTraj):

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


    

# check trajectory consistency
# stolen from PAT, needs testing
def check_consistency(time, Y):
    Ycheck = np.zeros_like(Y)
    # compute numerical differentiation of provided trajectory
    Ycheck[:,:,1] = np.gradient(Y[:,:,0], time[1]-time[0], axis=0)
    # compute further numerical differentiations
    for j in range(2, _nder):
        Ycheck[:,:,j] = np.gradient(Ycheck[:,:,j-1], time[1]-time[0], axis=0)

    figure, axes = plot(time, Y)
    _s = 4
    for i in range(_ylen): # x, y, z, psi
        for j in range(1, _nder): # the four time derivatives
            axes[j,i].plot(time[j:-j], np.rad2deg(Ycheck[j:-j,i,j]) if i == _psi else Ycheck[j:-j,i,j], label="check")
