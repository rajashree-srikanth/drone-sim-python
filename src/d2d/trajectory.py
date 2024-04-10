import math, numpy as np


#
# 2D+t trajectories
#

#
# 1D Utility fonctions
#
# Scalar trajectories (aka 1D)
# 
class CstOne:
    def __init__(self, c=-1.):
        self.c = c
    def get(self, t):
        return np.array([self.c, 0, 0, 0])

class AffineOne:
    def __init__(self, c1=-1., c2=0, duration=1.):
        self.c1, self.c2, self.duration = c1, c2, duration

    def get(self, t):
        return np.array([self.c1*t+self.c2, self.c1, 0, 0])

class SinOne:
    def __init__(self, c=0., a=1., om=1., duration=2*np.pi):
        self.duration = duration
        self.c, self.a, self.om = c, a, om
        self.t0 = 0.

    def get(self, t):
        alpha = self.om*(t-self.t0)
        asa, aca = self.a*np.sin(alpha), self.a*np.cos(alpha)
        return np.array([  self.c + asa,
                           self.om*aca,
                          -self.om**2*asa,
                          -self.om**3*aca])#,
                           #self.om**4*asa   ])
#
def arr(k,n): # arangements a(k,n) = n!/k!
    a,i = 1,n
    while i>n-k:
        a *= i; i -= 1
    return a

class PolynomialOne: # Min snap polynomial trajectories
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
        Y = np.zeros(self._der) # Horner method for computing polynomial value
        for d in range(0, self._der):
            v = self.coefs[d,-1]
            for j in range(self._order-2, -1, -1):
                v *= t
                v += self.coefs[d,j]
                Y[d] = v
        return Y


# 2D+t trajectories
# We represent them as a pair of smooth functions of time and  nder time derivatives
#
class Trajectory: # Trajectory base class
    desc = ""
    cx, cy, ncomp = np.arange(3)
    nder = 3
    extends = (0, 100, 0, 100)
    def __init__(self): self.t0 = 0.
    def get(self, t): return np.zeros((self.nder+1, self.ncomp))
    def reset(self, t0): self.t0 = t0

    def summarize(self):
        r = f'{self.desc}\n'
        r += f'duration: {self.duration:.2f}s\n'
        r += f'extends: {self.extends}'
        return r

    
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
    # sign of r specifies direction
    def __init__(self, c=[30., 30.],  r=30., v=10., t0=0., alpha0=0, dalpha=2*np.pi):
        self.c, self.r, self.v, self.t0 = np.asarray(c), r, v, t0 # mxm, m, m/s
        self.alpha0, self.dalpha = alpha0, dalpha # rad
        self.omega = self.v/self.r                # rad/s
        self.duration = np.abs(r)*dalpha/v

    def reset(self, t0): self.t0 = t0
       
    def get(self, t):
        alpha = (t-self.t0) * self.omega + self.alpha0
        ca, sa = np.cos(alpha), np.sin(alpha)
        p  = self.c+self.r*np.array([ca, sa])
        p1 = self.omega*self.r*np.array([-sa, ca])
        p2 = self.omega**2*self.r*np.array([-ca, -sa])
        p3 = self.omega**3*self.r*np.array([ sa, -ca])
        return np.array((p, p1, p2, p3))



class MinSnapPoly(Trajectory):
    def __init__(self, Y00=[0, 0], Y10=[1, 0], duration=1.):
        self.duration = duration
        Y0 = np.zeros((Trajectory.ncomp, Trajectory.nder+1))
        if len(np.asarray(Y00).shape) == 1: # we only got zero order derivatives, we assume others are null
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

    def reset(self, t0): self.t0 = t0

    def get(self, t):
        dt = t - self.t0
        Yc = np.zeros((5,4))
        dt_lapse = math.fmod(dt, self.duration)
        cur_step = np.argmax(self.steps_end > dt_lapse)
        Yc = self.steps[cur_step].get(dt_lapse)
        return Yc



class SpaceIndexedTraj(Trajectory):
    def __init__(self, geometry, dynamic):
        self.duration = dynamic.duration
        self.extends = geometry.extends
        self._geom, self._dyn = geometry, dynamic
        

    def set_dyn(self, dyn):
        self._dyn = dyn
        self.duration = dyn.duration

    def get(self, t):
        Yt = np.zeros((self._geom.nder+1, self._geom.ncomp))
        _lambda = self._dyn.get(t) # lambda(t), lambdadot(t)...
        _lambda[0] = np.clip(_lambda[0], 0., 1.)   # protect ourselvf against unruly dynamics 
        _g = self._geom.get(_lambda[0])  # g(lambda), dg/dlambda(lambda)...
        Yt[0, :] = _g[0,:]
        Yt[1, :] = _lambda[1]*_g[1,:]
        Yt[2, :] = _lambda[2]*_g[1,:] + _lambda[1]**2*_g[2,:]
        Yt[3, :] = _lambda[3]*_g[1,:] + 3*_lambda[1]*_lambda[2]*_g[2,:] + _lambda[1]**3*_g[3,:]
        #Yt[4, :] = _lambda[4]*_g[:,1] + (3*_lambda[2]**2+4*_lambda[1]*_lambda[3])*_g[:,2] + 6*_lambda[1]**2*_lambda[2]*_g[:,3] + _lambda[1]**4*_g[:,4]
        return Yt



## Let's make a controller instead
# class CircularFormationTraj(Trajectory):


# #   "ids": [102, 103, 104],
# #   "topology": [
# #     [ 1, 0],
# #     [-1, 1],
# #     [ 0,-1]
# #   ],
# #   "desired_intervehicle_angles_degrees": [0, 0],
# #   "gain": 10,
# #   "desired_stationary_radius_meters": 80
# # }



#     def __init__(self, c=[30., 30.],  r=30., v=10., t0=0., alpha0=0, dalpha=2*np.pi):
#         self.c, self.r, self.v, self.t0 = np.asarray(c), r, v, t0 # mxm, m, m/s
#         self.alpha0, self.dalpha = alpha0, dalpha # rad
#         self.omega = self.v/self.r                # rad/s
#         self.duration = np.abs(r)*dalpha/v
        

#     def reset(self, t0): self.t0 = t0
       
#     def get(self, t):
#         alpha = (t-self.t0) * self.omega + self.alpha0
#         ca, sa = np.cos(alpha), np.sin(alpha)
#         p  = self.c+self.r*np.array([ca, sa])
#         p1 = self.omega*self.r*np.array([-sa, ca])
#         p2 = self.omega**2*self.r*np.array([-ca, -sa])
#         p3 = self.omega**3*self.r*np.array([ sa, -ca])
#         return np.array((p, p1, p2, p3))


    
import matplotlib.pyplot as plt
def check_si(traj):
    time = np.arange(0, traj.duration, 0.01)
    lambdas = np.array([traj._dyn.get(t) for t in time])
    _f = plt.figure()
    _a = _f.subplots(4, 1)
    _a[0].plot(time, lambdas[:,0])
    _a[1].plot(time, lambdas[:,1])
    _a[2].plot(time, lambdas[:,2])
    _a[3].plot(time, lambdas[:,3])
    plt.show()
    breakpoint()
    
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
