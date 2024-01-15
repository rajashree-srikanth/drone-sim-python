import numpy as np

import d2d.trajectory as ddt


#
#
# A set of commonly used trajectories
#
#


trajectories = {}
def register(T): trajectories[T.name] = (T.desc, T)
def list_available():
    return ['{}: {}'.format(k,v[0]) for k,v in sorted(trajectories.items())]

class TrajDefaultCircle(ddt.TrajectoryCircle):
    name = "default_circle"
    desc = "30m radius (30,30) centered circle"
    extends = (-10, 70, -10, 70) # _xmin, _xmax, _ymin, _ymax 
    def __init__(self):
        ddt.TrajectoryCircle.__init__(self, c=[30., 30.],  r=30., v=10., t0=0., alpha0=0, dalpha=2*np.pi)
register(TrajDefaultCircle)


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
    def __init__(self, p1=[0,20], p2=[100,20], v=10., t0=0.):
        self.p1, self.p2, self.v, self.t0 = np.asarray(p1), np.asarray(p2), v, t0 # ends, velocity, initial time
        dep = self.p2-self.p1
        self.length = np.linalg.norm(dep)   # length
        self.un = dep/self.length           # unit vector
        self.duration = self.length/self.v  # duration

    def get(self, t):
        Yc = np.zeros((self.nder+1, self.ncomp))
        Yc[0,:] = self.p1 + self.un*self.v*(t-self.t0)
        Yc[1,:] =           self.un*self.v

        a, om = 10., 1.
        s, c = np.sin(om*(t-self.t0)), np.cos(om*(t-self.t0))
        y, yd, ydd, yddd = a*s, a*om*c, -a*om**2*s, -a*om**3*c
        Yc[0,1] += y
        Yc[1,1] += yd
        Yc[2,1] += ydd
        Yc[3,1] += yddd
        return Yc#Yc.T
register(TrajSlalom) 


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
         
def print_available():
    print('Available trajectories:')
    for i, n in enumerate(list_available()):
        print(f'{i} -> {n}')

def get(traj_name):
    return trajectories[traj_name][1](), trajectories[traj_name][0]

