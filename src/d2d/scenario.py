import numpy as np

import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.guidance as d2guid
import d2d.dynamic as d2dyn
from d2d.dynamic import Aircraft

#
#
# We define a set of commonly used scenarios (vehicles, trajectories, wind, etc)
#
#

_scenarios = {}
def register(S): _scenarios[S.name] = (S.desc, S)
def list_available():
    return ['{}: {}'.format(k,v[0]) for k,v in sorted(_scenarios.items())]

_default_dt = 0.01 # not proud of that :(


class Scenario:
    def __init__(self):
        # we expect time and trajs provided by children classes
        nv = len(self.trajs) # number of vehicles
        # we fill in some defaults
        try: self.aircrafts
        except AttributeError: self.aircrafts = [Aircraft() for i in range(nv)]
        try: self.perts
        except AttributeError: self.perts=[np.zeros((len(self.time), Aircraft.s_size)) for i in range(nv)]

    def summarize(self):
        r = f'{len(self.trajs)} trajectories\n'
        r += f'duration: {self.time[-1]-self.time[0]:.2f}s\n'
        r += f'wind: {self.windfield.summarize()}'
        return r
    
class ScenLine(Scenario):
    name = 'line'
    desc = 'line'
    def __init__(self):
        Y0, Y1 = [0,25], [100, 25]
        self.trajs = [ddt.TrajectoryLine(Y0, Y1, v=10., t0=0.)]
        self.extends = (-10, 110, 0, 50) # _xmin, _xmax, _ymin, _ymax
        self.windfield = d2guid.WindField()
        self.time = np.arange(0, 12., 0.01)
        self.X0s = [[10, 10, 0, 0, 10]]
        self.perts = [np.zeros((len(self.time), d2dyn.Aircraft.s_size))]
        self.perts[0][600,d2dyn.Aircraft.s_y]  =  10
        Scenario.__init__(self)
register(ScenLine)

class ScenLine2(Scenario):
    name = 'line2'
    desc = 'line2'
    def __init__(self):
        Y0, Y1 = [0,25], [100, 25]
        self.trajs = [ddtf.TrajTwoLines()]
        self.extends = self.trajs[0].extends #(-10, 110, 0, 50) # _xmin, _xmax, _ymin, _ymax
        self.windfield = d2guid.WindField()
        self.time = np.arange(0, 12., 0.01)
        self.X0s = [[0, 10, 0, 0, 10]]
        Scenario.__init__(self)
register(ScenLine2)

class ScenCircle(Scenario):
    name = 'circle'
    desc = 'circle'
    def __init__(self, duration=None):
        Y0, Y1 = [0,25], [200, 25]
        self.trajs = [ddt.TrajectoryCircle(alpha0=3*np.pi/2)]
        self.extends = (-10, 75, -10, 75) # _xmin, _xmax, _ymin, _ymax
        #self.windfield = d2guid.WindField([0, 0])
        #self.windfield = d2guid.WindField([2.5, 0])
        self.windfield = d2guid.WindField([5, 0])
        duration = duration or 30.
        #self.time = np.arange(0, duration, 0.01)
        self.time = np.arange(0, self.trajs[0].duration, 0.01)
        if 1:
            self.X0s = [[20, -5, 0, np.deg2rad(18.), 5.]] # for the 5m/s windfield
        else:
            self.X0s=[]
            ac = d2dyn.Aircraft()
            for traj in self.trajs:
                t0 = self.time[0]
                Yr = traj.get(t0)
                W = self.windfield.sample(t0, Yr[0])
                self.X0s.append(d2guid.DiffFlatness.state_and_input_from_output(Yr, W, ac)[0])
            self.X0s[0][0] += 5.; self.X0s[0][1] += -5.
        self.perts = [np.zeros((len(self.time), d2dyn.Aircraft.s_size))]
        self.perts[0][500,d2dyn.Aircraft.s_x]  =  10
        self.perts[0][1000,d2dyn.Aircraft.s_y] = -10
        Scenario.__init__(self)
register(ScenCircle)

class ScenSquare:
    name = 'square'
    desc = 'square'
    def __init__(self):
        self.trajs = [ddtf.TrajSquare()]
        self.extends = self.trajs[0].extends
        self.windfield = d2guid.WindField()
        self.time = np.arange(0, 30., 0.01)
        self.X0s = [[0, 0, 0, 0, 10]]
register(ScenSquare)


class ScenMultiCircle(Scenario):
    name = "mucir"
    desc = "5 circles (30m radius, x offset)"
    def __init__(self, dx=0., dalpha=np.deg2rad(30.), nc=5, v=10.):
        self.trajs = [ddt.TrajectoryCircle(c=[40.+i*dx, 50.], alpha0=i*dalpha, v=v) for i in range(nc)]
        self.extends = (0, 80, 10, 90) # _xmin, _xmax, _ymin, _ymax
        #self.X0s = [[0, 0, 0, 0, 10], [0, 2, 0, 0, 10], [0, 4, 0, 0, 10], [0, 6, 0, 0, 10], [0, 8, 0, 0, 10]]
        #self.X0s = [[75, 50, np.pi/2, 0, 10], [85, 50, np.pi/2, 0, 10], [75, 60, np.pi/2, 0, 10]]
        self.X0s = [[75-5*i, 50+7.5*i, np.pi/2+i*np.deg2rad(15), 0, v] for i in range(nc)]
        self.X0s = [[75-5*i, 50+10*i, np.pi, 0, 10] for i in range(nc)]
        self.windfield = d2guid.WindField([5, 0])
        t0, t1, dt = 0, 20, 0.01
        self.time = np.arange(t0, t1, dt)
        Scenario.__init__(self)

register(ScenMultiCircle)

class ScenMultiCircle2(Scenario):
    name = "mucir2"
    desc = "2 circles (30m radius, x offset)"
    def __init__(self):
        traj1 = ddt.TrajectoryCircle(c=[40., 50.], alpha0=np.deg2rad(0.))
        traj2 = ddt.TrajectoryCircle(c=[50., 50.], alpha0=np.deg2rad(30.))
        self.trajs = [traj1, traj2]
        self.extends = (0, 100, 0, 100) # _xmin, _xmax, _ymin, _ymax
        self.X0s = [[75, 50, np.pi/2, 0, 10], [85, 70, np.pi/1.5, 0, 10]]
        self.windfield = d2guid.WindField([1., 0])
        t0, t1, dt = 0, 18, 0.01
        self.time = np.arange(t0, t1, dt)
        Scenario.__init__(self)
        
register(ScenMultiCircle2)




class ScenLinePatrol(Scenario):
    name = "linpat"
    desc = "The original 'Line Patrol' scenario"
    def __init__(self):
        traj1 = ddtf.TrajLineWithIntro(Y0=[0., 100.], Y1=[0., 50.], Y2=[200., 50.], r=25.)
        
        traj2 = ddtf.TrajLineWithIntro(Y0=[0., 0.], Y1=[0., 50.], Y2=[200., 50.], r=-25.)

        self.trajs = [traj1, traj2]
        self.X0s = [[0, 100, -np.pi, 0, 10], [0, 0, -np.pi, 0, 10]]
        self.extends = (-30, 110, -10, 110) # _xmin, _xmax, _ymin, _ymax

        self.windfield = d2guid.WindField([0, 2.5])
        
        t0, t1, dt = 0, 20, 0.01
        self.time = np.arange(t0, t1, dt)
        Scenario.__init__(self)
        
register(ScenLinePatrol)

class ScenLinePatrol2(Scenario):
    name = "linpat2"
    desc = "The original 'Line Patrol' scenario 2"
    def __init__(self):
        traj1 = ddtf.TrajLineWithIntro(Y0=[0., 100.], Y1=[0., 50.], Y2=[100., 50.], r=25.)
        
        traj21 = ddtf.TrajSlalom(p1=[0,50], p2=[100,50], v=10.)
        traj2 = ddtf.TrajWithIntro([-20,0], traj21, duration=8.)

        traj31 = ddtf.TrajSlalom(p1=[0,40], p2=[100,40], v=10.)
        traj3 = ddtf.TrajWithIntro([0,0], traj31, duration=8.)

        self.trajs = [traj1, traj2, traj3]
        self.X0s = [[0, 100, -np.pi, 0, 10], [-15, 0, np.pi/2, 0, 10], [5, 0, np.pi/2, 0, 10]]
        self.extends = (-30, 110, -10, 110) # _xmin, _xmax, _ymin, _ymax

        self.windfield = d2guid.WindField([0, 2.5])
        self.time = np.arange(0., 17.5, _default_dt)
        Scenario.__init__(self)
register(ScenLinePatrol2)


class ScenOval:
    name = "oval"
    desc = "oval"
    def __init__(self):
        self.trajs = [ddtf.TrajLineWithIntro(Y0=[0., 100.], Y1=[0., 50.], Y2=[200., 50.], r=25.)]
        self.X0s = [[0, 100, -np.pi, 0, 10], [0, 0, -np.pi, 0, 10]]
        self.extends = (-30, 110, -10, 110) # _xmin, _xmax, _ymin, _ymax
        self.windfield = d2guid.WindField([0, 2.5])
        t0, t1, dt = 0, 20, 0.01
        self.time = np.arange(t0, t1, dt)
        
register(ScenOval)


def print_available():
    print('Available scenarios:')
    for i, n in enumerate(list_available()): print(f'{i} -> {n}')

def get(_name):
    return _scenarios[_name][1](), _scenarios[_name][0]

