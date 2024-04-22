import numpy as np

import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.guidance as d2guid
import d2d.dynamic as d2dyn
from d2d.dynamic import Aircraft

#
#
# A set of commonly used scenarios (vehicles, trajectories, wind, etc)
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
        try: self.time
        except AttributeError:
            tf = np.max([traj.duration for traj in self.trajs]) # min?
            self.time = np.arange(0., tf, _default_dt)
        try: self.aircrafts
        except AttributeError: self.aircrafts = [Aircraft() for i in range(nv)]
        try: self.perts
        except AttributeError: self.perts=[np.zeros((len(self.time), Aircraft.s_size)) for i in range(nv)]
        try: self.windfield
        except AttributeError: self.windfield = d2guid.WindField()
        try: self.X0s
        except AttributeError:
            self.X0s = []
            for ac, traj in zip(self.aircrafts, self.trajs):
                t0 = self.time[0]; Yr = traj.get(t0)
                W = self.windfield.sample(t0, Yr[0])
                self.X0s.append(d2guid.DiffFlatness.state_and_input_from_output(Yr, W, ac)[0])
        try: self.extends
        except AttributeError: 
            self.extends = (0., 100., 0., 100.)
            self.autoscale()
        try: self.ppctl    # control specification - needs love
        except AttributeError:
            self.ppctl = False
            
    def autoscale(self):
        pmin, pmax = (np.float('inf'), np.float('inf')), (-np.float('inf'), -np.float('inf'))
        for traj in self.trajs:
            for t in self.time:
                Yr = traj.get(t)
                pmin, pmax = np.min([Yr[0], pmin], axis=0), np.max([Yr[0], pmax], axis=0)
        extends = pmax-pmin; margin = 0.05* extends; pmin -= margin; pmax += margin
        self.extends = (pmin[0], pmax[0], pmin[1], pmax[1])
        #print(pmin, pmax)
        
    def summarize(self):
        r = f'{len(self.trajs)} trajectories\n'
        r += f'duration: {self.time[-1]-self.time[0]:.2f}s\n'
        r += f'wind: {self.windfield.summarize()}\n'
        extends = ''.join([f'{_e:.1f} ' for _e in self.extends])
        r += f'extends: {extends}'
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
    def __init__(self, duration=None, cst_gvel=False):
        Y0, Y1 = [0,25], [200, 25]
        if cst_gvel:
            self.trajs = [ddt.TrajectoryCircle(alpha0=3*np.pi/2)]
        else:  # cst air vel:
            self.trajs = [ddtf.TrajSiSpline(duration=20.)]
        
        self.extends = (-10, 75, -10, 75) # _xmin, _xmax, _ymin, _ymax
        #self.windfield = d2guid.WindField([0, 0])
        #self.windfield = d2guid.WindField([2.5, 0])
        self.windfield = d2guid.WindField([5, 0])
        #duration = duration or 30.
        #duration = duration or 30.
        #self.time = np.arange(0, duration, 0.01)
        self.time = np.arange(0, self.trajs[0].duration, 0.01)

        #breakpoint()
        if 0:
            self.X0s = [[20, -5, 0, np.deg2rad(18.), 5.]] # for the 5m/s windfield
        if 0:
            self.X0s=[]
            ac = d2dyn.Aircraft()
            for traj in self.trajs:
                t0 = self.time[0]
                Yr = traj.get(t0)
                W = self.windfield.sample(t0, Yr[0])
                self.X0s.append(d2guid.DiffFlatness.state_and_input_from_output(Yr, W, ac)[0])
            self.X0s[0][0] += 5.; self.X0s[0][1] += -5.
        if 0:
            self.perts = [np.zeros((len(self.time), d2dyn.Aircraft.s_size))]
            self.perts[0][500,d2dyn.Aircraft.s_x]  =  10
            self.perts[0][1000,d2dyn.Aircraft.s_y] = -10
        Scenario.__init__(self)
register(ScenCircle)

class ScenSquare(Scenario):
    name = 'square'
    desc = 'square'
    def __init__(self):
        self.trajs = [ddtf.TrajSquare()]
        self.extends = self.trajs[0].extends
        self.windfield = d2guid.WindField()
        self.time = np.arange(0, 30., 0.01)
        self.X0s = [[0, 0, 0, 0, 10]]
        Scenario.__init__(self)
register(ScenSquare)


class ScenMultiCircle(Scenario):
    name = "mucir"
    desc = "5 circles (30m radius, x offset)"
    def __init__(self, dx=0., dalpha=np.deg2rad(30.), nc=5, v=10.):
        self.trajs = [ddt.TrajectoryCircle(c=[40.+i*dx, 50.], alpha0=i*dalpha, v=v) for i in range(nc)]
        self.extends = (0, 80, 10, 90) # _xmin, _xmax, _ymin, _ymax
        #self.X0s = [[0, 0, 0, 0, 10], [0, 2, 0, 0, 10], [0, 4, 0, 0, 10], [0, 6, 0, 0, 10], [0, 8, 0, 0, 10]]
        #self.X0s = [[75, 50, np.pi/2, 0, 10], [85, 50, np.pi/2, 0, 10], [75, 60, np.pi/2, 0, 10]]
        #self.X0s = [[75-5*i, 50+7.5*i, np.pi/2+i*np.deg2rad(15), 0, v] for i in range(nc)]
        self.X0s = [[75-5*i, 60+5*i, np.pi, 0, 10] for i in range(nc)]
        self.windfield = d2guid.WindField([5, 0])
        t0, t1, dt = 0, 20, 0.01
        self.time = np.arange(t0, t1, dt)
        Scenario.__init__(self)

register(ScenMultiCircle)

class ScenMultiCircle2(Scenario):
    name = "mucir2"
    desc = "2 circles (30m radius, x offset)"
    def __init__(self, dx=0):
        traj1 = ddt.TrajectoryCircle(c=[40., 50.], alpha0=np.deg2rad(0.))
        traj2 = ddt.TrajectoryCircle(c=[40.+dx, 50.], alpha0=np.deg2rad(30.))
        self.trajs = [traj1, traj2]
        self.extends = (0, 100, 0, 100) # _xmin, _xmax, _ymin, _ymax
        self.X0s = [[75, 50, np.pi/2, 0, 10], [85, 70, np.pi/1.5, 0, 10]]
        self.windfield = d2guid.WindField([1., 0])
        t0, t1, dt = 0, 18, 0.01
        self.time = np.arange(t0, t1, dt)
        Scenario.__init__(self)
        
register(ScenMultiCircle2)




class ScenPatrol(Scenario):
    name = "patrol"
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
        
register(ScenPatrol)

class ScenPatrol2(Scenario):
    name = "patrol_2"
    desc = "dev patrol"
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
register(ScenPatrol2)

class ScenPatrol3(Scenario):
    name = "patrol_3"
    desc = "dev patrol"
    def __init__(self, nv=2):
        self.trajs=[]   
        for i in range(nv):
            dy = 5*i; dx=dy/2
            Y0, Y1 = [0,10+dy], [100-dx, 10+dy]
            l1 = ddt.TrajectoryLine(Y0, Y1, v=10., t0=0.)
            c1 = ddt.TrajectoryCircle(c=[100-dx, 40],  r=30.-dy, v=10., t0=0., alpha0=-np.pi/2, dalpha=np.pi)
            if 1:
                s2 = ddtf.TrajSlalom(p1=[100,60-dy], p2=[0,60-dy], v=10., t0=0., phi=np.pi/2)
                self.trajs.append(ddt.CompositeTraj([l1, c1, s2]))
            else:
                self.trajs.append(ddt.CompositeTraj([l1, c1]))
        #print(self.trajs[0].duration)    
        #self.time = np.arange(0., 28., _default_dt)
        self.windfield = d2guid.WindField([0, 5.])
        Scenario.__init__(self)

register(ScenPatrol3)



class ScenCircularFormation(Scenario):
    name = "circForm"
    desc = "circular formation"
    def __init__(self):
        P0s = [[30,10], [40, 10]]
        self.trajs = [ddt.TrajectoryCircle(alpha0=3*np.pi/2+i*np.pi/6) for i, P0 in enumerate(P0s)]
        Scenario.__init__(self)
        for P0, X0 in zip(P0s, self.X0s):
            X0[:Aircraft.s_y+1] = P0
        self.ppctl = False#True
        self.windfield = d2guid.WindField([0, 5.])
    
register(ScenCircularFormation)
    


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

