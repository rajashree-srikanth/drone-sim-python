import numpy as np

import d2d.trajectory as ddt
import d2d.trajectory_factory as ddtf
import d2d.guidance as dg

#
#
# We define a set of commonly used scenarios (vehicles, trajectories, wind, etc)
#
#

_scenarios = {}
def register(S): _scenarios[S.name] = (S.desc, S)
def list_available():
    return ['{}: {}'.format(k,v[0]) for k,v in sorted(_scenarios.items())]


class ScenLine:
    name = 'line'
    desc = 'line'
    def __init__(self):
        Y0, Y1 = [0,25], [100, 25]
        self.trajs = [ddt.TrajectoryLine(Y0, Y1, v=10., t0=0.)]
        self.extends = (-10, 110, 0, 50) # _xmin, _xmax, _ymin, _ymax
        self.windfield = dg.WindField()
        self.time = np.arange(0, 12., 0.01)
        self.X0s = [[0, 10, 0, 0, 10]]
register(ScenLine)

class ScenLine2:
    name = 'line2'
    desc = 'line2'
    def __init__(self):
        Y0, Y1 = [0,25], [100, 25]
        self.trajs = [ddtf.TrajTwoLines()]
        self.extends = self.trajs[0].extends #(-10, 110, 0, 50) # _xmin, _xmax, _ymin, _ymax
        self.windfield = dg.WindField()
        self.time = np.arange(0, 12., 0.01)
        self.X0s = [[0, 10, 0, 0, 10]]
register(ScenLine2)

class ScenCircle:
    name = 'circle'
    desc = 'circle'
    def __init__(self):
        Y0, Y1 = [0,25], [200, 25]
        self.trajs = [ddt.TrajectoryCircle(alpha0=3*np.pi/2)]
        self.extends = (-10, 75, -10, 75) # _xmin, _xmax, _ymin, _ymax
        self.windfield = dg.WindField([0, 0])
        self.windfield = dg.WindField([2.5, 0])
        #self.windfield = dg.WindField([5, 0])
        self.time = np.arange(0, 30., 0.01)
        self.X0s = [[10, 0, 0, 0, 10.]]
register(ScenCircle)

class ScenSquare:
    name = 'square'
    desc = 'square'
    def __init__(self):
        self.trajs = [ddtf.TrajSquare()]
        self.extends = self.trajs[0].extends
        self.windfield = dg.WindField()
        self.time = np.arange(0, 30., 0.01)
        self.X0s = [[0, 0, 0, 0, 10]]
register(ScenSquare)


class ScenMultiCircle:
    name = "mucir"
    desc = "5 circles (30m radius, x offset)"
    def __init__(self):
        traj1 = ddt.TrajectoryCircle(c=[40., 50.], alpha0=np.deg2rad(0.))
        traj2 = ddt.TrajectoryCircle(c=[45., 50.], alpha0=np.deg2rad(30.))
        traj3 = ddt.TrajectoryCircle(c=[50., 50.], alpha0=np.deg2rad(60.))
        traj4 = ddt.TrajectoryCircle(c=[55., 50.], alpha0=np.deg2rad(90.))
        traj5 = ddt.TrajectoryCircle(c=[60., 50.], alpha0=np.deg2rad(120.))
        self.trajs = [traj1, traj2, traj3, traj4, traj5]
        self.extends = (0, 100, 0, 100) # _xmin, _xmax, _ymin, _ymax
        self.X0s = [[0, 0, 0, 0, 10], [0, 2, 0, 0, 10], [0, 4, 0, 0, 10], [0, 6, 0, 0, 10], [0, 8, 0, 0, 10]]
        self.windfield = dg.WindField()

        
        t0, t1, dt = 0, 60, 0.01
        self.time = np.arange(t0, t1, dt)
register(ScenMultiCircle)

class ScenMultiCircle2:
    name = "mucir2"
    desc = "2 circles (30m radius, x offset)"
    def __init__(self):
        traj1 = ddt.TrajectoryCircle(c=[40., 50.], alpha0=np.deg2rad(0.))
        traj2 = ddt.TrajectoryCircle(c=[50., 50.], alpha0=np.deg2rad(30.))
        self.trajs = [traj1, traj2]
        self.extends = (0, 100, 0, 100) # _xmin, _xmax, _ymin, _ymax
        self.X0s = [[75, 50, np.pi/2, 0, 10], [85, 70, np.pi/1.5, 0, 10]]
        self.windfield = dg.WindField()
        t0, t1, dt = 0, 18, 0.01
        self.time = np.arange(t0, t1, dt)
register(ScenMultiCircle2)




class ScenLinePatrol:
    name = "linpat"
    desc = "Line Patrol"
    def __init__(self):
        traj1 = ddtf.TrajLineWithIntro(Y0=[0., 100.], Y1=[0., 50.], Y2=[200., 50.], r=25.)
        
        traj2 = ddtf.TrajLineWithIntro(Y0=[0., 0.], Y1=[0., 50.], Y2=[200., 50.], r=-25.)

        self.trajs = [traj1, traj2]
        self.X0s = [[0, 100, -np.pi, 0, 10], [0, 0, -np.pi, 0, 10]]
        self.extends = (-30, 110, -10, 110) # _xmin, _xmax, _ymin, _ymax

        self.windfield = dg.WindField([0, 2.5])
        
        t0, t1, dt = 0, 20, 0.01
        self.time = np.arange(t0, t1, dt)
register(ScenLinePatrol)

def print_available():
    print('Available scenarios:')
    for i, n in enumerate(list_available()): print(f'{i} -> {n}')

def get(_name):
    return _scenarios[_name][1](), _scenarios[_name][0]

