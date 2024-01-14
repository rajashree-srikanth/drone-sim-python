import numpy as np

import d2d.trajectory as ddt


#
#
# We define a set of commonly used trajectories
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
        ddt.TrajectoryCircle.__init__(self, c=[30., 30.],  r=30., v=10.)
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
    def __init__(self, Y0=[0, 0], Y1=[0, 50], Y2=[200, 50]):
        Yc = (np.asarray(Y0)+Y1)/2
        s1 = ddt.TrajectoryCircle(c=Yc,  r=25., v=10., alpha0=np.pi/2, dalpha = np.pi)
        s2 = ddt.TrajectoryLine(Y1, Y2, v=10., t0=s1.duration)
        ddt.CompositeTraj.__init__(self, [s1, s2])
        self.extends = (-30, 230, -30, 130)  # _xmin, _xmax, _ymin, _ymax
register(TrajLineWithIntro) 

# to remove
def traj_factory(i):
    extends = (0, 100, 0, 100)  # _xmin, _xmax, _ymin, _ymax 
    if i==0:
        traj = ddt.TrajectoryCircle()
        extends = (-30, 100, -60, 100)  # _xmin, _xmax, _ymin, _ymax 
    elif i==1:
        traj = ddt.TrajectoryLine([0,0], [400, 400], v=10.)
        extends = (-30, 430, -30, 430)  # _xmin, _xmax, _ymin, _ymax 
    elif i==2:
        Y0 = [[0, 10, 0, 0], [0, 0, 0, 0]]
        Y1 = [[0, 0, 0, 0], [500, 10, 0, 0]]
        traj = ddt.MinSnapPoly(Y0,  Y1, duration=60.)
        extends = (-30, 430, -30, 430)  # _xmin, _xmax, _ymin, _ymax 
    elif i==3:
        Y0 = [0, 0]
        Y1 = [0, 200]
        traj = ddt.MinSnapPoly(Y0,  Y1, duration=60.)
        extends = (-30, 430, -30, 430)  # _xmin, _xmax, _ymin, _ymax 
    elif i==4:
        traj = ddt.TrajectoryLine([0,0], [400, 400], v=10.)
        extends = (-30, 430, -30, 430)  # _xmin, _xmax, _ymin, _ymax
    else:
        pass
    return traj, extends




def print_available():
    print('Available trajectories:')
    for i, n in enumerate(list_available()):
        print(f'{i} -> {n}')

def get(traj_name):
    return trajectories[traj_name][1](), trajectories[traj_name][0]

