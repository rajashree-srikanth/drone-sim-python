#
# Scenarios for the mutiple vehicles planner
#

import sys, os, argparse
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import collections

import opty.direct_collocation

import d2d.ploting as d2p
import d2d.opty_utils as d2ou
import d2d.multiopty_utils as d2mou



class exp_0:  # single aircraft
    name = 'exp_0'
    desc = 'single aircraft'
    t0, t1, hz = 0., 10., 50.
    p0s = ((   0., 0., 0.,   0., 10.), )
    p1s = (( 100., 0,  0.,   0., 10.), )

    wind = d2ou.WindField()

    initial_guess = 'rnd'
    ###initial_guess = 'tri'
    
    tol, max_iter = 1e-5, 1000
    #cost, obj_scale = SetCostBank(), 1.
    #cost, obj_scale = SetCostAirvel(vsp=12.), 1.
    vref = 12.
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=5., kphi=1.), 1.e-1
    x_constraint, y_constraint = None, None
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    v_constraint = (9., 15.)
    obstacles = []
    ncases = 1
    def set_case(idx): pass
    def label(idx): return ''
   
class exp_0_1(exp_0):  # varying weights
    name = 'exp_0_1'
    desc = 'single aircraft, varying weights'
    t0, p0s =  0., (( 0.,  0., 0.,  0., 10.), )
    t1, p1s = 10., (( 100, 0,  0.,  0., 10.), )
    x_constraint, y_constraint = None, None
    #initial_guess = 'rnd'
    #Ks = [[1., 0.5], [1., 1.],[1., 10.],[1., 20.], [1., 30.], [1., 40.], [1., 50.]]
    Ks = [[1., 1.],[1., 20.], [1., 40.], [1., 60.]]
    ncases = len(Ks)
    def set_case(idx):
        exp_0_1.K = exp_0_1.Ks[idx]
        exp_0_1.cost = d2mou.CostInput(vsp=13., kv=exp_0_1.K[0], kphi=exp_0_1.K[1])
    def label(idx):  return f'kvel, kbank {exp_0_1.K}'



class exp_0_2(exp_0):  # varying durations
    name = 'exp_0_2'
    desc = 'single aircraft, duration'
    _dur = [7, 8, 9, 10, 11, 14, 16, 20]
    ncases = len(_dur)
    def set_case(idx): exp_0_2.t1 = exp_0_2._dur[idx]
    def label(idx):  return f'duration {exp_0_2._dur[idx]}'


class exp_0_3(exp_0):  # varying wind
    name = 'exp_0_3'
    desc = 'single aircraft, wind'
    _wind = [[-1, -1], [0, 0] , [1, 1], [3, 3]]
    ncases = len(_wind)
    def set_case(idx): exp_0_3.wind = d2ou.WindField(exp_0_3._wind[idx])
    def label(idx):  return f'wind {exp_0_3._wind[idx]}'
    
    
class exp_1(exp_0): # 2 aircraft face to face
    name = 'exp_1'
    desc = '2 aicraft face to face'
    t1 = 4.5
    vref = 12.
    dpsi = 0.01
    p0s = (( 0.,  0.,   0.,  0., 12.), ( 50.,  0.,  np.pi-dpsi, 0., 12.))
    p1s = (( 50., 0.,   0.,  0., 12.), (  0.,  0.,  np.pi+dpsi, 0., 12.))
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=5., kphi=1.), 1.e-1
    #cost, obj_scale = SetCostCollision(), 1.
    x_constraint, y_constraint = None, None
    obstacles = []
    #obstacles = ((25, -10, 5), )
    initial_guess = 'rnd'

class exp_1_0(exp_1): # 2 aircraft meeting
    name = 'exp_1_0'
    desc = '2 aicraft meeting'
    t1 = 4.5
    vref = 12.
    p0s = ((  0., -20., np.pi/2,  0., 12.), ( 7.5, -20.,  np.pi/2,  0., 12.))
    p1s = (( 40.,   5.,      0.,  0., 12.), ( 40.,   10.,  0,        0., 12.))
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=1., kphi=1.), 1.e-1
    x_constraint, y_constraint = None, None

class exp_1_1(exp_1): # face to face, wind
    name = 'exp_1_1'
    desc = '2 aicraft face to face, wind'
    #initial_guess = 'rnd'
    initial_guess = 'tri'


    
class exp_2(exp_0): # four aircaft in cross
    name = 'exp_2'
    desc = '4 aicraft'
    t1 = 5.5
    vref = 12.
    overtime=1.5#1.75
    d = t1*vref/2 / overtime
    p0s = ((-d, 0., 0., 0., vref), ( d,  0.,  np.pi, 0., vref), ( 0.,  d, -np.pi/2,  0., vref), ( 0., -d,  np.pi/2, 0., vref))
    p1s = (( d, 0., 0., 0., vref), (-d,  0.,  np.pi, 0., vref), ( 0., -d, -np.pi/2,  0., vref), ( 0.,  d,  np.pi/2, 0., vref))
    cost, obj_scale = d2mou.CostInput(vsp=vref, kv=1., kphi=1.), 1.

class exp_2_1(exp_2):
    pass
    

class exp_3(exp_0): # testing obstacles
    name = 'exp_3'
    desc = 'single obstacle'
    t1 = 6.5 #7.5
    vref = 12.
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( 50., 0.,   0.,  0., 10.), )

    obstacles = ((25, -20, 10), )
    #cost, obj_scale = SetCostInput(vsp=vref, kv=5., kphi=1.), 1.e-1
    cx, cy, r = obstacles[0]
    cost, obj_scale = d2mou.CostObstacle(c=(cx,cy), r=r, kind=0), 1.
    x_constraint, y_constraint = None, None
    #x_constraint, y_constraint = (-100, 100), (-100, 100)
    #v_constraint = (11.99, 12.01)
    v_constraint = (8., 18.)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))


class exp_3_1(exp_3): # testing obstacles
    name = 'exp_3_1'
    desc = 'single obstacle, size/location'
    obstacles = ((25, -20, 10), (25, -10, 10))

    ncases = len(obstacles)
    def set_case(idx):
        cx, cy, r = exp_3_1.obstacles[idx] 
        exp_3_1.cost = d2mou.CostObstacle(c=(cx,cy), r=r, kind=0)
    def label(idx):  return f'obstacle {exp_3_1.obstacles[idx]}'

class exp_4(exp_0): # testing obstacles
    name = 'exp_4'
    desc = 'set of obstacle'
    t1 = 10.5
    vref = 12.
    p0s = (( 0.,  0.,   0.,  0., 10.), )
    p1s = (( 100., 0.,   0.,  0., 10.), )

    obstacles = ((30, -10, 20), (70, 15, 20), )
    #cost, obj_scale = d2mou.CostInput(vsp=vref, kv=1., kphi=1.), 1.
    #cost, obj_scale = d2mou.CostObstacles(obstacles, kind=1), 1.
    cost, obj_scale = d2mou.CostComposit(kvel=1., kbank=1., kobs=1., kcol=float('NaN'), vsp=vref, obss=obstacles, obs_kind=1, rcol=3.), 1.
    x_constraint, y_constraint = None, None
    #x_constraint, y_constraint = (-5., 105.), (-20., 80.)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    #v_constraint = (11.99, 12.01)
    v_constraint = (9., 18.)
    initial_guess = 'rnd'
    #initial_guess = 'tri'
    
class exp_4_1(exp_4): # testing obstacles
    name = 'exp_4_1'
    desc = 'set of obstacles, size'
    v_constraint = (9., 15.)
    _obstacles = (((30, -10, 15), (30, 25, 15)),
                  ((50, -10, 15), (50, 25, 15)),
                  ((70, -10, 15), (70, 25, 15)))
    #obstacles = _obstacles[0]
    #cost, obj_scale = d2mou.CostObstacles(obstacles, kind=1), 1.
    #cost, obj_scale = d2mou.CostComposit(obstacles, vsp=10., kobs=1., kvel=1., kbank=1., obs_kind=1), 1.
    obj_scale = 1e-2
    ncases = len(_obstacles)
    def set_case(idx):
        exp_4_1.obstacles = exp_4_1._obstacles[idx] 
        #exp_4_1.cost = d2mou.CostObstacles(exp_4_1.obstacles, kind=1)
        exp_4_1.cost, exp_4_1.obj_scale = d2mou.CostComposit(kvel=1., kbank=1., kobs=1., kcol=float('NaN'), vsp=14., obss=exp_4_1.obstacles, obs_kind=1, rcol=3.), 1.
        #d2mou.CostComposit(exp_4_1.obstacles, vsp=14., kobs=1., kvel=1., kbank=.1, obs_kind=0)
    def label(idx):  return f'obstacles {exp_4_1._obstacles[idx]}'


class exp_4_2(exp_4): # testing obstacles
    name = 'exp_4_2'
    desc = 'set of obstacles, duration'
    _durations = [9, 10, 11, 12]
    ncases = len(_durations)
    def set_case(idx):
        exp_4_2.t1 = exp_4_2._durations [idx]
    def label(idx):  return f'duration {exp_4_2._durations[idx]} s'
    initial_guess = 'rnd'

    
class exp_5(exp_0): # testing collisions
    name = 'exp_5'
    desc = '2 aicraft face to face'
    t1 = 4.2
    vref = 12.
    dpsi = 0.
    p0s = (( 0.,  0.,   0.,  0., 12.), ( 50.,  0.,  np.pi-dpsi, 0., 12.))
    p1s = (( 50., 0.,   0.,  0., 12.), (  0.,  0.,  np.pi+dpsi, 0., 12.))
    x_constraint, y_constraint = None, None
    obstacles = []
    #initial_guess = 'rnd'
    initial_guess = 'tri'
    ncases = 1
    def set_case(idx):
        if idx==0:
            #exp_5.cost, exp_5.obj_scale = d2mou.CostInput(vsp=exp_5.vref, kv=70., kphi=1.), 1.e0
            exp_5.cost, exp_5.obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=float('NaN'), vsp=exp_5.vref, obss=[], obs_kind=0, rcol=3.), 1.e0
        else:
            #exp_5.cost, exp_5.obj_scale = d2mou.CostCollision(r=6, k=2), 1.e0
            exp_5.cost, exp_5.obj_scale = d2mou.CostComposit(kvel=70., kbank=1., kobs=float('NaN'), kcol=10., vsp=exp_5.vref, obss=[], obs_kind=0, rcol=10.), 1.e0
            
    def label(idx):  return f'obj {["Ref", "AntiCol"][idx]}'
    
class exp_5_1(exp_5): # testing collisions
    name = 'exp_5_1'
    desc = '2 aicraft next to one another'
    t1 = 8.
    vref = 12.
    if 0:
        p0s = (( 0.,  0.,   0.,  0., 12.), ( 0.,  0.,  0, 0., 12.))
        p1s = (( 50., 50.,   np.pi/2,  0., 12.), (  50.,  50.,  np.pi/2, 0., 12.))
    else:
        p0s = (( 0.,  0.,   0.,  0., 12.), ( 0.,  5.,  0, 0., 12.))
        p1s = (( 50., 50.,   np.pi/2,  0., 12.), (  55.,  50.,  np.pi/2, 0., 12.))
    initial_guess = 'tri'
    #initial_guess = 'rnd'

class exp_5_2(exp_5): # testing collisions
    name = 'exp_5_2'
    desc = '4 aircraft facing one another'
    t1 = 100/12.#9.
    vref = 12.
    nac=6
    c, r , alpha0, dalpha = (0, 0), 50., 0., np.pi/3
    p0s = d2ou.circle_formation(nac, c, r, alpha0, dalpha, vref, dpsi=np.pi)
    p1s = d2ou.circle_formation(nac, c, r, alpha0+np.pi, dalpha, vref, dpsi=0)
    #p1s = d2ou.line_formation(nac, (50,50), 0, 5, 5, 12)
    initial_guess = 'tri'
    #initial_guess = 'rnd'


class exp_6(exp_0): # testing collisions
    name = 'exp_6'
    desc = '6 aicraft on circle'
    nac = 12
    alpha0, dalpha = np.deg2rad(0), np.deg2rad(20)
    alphas= alpha0 + np.arange(0., nac*dalpha, dalpha)
    c, r = np.array([0,0]), 20.
    #def _p(_a): return exp_5_2.c+exp_5_2.r*np.array([np.cos(_a), np.sin(_a)])
    #print(c)
    #c = [0, 0]
    #p0s = [c for _a in alphas]
    p0s = []
    for _a in alphas:
        p0s.append((c[0]+r*np.cos(_a), c[1]+r*np.sin(_a), _a+np.pi/2, 0., 10))

    p1s = [(0, 50+5*i, np.pi, 0., 10.) for i in range(len(p0s))]
    t1 = 15.
    cost, obj_scale = d2mou.CostInput(vsp=12, kv=5., kphi=1.), 1.e0
    #x_constraint, y_constraint = None, None
    x_constraint, y_constraint = (0,100), (0,100)
    phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    v_constraint = (11.99, 12.01)
    #v_constraint = (9., 18.)
    #initial_guess = 'tri'
    initial_guess = 'rnd'




class exp_7(exp_0): # formation flight planing - comparison with single vehicle
    name = 'exp_7'
    desc = 'formation flight - random, line arrival'
    nac = 4
    vref = 12.
    seed = 6789; rng = np.random.default_rng(seed)
    p0s = d2ou.random_states(rng, nac, xlim=(0, 50), ylim=(-50, 0), v=vref)
    p1s = d2ou.line_formation(nac, (50, 50), np.pi, dx=7.5, dy=7.5, v=vref)
    t1 = 20.
    #cost, obj_scale = d2mou.CostInput(vsp=12, kv=5., kphi=1.), 1.e0
    cost, obj_scale = d2mou.CostComposit(kvel=5., kbank=1., kobs=float('NaN'), kcol=10., vsp=vref, obss=[], obs_kind=0, rcol=7.5), 1.e0
    #x_constraint, y_constraint = None, None
    #x_constraint, y_constraint = (0,100), (0,100)
    #phi_constraint = (-np.deg2rad(40.), np.deg2rad(40.))
    #v_constraint = (11.99, 12.01)
    #v_constraint = (9., 18.)
    initial_guess = 'tri'
    #initial_guess = 'rnd'


class exp_7_1(exp_7): # formation flight planing - comparison with single vehicle
    name = 'exp_7_1'
    desc = 'formation flight - circle start, line arrival'
    c, r, alpha0, dalpha = np.array([0, -25]), 25., -np.pi, np.deg2rad(30.)
    p0s = d2ou.circle_formation(exp_7.nac, c, r, alpha0, dalpha, exp_7.vref)

class exp_7_2(exp_7): # formation flight planing - comparison with single vehicle
    name = 'exp_7_2'
    desc = 'formation flight - circle start, diamond arrival'
    c, r, alpha0, dalpha = np.array([0, -25]), 25., -np.pi, np.deg2rad(30.)
    p0s = d2ou.diamond_formation(exp_7.nac, (50, 50), np.pi, dx=5., dy=5., v=exp_7.vref)

class exp_7_3(exp_7): # formation flight planing - comparison with single vehicle
    name = 'exp_7_3'
    desc = 'formation flight - circle start, diamond arrival'
    nac = 8
    vref, t1 = 12., 20.
    c, r, alpha0, dalpha = np.array([0, -25]), 25., -np.pi, np.deg2rad(30.)
    p0s = d2ou.circle_formation(nac, c, r, -np.pi/2, dalpha, vref)
    p1s = d2ou.diamond_formation(nac, (50, 50), np.pi, dx=7.5, dy=5., v=vref)
    #cost, obj_scale = d2mou.CostInput(vsp=12, kv=5., kphi=1.), 1.e0
    cost, obj_scale = d2mou.CostComposit(kvel=5., kbank=1., kobs=float('NaN'), kcol=10., vsp=vref, obss=[], obs_kind=0, rcol=5.), 1.e0
    
scens = [exp_0, exp_0_1, exp_0_2, exp_0_3,
         exp_1, exp_1_0, exp_1_1, exp_2,
         exp_3, exp_3_1,
         exp_4, exp_4_1, exp_4_2,
         exp_5, exp_5_1, exp_5_2,
         exp_6,
         exp_7, exp_7_1, exp_7_2, exp_7_3]
def desc_all_scens():
    return '\n'.join([f'{i}: {s.name} {s.desc}' for i, s in enumerate(scens)])
def get_scen(idx): return scens[idx]

def info_scen(idx):
    res = ''
    res += f'{scens[idx].name} {scens[idx].desc}\n'
    res += f't0 {scens[idx].t0} t1 {scens[idx].t1}\n'
    res += f'p0s {scens[idx].p0s}\np1s {scens[idx].p1s}\n'
    res += f'x_constraint {scens[idx].x_constraint}\ny_constraint {scens[idx].y_constraint}\n'
    res += f'v_constraint {scens[idx].v_constraint}\nphi_constraint {np.rad2deg(scens[idx].phi_constraint)}\n'
    res += f'wind {scens[idx].wind}\n'
    res += f'initial guess {scens[idx].initial_guess}\n'
    return res
