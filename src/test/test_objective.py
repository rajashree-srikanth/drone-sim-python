#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, argparse
import numpy as np, sympy as sym
import matplotlib.pyplot as plt
import d2d.ploting as d2plot
import d2d.opty_utils as d2optu
import d2d.multiopty_utils as d2moptu

class TestPlanner:
    num_nodes = 10
    _nstate = 2
    _slice_x = slice(0*num_nodes, 1*num_nodes, 1)
    _slice_y = slice(1*num_nodes, 2*num_nodes, 1)
    obj_scale = 1.

def cost1(x, y, c, r, k=2.):
    dx, dy = x-c[0], y-c[1]
    d2 = np.square(dx/r*k) + np.square(dy/r*k)
    #e = np.exp(r**2-d2)
    e = np.exp(-d2)
    e = np.clip(e, 0., 1e3)
    grad = np.zeros_like(x)
    grad = np.vstack([-2*dx/r*k*e, -2*dy/r*k*e])
    #breakpoint()
    return e, grad


def plot_vs_dist(dist, cost, grad, _f=None, _a=None):
    if _f is None: fig, axes = plt.subplots(2, 1)
    else: fig, axes = _f, _a
    axes[0].plot(dist, cost)
    d2plot.decorate(axes[0], 'Cost', 'dist in m')
    axes[1].plot(dist, grad)
    d2plot.decorate(axes[1], 'Grad', 'dist in m')
    return _f, _a

def test_1():
    c, r = [0, 0], 10.
    x = np.arange(0, 2*r, 0.01)
    y = np.zeros(len(x))
    dx, dy = x-c[0], y-c[1]
    dist = np.sqrt(np.square(dx) + np.square(dy))
    cost, grad = cost1(x, y, c, r)
    #breakpoint()
    plot_vs_dist(dist, cost, grad)
    

def test_2():
    c, r = [0, 0], 10.
    cf = d2optu.CostObstacle(c, r)
        
    _p = TestPlanner()
    free = np.zeros(_p.num_nodes*_p._nstate)
    free[_p._slice_x] = np.linspace(0, 2*r, _p.num_nodes)
    cost = cf.cost1(free, _p)
    grad = cf.cost_grad(free, _p)
    #breakpoint()
    dist = np.sqrt(np.square(free[_p._slice_x]-c[0]) + np.square(free[_p._slice_y]-c[1]))
    plot_vs_dist(dist, cost, grad[_p._slice_x])


def dyn(X, t, gamma=2):
    return np.array([X[0]*(1.-X[1]), X[1]*(-gamma+X[0])])
class TestPlanner1:
    num_nodes = 1
    _nstate = 2
    _slice_x = slice(0*num_nodes, 1*num_nodes, 1)
    _slice_y = slice(1*num_nodes, 2*num_nodes, 1)
    obj_scale = 1.

def test_3():
    c, r = [25, -10], 10.
    _nx, _ny = 40, 30
    x0, x1, y0, y1 = 0, 60, -40, 20
    x, y = np.linspace(x0, x1, _nx), np.linspace(y0, y1, _ny)

    if 0:
        x,y = np.meshgrid(x,y)
        X,Y = dyn([x,y], 0)
        # normalisation
        M = (np.hypot(X,Y))
        M[M==0]=1.
        X,Y = X/M, Y/M
        V = M
        #breakpoint()
    else:
        X, Y = np.zeros((_ny, _nx)), np.zeros((_ny, _nx)) 
        V = np.zeros((_ny, _nx))
        for i, _x in enumerate(x):
            for j, _y in enumerate(y):
                _e, _g = cost1(_x, _y, c, r, k=2.)
                V[j, i] = _e
                X[j, i], Y[j, i] = _g
        M = (np.hypot(X,Y))
        M[M==0]=1.
        X,Y = X/M, Y/M
        
        
    ax = plt.gca()
    if 0:
        ax.quiver(x,y,X,Y,V,pivot='mid',cmap=plt.cm.jet)
    else:
        ax.contourf(x, y, V,cmap=plt.cm.jet, alpha=0.2)
    ax.xaxis.set_label_text('$x$', fontsize=20)
    ax.yaxis.set_label_text('$y$', fontsize=20)
    ax.set_aspect('equal', adjustable='box')

    #for i,X in enumerate(Xs):
    #    plt.plot(X[:-100,0], X[:-100,1], linewidth=2, label=f'$X_{i}$')
    plt.legend()

class TestPlanner2:
    num_nodes = 1
    _nstate = 2
    _slice_x = slice(0*num_nodes, 1*num_nodes, 1)
    _slice_y = slice(1*num_nodes, 2*num_nodes, 1)
    _ninp = 2
    _slice_v   = slice(2*num_nodes, 3*num_nodes, 1)
    _slice_phi = slice(3*num_nodes, 4*num_nodes, 1)
    obj_scale = 1.

def plot2d(cf, _p, x0=0, x1=60, y0=-40, y1=20, dx=1., dy=1., sty='c'):
    c, r = [25, -10], 10.
    x, y = np.arange(x0, x1, dx), np.arange(y0, y1, dy)
    nx, ny = len(x), len(y)
    X, Y, V = [np.zeros((ny, nx)) for _ in range(3)]
    free = np.zeros(_p.num_nodes*_p._nstate)
    if type(_p._slice_x) == list: _sx, _sy = _p._slice_x[0], _p._slice_y[0]
    else:_sx, _sy = _p._slice_x, _p._slice_y
    for i, _x in enumerate(x):
        for j, _y in enumerate(y):
            free[_sx], free[_sy] = _x, _y
            V[j, i] = cf.cost(free, _p)
            X[j, i], Y[j, i] = cf.cost_grad(free, _p)

    ax = plt.gca()
    ax.axis('equal')
    if sty == 'q':
        M = (np.hypot(X,Y));M[M==0]=1.;X,Y = X/M, Y/M
        ax.quiver(x,y,X,Y,V,pivot='mid',cmap=plt.cm.jet)
    else:
        ax.contourf(x, y, V,cmap=plt.cm.jet, alpha=0.2)


def plot1d(cf, _p, x0=0, x1=60, y0=-40, y1=20, dx=1., dy=1., _f=None, _a=None):
    x, y = np.arange(x0, x1, dx), np.arange(y0, y1, dy)
    
    if _f is None: fig, axes = plt.subplots(4, 1)
    else: fig, axes = _f, _a

    xc, yc, vc, phic = 25, -10, 10, 0
    
    e = [cf.cost(np.array([_x, yc, vc, phic]), _p) for _x in x]
    axes[0].plot(x, e)
    d2plot.decorate(axes[0], f'x (y={yc})')

    e = [cf.cost(np.array([xc, _y]), _p) for _y in y]
    axes[1].plot(y, e)
    d2plot.decorate(axes[1], f'y (x={xc})')

    v0, v1, dv = 8., 16., 0.1
    v = np.arange(v0, v1, dv)
    e = [cf.cost(np.array([xc, yc, _v, phic]), _p) for _v in v]
    axes[2].plot(v, e)
    d2plot.decorate(axes[2], 'v', 'm/s')

    phi0, phi1, dphi = -np.deg2rad(40), np.deg2rad(40), np.deg2rad(0.1)
    phi = np.arange(phi0, phi1, dphi)
    e = [cf.cost(np.array([xc, yc, vc, _phi]), _p) for _phi in phi]
    axes[3].plot(np.rad2deg(phi), e)
    d2plot.decorate(axes[3], 'phi', 'deg')

    
    #d2plot.decorate(axes[0], 'Cost', 'dist in m')
    #axes[1].plot(dist, grad)
    #d2plot.decorate(axes[1], 'Grad', 'dist in m')
    return _f, _a



class TestPlanner3:
    num_nodes = 1
    _nstate = 2
    _slice_x = [slice(0*num_nodes, 1*num_nodes, 1)]
    _slice_y = [slice(1*num_nodes, 2*num_nodes, 1)]
    _ninp = 2
    _slice_v   = [slice(2*num_nodes, 3*num_nodes, 1)]
    _slice_phi = [slice(3*num_nodes, 4*num_nodes, 1)]
    obj_scale = 1.
    acs = d2moptu.AircraftSet(1)


def test_4(which=0):

    _p = TestPlanner2()
    if which==0:   # single obstacle
        c, r = [25, -10], 10.
        cf = d2optu.CostObstacle(c, r, kind=1)

    elif which==1: # obstacles
        obstacles = ((25, -10, 10), (25, -35, 5), (40, -30, 5))
        cf = d2optu.CostObstacles(obstacles)
    elif which==2: # composit
        obstacles = ((25, -10, 10), (25, -35, 5), (40, -30, 5))
        cf = d2optu.CostComposit(obstacles, vsp=10., kobs=1., kvel=1., kbank=1.)
    elif which==3:
        _p = TestPlanner3()
        #cf = d2moptu.CostNull()
        #cf = d2moptu.CostAirvel(vsp=12.)
        #cf = d2moptu.CostBank()
        cf = d2moptu.CostInput(vsp=10., kv=1., kphi=1.)
    elif which==4:
        _p = TestPlanner3()
        c, r = [25, -10], 10.
        cf = d2moptu.CostObstacle(c, r, kind=1)
        #CostInputPlusObstacles
        
    plot2d(cf, _p, x0=0, x1=60, y0=-40, y1=20, dx=1., dy=1.)
    
    
    plot1d(cf, _p, _f=None, _a=None)


    
    
def main():
    #test_1()
    #test_2()
    #test_3()
    test_4()
    plt.show()

if __name__ == '__main__':
    main()
    
