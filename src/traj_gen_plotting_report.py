import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import opty.direct_collocation
import d2d.opty_utils as d2ou
# import d2d.optyplan_scenarios as d2oscen
import single_opt_planner
import multi_opt_planner
import timeit

def traj_gen_infty():
    scen=multi_opt_planner.inf_traj_4ac
    f1, a1, f2, a2 = None, None, None, None
    for _case in range(scen.ncases):
        scen.set_case(_case)
        _p = multi_opt_planner.Planner(scen, initialize=True)
        initial_guess = None
        initial_guess = _p.get_initial_guess(scen.initial_guess)
        _p.run(initial_guess=initial_guess, tol=scen.tol, max_iter=scen.max_iter)
        _p.interpret_solution()
        f1, a1 = multi_opt_planner.plot2d(_p, f1, a1, scen.label(_case))
        f2, a2 = multi_opt_planner.plot_chrono(_p, f2, a2)
    for _a in a2: _a.autoscale()
    plt.show()
    
def anticoll():
    scen=multi_opt_planner.exp_5
    f1, a1, f2, a2 = None, None, None, None
    for _case in range(scen.ncases):
        scen.set_case(_case)
        _p = multi_opt_planner.Planner(scen, initialize=True)
        initial_guess = 'rnd'
        initial_guess = _p.get_initial_guess(scen.initial_guess)
        _p.run(initial_guess=initial_guess, tol=scen.tol, max_iter=scen.max_iter)
        _p.interpret_solution()
        f1, a1 = multi_opt_planner.plot2d(_p, f1, a1, scen.label(_case))
        f2, a2 = multi_opt_planner.plot_chrono(_p, f2, a2)
    for _a in a2: _a.autoscale()
    plt.label(loc="upper right")
    plt.show()

def compare_single():
    scen=single_opt_planner.exp_2

    for _case in range(scen.ncases):
        scen.set_case(_case)
        p = single_opt_planner.Planner(scen, initialize=True)
        p.configure(tol=1e-5, max_iter=1500)
        initial_guess = p.get_initial_guess(scen.initial_guess)
        p.run(initial_guess)
        
        # f1, a1 = d2ou.plot2d(p, None)
        # f2, a2 = d2ou.plot_chrono(p, None)
        plt.figure(1)
        plt.plot(p.sol_x, p.sol_y, label=f'case {_case+1}: t = {scen.t1} s')
    # for _a in a2: _a.autoscale()
    plt.legend()
    plt.xlabel("$\\text{x (in m)}$")
    plt.ylabel("$\\text{y (in m)}$")
    plt.title("$\\text{Optimal Trajectory Generated}$")
    
    
    plt.figure(2)
    plt.plot(p.sol_time, np.rad2deg(p.sol_phi))
    plt.xlabel("time (in s)")
    plt.ylabel("Bank angle (in deg)")
    plt.title("Bank angle input required for executing trajectory")
    
    plt.grid(True)
    plt.show()
        


# traj_gen_infty()
compare_single()
# anticoll()