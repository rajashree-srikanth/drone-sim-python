# trial stuff 
import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import opty.direct_collocation
import d2d.opty_utils as d2ou
# import d2d.optyplan_scenarios as d2oscen
import single_opt_planner
import timeit

# plotting infinity mathematically - NOT USEFUL CODE!!!!
def inf_plot():
    # Define the parametric equations for the infinity shape
    t = np.linspace(0, 2 * np.pi, 1000)
    x = 12.5 * np.cos(t) + 87.5
    y = 10 * np.sin(2 * t) + 30

    # Define the extreme end points
    extreme_points = [(75, 40), (100, 20)]

    # Plotting the lemniscate (infinity shape)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Infinity Shape', color='blue')
    plt.scatter(*zip(*extreme_points), color='red', zorder=5, label='Extreme End Points')

    # Mark the extreme points with coordinates
    for point in extreme_points:
        plt.text(point[0], point[1], f'{point}', fontsize=10, ha='right')

    plt.title('Visualization of Infinity Shape with Specified Endpoints')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.show()

# attempting to generate entire trajectory from just one data
def opt_gen():
    start_time_real = timeit.default_timer()
    X1_f = np.array([[-49.98, -58.14, 2.22, -0.35, 15.],[-24.69, -58.4, 2.23, -0.35, 15.  ], 
            [-24.27, -138.77, 2.24, -0.35, 15.],[-48.98, -139.03, 2.24, -0.35, 15.]])
    scen = single_opt_planner.exp_1
    scen.t1 = 12
    scen.p0 = tuple(X1_f[0])# this works only of X1_f is an array!
    print(f'Scenario: {scen.name} - {scen.desc}')
    p = single_opt_planner.Planner(scen)
    print("planner initialized...")
    p.configure(tol=1e-5, max_iter=1500)
    p.run(initial_guess=p.get_initial_guess())
    # print(p.sol_x)
    elapsed = timeit.default_timer()-start_time_real
    print('Planner ran! in', elapsed, "seconds")
    f1, a1 = d2ou.plot2d(p, None)
    f2, a2 = d2ou.plot_chrono(p, None)
    # plt.show()
    
    # trying to generate other trajs using this
    delta = X1_f[1:, 0:2] - X1_f[0, 0:2]
    print(delta)
    delta = delta.T # rows - x, y; columns - a/c number
    x = p.sol_x.reshape(-1,1) # reshaping 1d to 2d array is mandatory before appending
    y = p.sol_y.reshape(-1,1)
    x = np.append(x, x+delta[0], axis=1)
    y = np.append(y, y+delta[1], axis=1)
    # breakpoint()
    plt.figure()
    for i in range(len(x[0,:])):
        plt.plot(x[:,i], y[:,i], label=f'aircraft {i+1}')
    plt.legend()
    plt.figure()
    for i in range(len(x[0,:])):
        plt.plot(p.sol_time, x[:,i],label=f'aircraft {i+1}')
    plt.title("x position with time")
    plt.legend()
    plt.figure()
    for i in range(len(x[0,:])):
        plt.plot(p.sol_time, y[:,i],label=f'aircraft {i+1}')
    plt.title("y position with time")
    plt.legend()
    plt.show()
opt_gen()