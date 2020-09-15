# Summary: Some Polar Plots
#          Python 3.7
#
# Notes: Written on 2020-07-10
#        Extra context: This represents attempts to solve the differential equations
#                       from the Euler-Lagrange, for w(theta) = sin(theta)


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

from scipy.integrate import solve_ivp
from r_theta import *

def r_optimal(w, r0):
    '''Solves the differential equation to find the optimal r(theta) for a certain w(theta).
    Args:
      w: Function (float)->float, weight function
      r0: 2-item numpy array, initial conditions r(0) and r'(0) for r(theta)
    Returns:
      radius: numpy array of floats, radius output
    '''
    def isoperim_euler_lagrange(t, r):
        return [r[1], 
                r[0] + 2*r[1]**2/r[0] - w(t)*(r[0]**2 + r[1]**2)**(3/2) / r[0] ]
    
    theta = np.arange(0.001, np.pi-0.001, 0.001)

    solution = solve_ivp(isoperim_euler_lagrange, (0.001,np.pi-0.001), r0, tspan=theta)
    plt.polar(solution.y[0], solution.t)
    #plt.polar(solution.y[1], solution.t)
    plt.show()

if __name__ == '__main__':
    w = np.sin
    r_optimal(w, [1, 1])
