'''This file is an attempt to re-implement the MIRK (I assume
M-something Implicit Runge Kutta) solver from DifferentialEquatons.jl.
It is a collocation method using 4th order implicit Runge Kutta and
a trust region dogleg method.'''

'''Currently either not fully debugged or running way to slow to be usable.
I will revisit it later.'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from numdifftools import Jacobian, Hessian
import time
from math import ceil

from r_theta import *

def el_diffq(w, mu=1):
    '''Generates and returns the expanded
    Euler-Lagrange differential
    equation for our weight function.
    Args:
        w: weight function, takes in a float and returns a float
        mu: factor of mu for the weight function
    Returns:
        diffeq_sys: system of differential equations that
            can be plugged into a numerical solver
    '''
    def diffeq_sys(u, t):
        '''Returns the differential Equation we are trying to solve.
        Args:
            u: numPy array of shape (2,), dependent variables, where
                u[0] represents r(t) and u[1] represents r'(t)
                for our isoperimetric problem
            t: independent variable t
        Returns:
            du: first derivatives of u, where du[0] represents
                r'(t) and du[1] represents r''(t)'''

        return np.array([
            u[1], #= du[0]
            u[0] + 2*u[1]**2/u[0] - (w(t)/(mu*u[0]))*(u[0]**2 + u[1]**2)**(3/2) #= du[1]
        ])

    return diffeq_sys

def tableaus():
    c = [0, 1, 1/2, 3/4]
    v = [0, 1, 1/2, 27/32]
    b = [1/6, 1/6, 2/3, 0]
    x = [[0,      0,     0,  0],
         [0,      0,     0,  0],
         [1/8,   -1/8,   0,  0],
         [3/64,  -9/64,  0,  0]]
         
    return np.array(c), np.array(v), np.array(b), np.array(x)

def IRK4(diffeq, u0, tspan, dt):
    '''Uses 3rd order implicit Runge-Kutta to solve 
    a system of two differential equations
    Args:
        diffeq: function(u,t) -> array of size (2,), where u is also (2,)
        u0: initial guess for the first component.
        dt: step size
    Returns:
        array of shape (3,n). First row is time points, and second and third
            rows are the values of u and v at those points.
    '''
    steps = int(ceil((tspan[1]-tspan[0])/dt))
    t_range = np.linspace(tspan[0], tspan[1], num=steps)
    h = t_range[1] - t_range[0]

    #Initial guess
    x0 = np.array(steps*[u0] + 7*steps*[0])

    def fun(x):
        '''Function to be put into the minimzer.
        Args:
            x: numpy array. Unraveling of a matrix that is           
                [[u0   u1   u2   ... uN-1 uN]
                [k0,1 k1,1 k2,1 ... kN,1 0 ]
                [k0,2 k1,2 k2,2 ... kN,2 0 ]
                [k0,3 k1,3 k2,3 ... kN,3 0 ]
                [v0   v1   v2   ... vN-1 vN]
                [j0,1 j1,1 j2,1 ... jN,1 0 ]
                [j0,2 j1,2 j2,2 ... jN,2 0 ]
                [j0,3 j1,3 j2,3 ... jN,3 0 ]]
        Returns:
            float.'''
        M = np.reshape(x, (8, steps))
        total = []
        f = diffeq
        '''scipy.optimize.minimize only takes a function that returns
            one number, and adding the square of every equation set
            to zero was the most obvious way of minimizing something'''
        for i in range(steps-1):
            #For the variable u

            total.append(M[1,i] - f(M[[0,4],i], t_range[i])[0]) #k1 = f(u0, t0)

            total.append(M[2,i] - f(M[[0,4],i], t_range[i]+h)[0]) #k2 = f(u0, t0 + h)

            total.append(M[3,i] - f(M[[0,4],i]+(1/8)*(M[[1,5],i]-M[[2,6],i])*h, t_range[i]+h/2)[0]) #k3 = f(u0 + [(1/8)k1 - (1/8)k2]h, t0+(1/2)h)

            total.append(M[0,i+1] - M[0,i] - ((1/6)*M[1,i] + (1/6)*M[2,i] + (2/3)*M[3,i])*h) #u1 = u0 + [(1/6)k1 + (1/6)k2 + (2/3)k3]*h
            #Repeat for v
            total.append(M[5,i] - f(M[[0,4],i], t_range[i])[1]) #k1 = f(u0, t0)

            total.append(M[6,i] - f(M[[0,4],i], t_range[i]+h)[1]) #k2 = f(u0, t0 + h)

            total.append(M[7,i] - f(M[[0,4],i]+(1/8)*(M[[1,5],i]-M[[2,6],i])*h, t_range[i]+h/2)[1]) #k3 = f(u0 + [(1/8)k1 - (1/8)k2]h, t0+(1/2)h)

            total.append(M[4,i+1] - M[4,i] - ((1/6)*M[5,i] + (1/6)*M[6,i] + (2/3)*M[7,i])*h) #u1 = u0 + [(1/6)k1 + (1/6)k2 + (2/3)k3]*h
        total.append(M[0,0] - M[0,-1])
        total.append(M[4,0] - M[4,-1])
        print('Iterating...')
        return total

    def jac(x):
        return Jacobian(fun)(x)

    def hess(x):
        return Hessian(fun)(x)

    #Call scipy function
    sol = least_squares(fun, x0, method='dogbox')#, jac=jac, hess=hess, tol=10E-5, options={'disp':True})

    return sol, steps
        
if __name__ == '__main__':
    """Test on a function"""
    '''tspan = (0.0,np.pi/2)
    def simple(u,t):
        return np.array([u[1], -np.sin(u[0])])
    
    t = time.process_time()
    print("Timer started")
    sol, steps = IRK4(simple, np.pi/2, (0, 2*np.pi), 0.1)
    elapsed_time = time.process_time() - t
    print("Elapsed time:", elapsed_time)
    M = np.reshape(sol.x, (8,steps))
    plt.plot(M[4,:])
    plt.show()'''
    #plot(sol1)
    def constant_w(x):
        return 1

    ellipse_w = weight_reg(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, 2, 5)
    diff = el_diffq(ellipse_w)

    t = time.process_time()
    sol, steps = IRK4(diff, ellipse_reg(0,2,5), (0,2*np.pi), 0.2)
    elapsed_time = time.process_time() - t
    print("Elapsed time:", elapsed_time)

    M = np.reshape(sol.x, (8,steps))
    plt.plot(M[0,:])
    plt.plot(M[4,:])
    plt.show()