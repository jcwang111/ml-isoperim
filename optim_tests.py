'''This files has attempts to solve our problem using the CVXPY
and PyOMO libraries in Python. So far, no success with the polar
representation of the isoperimetric problem, even with w(θ)=1. Perhaps
trying more solvers in PyOMO would do it, but we would still eventually
run into problems with more complicated w(θ).'''

import numpy as np
import cvxpy as cp
from pyomo import environ as pyo

from r_theta import *
from functionals import *

def el_diffq(w):
    '''Returns the Euler-Lagrange differential
    equation for our isoperimetric problem given
    the weight function w.'''
    def diffeq_sys(t, u):
        return np.array([
            u[1], #= du[0]
            u[0] + 2*u[1]**2/u[0] - w(t)/u[0]*(u[0]**2 + u[1]**2)**(3/2) #= du[1]
        ])
    return diffeq_sys

def cvxpy_test(wfunc):
    '''Uses the CVXPY constrained optimization library.
    Represents r(theta) as an array of points and tries
    to maximize A[r]. Currently doesn't work because
    it says we can't maximize a function that is convex.'''
    # Problem Data
    n = 500 # points in interval [0,2pi] to discretize r at
    L = 1 # Desired perimeter

    r = cp.Variable(n) # y[i] = y(i/n) for i=0,1,...,n
    t, dt = np.linspace(0, 2*np.pi, n, retstep=True)

    w = cp.Parameter(n, nonneg=True) #weight function, w[i] = w(i/n) for i=0,1,...,n
    print(n)
    print(t.shape)
    w.value = [wfunc(t_i) for t_i in t] #numpy array of w(θ) at our values of θ

    #dr = (r[1:] - r[:-1])

    # Construct Optimization Problem
    objective =  -1/2 * cp.sum(r**2)*dt #Area integral
    constraints = [
        r[0] == r[-1],
        cp.sum(cp.sqrt(r[:-1]**2 + ((r[1:] - r[:-1])/dt)**2))*dt <= L, #perimeter integral
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # Solve Problem
    prob.solve(verbose=True)

def pyomo_test(wfunc):
    '''Attempts to use the PyOMO optimization modeling library.'''
    n = 500 # points in interval [0,2pi] to discretize r at
    L = 1 # Desired perimeter
    
    model = pyo.ConcreteModel()

    model.n = pyo.RangeSet(1, n) #points in interval [0,2pi] to discretize at
    t, dt = np.linspace(0, 2*np.pi, n, retstep=True)

    model.r = pyo.Var(model.n, domain=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(expr = 1/2 * pyo.summation(model.r, model.r)*dt, sense=pyo.maximize)

    model.constraints = pyo.ConstraintList()
    model.constraints.add(model.r[1] == model.r[n])

    def perim_sum():
        for i in range(1,n):
            yield pyo.sqrt(model.r[i]**2 + ((model.r[i+1] - model.r[i])/dt)**2)*dt

    model.constraints.add(pyo.quicksum(perim_sum()) <= L)

    opt = pyo.SolverFactory('glpk')
    opt.solve(model)

    model.display()

if __name__ == '__main__':
    cvxpy_test(lambda t: 1)
    #pyomo_test(None)