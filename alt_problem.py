'''In this file, the weight function will be put into P[r,w]
instead of A[r], and we will see what results we get from there.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline

from r_theta import *
from mirk import MIRK

def integrate_quad(r):
    '''integral from 0 to 2pi of the given function times dtheta
        *x is any extra arguments for r'''
    return quad(r, 0, 2*np.pi, limit=400)[0]

def perim(r, r_prime, w):
    return integrate_quad(lambda t : w(t) * np.sqrt( r(t)**2 + r_prime(t)**2 ))

def area(r):
    return 0.5 * integrate_quad( lambda t : r(t)**2 )

def reg_area(r, r_prime, w):
    return area(r) / perim(r, r_prime, w)**2

def el_diffq(w, w_prime, mu=1):
    '''Generates and returns the expanded
    Euler-Lagrange differential
    equation for our weight function.
    Args:
        w: weight function, takes in a float and returns a float
        mu: factor of mu for the weight function
    '''
    def diffeq_sys(t, u):
        '''Returns the differential Equation we are trying to solve.
        Args:
            t: independent variable t
            u: numPy array of shape (2,), dependent variables, where
                u[0] represents r(t) and u[1] represents r'(t)
                for our isoperimetric problem
        Returns:
            du: first derivatives of u, where du[0] represents
                r'(t) and du[1] represents r''(t)'''

        return np.array([
            u[1], #= du[0]
            u[0] + 2*u[1]**2/u[0] - w_prime(t)*u[1]/w(t)*(1+u[1]**2/u[0]**2)- mu/w(t)/u[0]*(u[0]**2+u[1]**2)**(3/2)#= du[1]
        ])

    return diffeq_sys

def periodic_bc(Y):
    return np.array([Y[0,0] - Y[0,-1],
                        Y[1,0] - Y[1,-1]
    ])

def check_EL_eq(w, w_prime, t, sol):
  '''Evaluates the left side of our euler-lagrange equation.
    It should be uniformly zero.
    args:
      w, w_prime: weight function and derivative used in el_diffq
      t, sol: time-points and solution returned by MIRK'''
  mu = 1
  r = CubicHermiteSpline(t, sol[0], sol[1], extrapolate='periodic')
  t = np.linspace(0,2*np.pi,300)
  expr = w(t)*(r(t)**3 + 2*r(t)*r(t,1)**2 - r(t)**2*r(t,2))/(r(t)**2+r(t,1)**2)**(3/2) - w_prime(t)*r(t,1)/(r(t)**2 + r(t,1)**2)**(1/2) - mu*r(t)

  plt.plot(t, expr, label='Should be uniformly zero.')
  plt.show()

if __name__ == '__main__':
  #Set up w(theta), etc.

  def constant_w(t):
    return 1

  def constant_w_prime(t):
    return 0
  
  third_weight = weight_function(third_reg, third_reg_d1, third_reg_d2)

  a, b = 2, 5
  ellipse_weight = weight_function(return_ellipse_reg(a,b), return_ellipse_reg_d1(a,b), return_ellipse_reg_d2(a, b))
  ellipse_weight_d1 = weight_d1(return_ellipse_reg(a,b), return_ellipse_reg_d1(a,b), return_ellipse_reg_d2(a, b), return_ellipse_reg_d3(a,b))

  #Use the solver

  diffeq = el_diffq(ellipse_weight, ellipse_weight_d1, 1)
  dt = 0.01
  sol, t = MIRK(diffeq, periodic_bc, [1/ellipse_weight(0),0], (0.0, 2*np.pi), dt, endpoint_bc=True)
  check_EL_eq(ellipse_weight, ellipse_weight_d1, t, sol)
  #print(sol[0,:])
  plt.polar(t, -sol[0,:], label='r(θ) produced by the solver, Δt={}'.format(dt))
  #plt.polar(t+(sol[0,:]<0)*np.pi, np.abs(sol[0,:]))
  #plt.polar(t, ellipse_weight(t), label='w(θ) used')
  #plt.legend()
  #plt.savefig('neg_test_ellipse_problem_dt{}.png'.format(dt), bbox='tight')
  plt.show()