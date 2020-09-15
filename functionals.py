# Summary: Functionals for A[r] and P[r], to be imported into other files
#          Python 3.7
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from r_theta import integrate

def integrate_quad(r):
    '''integral from 0 to 2pi of the given function times dtheta
        *x is any extra arguments for r'''
    return quad(r, 0, 2*np.pi, limit=400)[0]
    #scipy.integrate.quad runs slower than this manual rectangle summing
    #theta = np.arange(0, (2*np.pi), 0.001)

    #return 0.001*np.sum(r(theta, *x))

def perim(r, r_prime):
    '''Perimeter functional for a given r(theta)
    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      r_prime: a function, first derivative with respect to theta of r
    Returns:
      perimeter: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    return integrate_quad(lambda t : np.sqrt( r(t)**2 + r_prime(t)**2 ))

def area(r, w):
    '''Area functional for a given r(theta).
    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      w: a function, the weight function. the first parameter should be theta
    Returns:
      area: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    
    return 0.5 * integrate_quad( lambda t : r(t)**2 * w(t) )

def p_loc(r):
    '''Local approximation for the perimeter functional'''
    return integrate(r)

def p_gea(r, r_prime):
  '''Gradient expansion approximation for the perimeter functional'''
  return p_loc(r) + (1/2) * integrate(lambda t : r_prime(t)**2/r(t))

def reg_area(r, r_prime, w):
  '''Returns the weighted area of a curve, after regularizing
  the perimeter to be 1.
  Args:
    r: r(t, *args), the polar curve
    r_prime: r'(t, *args), first derivative of r
    w: w(t), polar weight function to be used for the area
  Returns:
    area: double, the calculated area'''

  #For perimeter to be 1, r(theta) must be divided by the current perimeter. Since we have
  # r^2(theta) in A[r], we can simply divide the original A[r] by P^2
  return 0.5 * integrate_quad(lambda t : r(t)**2 * w(t)) / perim(r, r_prime)**2