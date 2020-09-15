# Summary: Functions for r(theta) and for producing the w(theta)
#          that solves the Euler-Lagrange equation

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from scipy.integrate import quad

def ellipse(theta, a, b):
    '''Polar function for an ellipse centered at the origin
    Args:
      theta: numpy array of floats, values for theta (preferably [0,2pi])
      a: coefficient for horizontal axis of ellipse
      b: coefficient for vertical axis of ellipse

    Returns:
      radius: numpy array of floats, radius output
    '''
    radius = a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)

    return radius

def ellipse_d1(theta, a, b):
    '''first derivative of r(theta) with respect to theta for the ellipse'''

    return (-1/2)*a*b*(a**2-b**2)*np.sin(2*theta)*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 )**(-3/2)

def return_ellipse(a, b):
    def r(theta):
        return a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)
    return r

def return_ellipse_d1(a, b):
    def r_i(theta):
        return (-1/2)*a*b*(a**2-b**2)*np.sin(2*theta)*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 )**(-3/2)
    return r_i

def smooth_curve(theta, n, epsilon, r):
    '''Polar function for a smooth curve with an oscillating boundary

    Args:
      theta: numpy array of floats, values for theta (preferably [0,2pi])
      n: int, coefficient of theta, number of bumps
      epsilon: float, small number determining height of bumps
      r: base radius

    Returns:
      radius: numpy array of floats, radius output
    '''
    assert type(n) is int
    assert epsilon < 1 and epsilon > -1
    
    radius = r * (1 + epsilon * np.cos(n*theta))

    return radius

def third(theta):
    '''A periodic polar graph'''
    r = 1 + 0.1*np.cos(theta)*np.sin(2*theta)

    return r

def integrate(r, *x):
    '''integral from 0 to 2pi of the given function times dtheta
        *x is any extra arguments for r'''
    #return quad(lambda t: r(t, *x), 0, 2*np.pi)[0]
    #scipy.integrate.quad runs slower than this manual rectangle summing
    theta = np.arange(0, (2*np.pi), 0.001)

    return 0.001*np.sum(r(theta, *x))

def return_ellipse_reg(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg(theta):
        '''Polar function for an ellipse centered at the origin. Regularized'''
        radius = a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)
        return radius/reg_factor

    return ellipse_reg

def return_ellipse_reg_d1(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg_d1(theta):
        '''first derivative of r(theta) with respect to theta for the ellipse'''
        radius = (-1/2)*a*b*(a**2-b**2)*np.sin(2*theta)*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 )**(-3/2)
        return radius/reg_factor

    return ellipse_reg_d1

def return_ellipse_reg_d2(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg_d2(theta):
        '''second derivative of r(theta) with respect to theta for the ellipse'''
        c = ( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 ) # b^2cos^2(th) + a^2sin^2(th)
        radius = (-1)*a*b*(a**2-b**2)*np.cos(2*theta)* c**(-3/2) + (3/4)*a*b*(a**2-b**2)**2 * (np.sin(2*theta))**2 * c**(-5/2)
        return radius/reg_factor

    return ellipse_reg_d2

def return_ellipse_reg_d3(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg_d3(theta):
        '''second derivative of r(theta) with respect to theta for the ellipse'''
        c = ( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 ) # b^2cos^2(th) + a^2sin^2(th)
        d = a**2 - b**2
        result = 2*a*b*d*np.sin(2*theta)*c**(-3/2) + (9/4)*a*b*(d**2)*np.sin(4*theta)*c**(-5/2) \
                    - (15/8)*a*b*(d**3)*(np.sin(2*theta))**3 *(c**(-7/2))

        return result / reg_factor
    
    return ellipse_reg_d3

def return_smooth_reg(n, epsilon, r):
    reg_factor = integrate(smooth_curve, n, epsilon, r)
    def smooth_reg(theta):
        '''Polar function for a smooth curve with an oscillating boundary''' 
        radius = r * (1 + epsilon * np.cos(n*theta))
        return radius/reg_factor
    return smooth_reg

def return_smooth_reg_d1(n, epsilon, r):
    reg_factor = integrate(smooth_curve, n, epsilon, r)
    def smooth_reg_d1(theta):
        '''first derivative of r(theta) with respect to theta for the smooth curve'''
        radius = (-n) * r * epsilon * np.sin(n*theta) 
        return radius/reg_factor
    return smooth_reg_d1

def return_smooth_reg_d2(n, epsilon, r):
    reg_factor = integrate(smooth_curve, n, epsilon, r)
    def smooth_reg_d2(theta):
        '''second derivative of r(theta) with respect to theta for the smooth curve'''
        radius = -1*(n**2) * r * epsilon * np.cos(n*theta)
        return radius/reg_factor
    return smooth_reg_d2

def return_smooth_reg_d3(n, epsilon, r):
    reg_factor = integrate(smooth_curve, n, epsilon, r)
    def smooth_reg_d3(theta):
        '''first derivative of r(theta) with respect to theta for the smooth curve'''
        result =  (n**3) * r * epsilon * np.sin(n*theta)
        return result / reg_factor
    return smooth_reg_d3

third_reg_factor = integrate(third)
def third_reg(theta):
    '''A periodic polar graph'''
    r = 1 + 0.1*np.cos(theta)*np.sin(2*theta)
    return r/third_reg_factor

def third_reg_d1(theta):
    '''First derivative of the periodic curve'''
    r_prime = 0.2*np.cos(theta)*np.cos(2*theta) - 0.1*np.sin(theta)*np.sin(2*theta)

    return r_prime / third_reg_factor

def third_reg_d2(theta):
    '''Second derivative of the periodic curve'''
    r_dprime = -0.4*np.sin(theta)*np.cos(2*theta) - 0.5*np.cos(theta)*np.sin(2*theta)

    return r_dprime / third_reg_factor

def third_reg_d3(theta):
    '''Second derivative of the periodic curve'''
    r_dprime = -1.4*np.cos(theta)*np.cos(2*theta) + 1.3*np.sin(theta)*np.sin(2*theta)

    return r_dprime / third_reg_factor


def fourth(theta):
    return 1 + 0.1*np.sin(3*theta)

fourth_reg_factor = integrate(fourth)
def fourth_reg(theta):
    return ( 1 + 0.1*np.sin(3*theta) ) / fourth_reg_factor

def fourth_reg_d1(theta):
    return 0.3*np.cos(3*theta) / fourth_reg_factor

def fourth_reg_d2(theta):
    return -0.9*np.sin(3*theta) / fourth_reg_factor

def fourth_reg_d3(theta):
    return -2.7*np.cos(3*theta) / fourth_reg_factor

def weight_function(r_, r_i, r_ii):
    '''weight function'''
    def w(t):
      return (r_(t)**2 + 2*r_i(t)**2 - r_(t)*r_ii(t)) / (r_(t)**2 + r_i(t)**2)**(3/2)
    
    return w

def weight_d1(r_, r_i, r_ii, r_iii):
    '''First derivative of w(θ)'''
    def w_i(t):
        R = r_(t)
        R_i = r_i(t)
        R_ii = r_ii(t)
        R_iii = r_iii(t)
        return (-R**3*R_i - 4*R*R_i**3 - 3*R_i**3*R_ii - R*R_iii*(R**2+R_i**2) + 3*R*R_i*R_ii*(R+R_ii)) / (R**2 + R_i**2)**(5/2)
    
    return w_i

def weight_reg(r_, r_i, r_ii):
    '''weight function regularized to integrate to 1'''
    w = weight_function(r_, r_i, r_ii)
    reg_factor = integrate(w)
    return lambda t: w(t)/reg_factor

def weight_reg_d1(r_, r_i, r_ii, r_iii):
    '''First derivative of w(θ), regularized'''
    w = weight_function(r_, r_i, r_ii)
    w_i = weight_d1(r_, r_i, r_ii, r_iii)
    reg_factor = integrate(w)
    return lambda t: w_i(t)/reg_factor