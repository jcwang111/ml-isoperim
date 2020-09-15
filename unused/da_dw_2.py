import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from r_theta import *
from interpolations import *

'''This file is for the July 12, 2020:Interpolation on A[r]
    entry of the logbook. Here, I keep my w(θ), set to
    be optimal for a specific ellipse, unchanging while varying
    the parameter a for the ellipse and its r(θ)'''

def perim(r, r_prime, *args):
    '''Perimeter functional for a given r(theta)
    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      r_prime: a function, first derivative with respect to theta of r
      args: positional arguments, after theta, to be passed into r and r_prime
    Returns:
      perimeter: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    return integrate(lambda t : np.sqrt( r(t, *args)**2 + r_prime(t, *args)**2 ))

def area(r, w, *args):
    '''Area functional for a given r(theta).
        Regularization isn't included.
    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      w: a function, the weight function. the first parameter should be theta
      args: positional arguments, after theta, to be passed into r and r_prime
    Returns:
      area: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    return 0.5 * integrate( lambda t : r(t, *args)**2 * w(t) )

def variation_area(r, w, delr):
    '''Generates the variation in area from variational derivatives.
        This provides the function inside the integral
    Args: r, w: radius and weight function
        delr: variation in r
    Returns:
        y: double, value of integral'''

    return integrate(lambda t:w(t)*r(t)*delr(t))

def variation_area_left(r, w, r2):
    '''Left side of the equation'''
    return area(r2, w) - area(r, w)

def generate_graph(a_list, B, w):
    '''Generates the graph and caches it in a text file, since
       the area calculations might take a while
       Args: a_list: array of x_coords for the graph, parameter a'''
    '''Row 0: value of a for ellipse'''
    '''Row 1: value of area for ellipse'''

    with open("da_dw_2.txt", 'w') as outfile:
        for a in a_list:
            #Important: our area function already includes regu
            print(a, area(ellipse_p, w, a, B), file=outfile)

def load_array():
    '''returns the array stored in da_dw_july.txt'''
    A = np.zeros((2, len(a_list)))
    with open("da_dw_2.txt") as infile:
        i = 0
        for line in infile:
            A[0,i],A[1,i] = line.lstrip().split(' ')
            i += 1
    return A

B = 50
optimal_a = 30

w_unreg = weight_function(ellipse_p, ellipse_p_d1, ellipse_p_d2, optimal_a, B)
mu = integrate(w_unreg)
weight = lambda t : w_unreg(t)/mu

a_list = np.arange(1, 100, 0.5)
generate_graph(a_list, B, weight)
Area_list = load_array()

points_list = select_points(Area_list, 6, return_endpoints=True)

dAda = np.zeros((points_list.shape[1],))
for i in range(len(points_list[0,:])-1): #loop through each value of a in our list
    a = points_list[0,i]
    a_next = points_list[0,i+1]

    r = lambda t: ellipse_p(t, a, B)
    r2 = lambda t: ellipse_p(t, a_next, B)
    delr = lambda t: r2(t) - r(t)

    dAda[i] = variation_area(r, weight, delr) / (a_next - a)
dAda[-1] = dAda[-2]

points_list = np.vstack((points_list, dAda))

line_interp = line_interpolate(points_list, Area_list[0,:])

cubic_interp = piecewise_cubic(points_list, Area_list[0,:])
cubic_interp[-1] = cubic_interp[-2]

plt.plot(Area_list[0,:], Area_list[1,:], label='Original')
plt.plot(Area_list[0,:], line_interp, label='Line interpolation', color='g')
plt.plot(Area_list[0,:], cubic_interp, label='Piecewise Cubic interpolation',color='r')
plt.plot(points_list[0,:], points_list[1,:], 'x', label='Points used for interpolation',color='orange')
plt.legend()

plt.xlabel("a")
plt.ylabel("A[r]")
plt.savefig("constant_w_A_a.png")
plt.show()