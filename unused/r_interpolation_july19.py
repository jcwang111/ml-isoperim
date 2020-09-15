import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

from r_theta import *
from functional_plots import perim, area, p_loc, p_gea
from interpolations import *


'''In this one, we'll try to approximate a r(theta)
    at any given a and point of w(theta)'''
def w_variation_area(r, delw):
    '''Approximates the change in the area due to the
        change in the weight function. 2nd numbered
        equation in the july 17 logbook.
    Args: r: radius function
        delw: function, change in w
    Returns:
        y: double, value of integral'''

    return integrate(lambda t: 0.5*r(t)**2 *delw(t))

def variation_area(r, w, delr, delw):
    '''Third equation in Logbook July 12, generating
    the variation in area from variational derivatives.
        This provides the function inside the integral
    Args: r, w: radius and weight function
        delr, delw: variation in r and w
    Returns:
        y: double, value of integral'''

    return integrate(lambda t:w(t)*r(t)*delr(t) + 0.5*r(t)**2 *delw(t))

def generate_graph(a_list, b):
    '''Generates the graph and caches it in a text file, since
       the area calculations might take a while
       Args: a_list: array of x_coords for the graph, parameter a'''
    '''Row 0: value of a for ellipse'''
    '''Row 1: value of area for ellipse'''
    with open("r_interpolation_july19.txt", 'w') as outfile:
        for a in a_list:
            w = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a, b)
            print(a, area(ellipse_reg, w, a, b), file=outfile)

def load_array():
    '''returns the array stored in da_dw_july.txt'''
    A = np.zeros((2, len(a_list)))
    with open("r_interpolation_july19.txt") as infile:
        i = 0
        for line in infile:
            A[0,i],A[1,i] = line.lstrip().split(' ')
            i += 1
    return A

B = 50
a_list = np.arange(1, 30, 0.1)
generate_graph(a_list, B)
Area_list = load_array()

'''select our 5 data points to interpolate from'''
points_list = select_points(Area_list, 5, return_endpoints=True)

'''This just repeats the process in da_dw_july2.py so that we can
interpolate Area against the parameter a, nothing new here'''
dAda = np.zeros((points_list.shape[1],))
for i in range(len(points_list[0,:])-1): #loop through each value of a in our list
    a = points_list[0,i]
    a_next = points_list[0,i+1]

    r = lambda t: ellipse_reg(t, a, B)
    w_unreg = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a, B)
    mu = integrate(w_unreg)
    w = lambda t: w_unreg(t)/mu

    r2 = lambda t: ellipse_reg(t, a_next, B)
    w2_unreg = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a_next, B)
    mu2 = integrate(w2_unreg)
    w2 = lambda t: w2_unreg(t)/mu2

    delr = lambda t:r2(t)-r(t)
    delw = lambda t:w2(t)-w(t)

    dAda[i] = variation_area(r, w, delr, delw) / (a_next - a)
dAda[-1] = dAda[-2]

points_list = np.vstack((points_list, dAda))

'''This is where this file is different from da_dw_july2.py'''

'''Points of $a$ to estimate $A[r]$ on using our
    interpolation. I just chose a few at random'''
interp_a_list = np.array([random.random()*29+1 for _ in range(4)])

cubic_interp_areas = piecewise_cubic(points_list, interp_a_list)

'''initialize a list to store our interpolated r functions'''
interp_r_functions = np.empty(interp_a_list.shape[0], dtype=object)
change_in_a = np.empty(interp_a_list.shape[0], dtype=object)
for i in range(len(interp_a_list)):
    '''Use the closest $a$ from points_list, our selected
        points that we had exact values from. That will
        be called closest_a'''
    interp_a = interp_a_list[i]

    differences = [abs(interp_a - point) for point in points_list[0,:]]
    closest_a = points_list[0, differences.index(min(differences))]

    '''we didn't save this info, so let's just reevaluate the
        r(theta) and w(theta) at the point where we have them
        exactly'''
    closest_r = lambda t: ellipse_reg(t, closest_a, B)
    closest_w = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, closest_a, B)

    '''exact w function at interp_a, and delw, the change in w(theta)'''
    interp_w = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, interp_a, B)
    delw = lambda t: interp_w(t) - closest_w(t)

    '''change in area between interp_a and closest_a'''
    area_change = cubic_interp_areas[i] - area(closest_r, closest_w)
    '''Subtract the change in area due to the change in weight
    function so that hopefully we're left with the change in
    area due to the change in r(theta)'''
    area_change_from_r = area_change - w_variation_area(closest_r, delw)

    '''It's hard to recover a specific function you want from 
        inside a definite integral. The best we can do is assume
        delr is a constant function and factor it out to evaluate it'''
    delr = area_change_from_r / integrate(lambda t: closest_w(t)*closest_r(t))
    #print(delr, closest_a, interp_a)

    '''store the interpolated r_function. the index in the array
    is the same as its index in interp_a_list'''
    interp_r_functions[i] = lambda t: (closest_r(t) + delr)
    change_in_a[i] = interp_a - closest_a

'''Let's compare the exact and interpolated r functions for
    one of our points in interp_a_list'''
exact_r = lambda t: ellipse_reg(t, interp_a_list[1], B)
interpolated_r = interp_r_functions[2]

theta = np.arange(0, (2*np.pi), 0.01)

fig = plt.figure()
matplotlib.rc('font', size=7)
for i in range(len(interp_a_list)):
    exact_r = lambda t: ellipse_reg(t, interp_a_list[i], B)
    ax = plt.subplot(2, 2, i+1, projection='polar')
    ax.plot(theta, exact_r(theta), label='Exact r(θ)')
    ax.plot(theta, interp_r_functions[i](theta), label='Interpolated r(θ)')
    ax.axes.xaxis.set_ticklabels([])
    ax.set_title('r(θ) for a={:.3f}, change in a from\n an exact data point is {:.3f}'.format(interp_a_list[i], change_in_a[i]))
    ax.legend()

fig.subplots_adjust(hspace=0.4)
plt.savefig('july19.png')
plt.show()