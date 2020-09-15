import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from r_theta import *
from functional_plots import perim, area, p_loc, p_gea
from interpolations import *

'''In this one, we only use those functions of w(θ) and
    r(θ) that we have from those specific points of a'''
def variation_area(r, w, delr, delw):
    '''Third equation in Logbook July 12, generating
    the variation in area from variational derivatives.
        This provides the function inside the integral
    Args: r, w: radius and weight function
        delr, delw: variation in r and w
    Returns:
        y: double, value of integral'''

    return integrate(lambda t:w(t)*r(t)*delr(t) + 0.5*r(t)**2 *delw(t))

def variation_area_left(r, w, r2, w2):
    '''Left side of the equation'''
    return area(r2, w2) - area(r, w)

def generate_graph(a_list, b):
    '''Generates the graph and caches it in a text file, since
       the area calculations might take a while
       Args: a_list: array of x_coords for the graph, parameter a'''
    '''Row 0: value of a for ellipse'''
    '''Row 1: value of area for ellipse'''
    with open("da_dw_july.txt", 'w') as outfile:
        for a in a_list:
            w = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a, b)
            print(a, area(ellipse_reg, w, a, b), file=outfile)

def load_array():
    '''returns the array stored in da_dw_july.txt'''
    A = np.zeros((2, len(a_list)))
    with open("da_dw_july.txt") as infile:
        i = 0
        for line in infile:
            A[0,i],A[1,i] = line.lstrip().split(' ')
            i += 1
    return A

B = 50
a_list = np.arange(1, 30, 0.1)
generate_graph(a_list, B)
Area_list = load_array()

points_list = select_points(Area_list, 6, return_endpoints=True)
print(points_list.shape)

dAda = np.zeros((points_list.shape[1],))
print(points_list)
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

    print(variation_area_left(r, w, r2, w2), variation_area(r, w, delr, delw))

    dAda[i] = variation_area(r, w, delr, delw) / (a_next - a)
dAda[-1] = dAda[-2]

points_list = np.vstack((points_list, dAda))

line_interp = line_interpolate(points_list, Area_list[0,:])
cubic_interp = piecewise_cubic(points_list, Area_list[0,:])

plt.plot(Area_list[0,:], Area_list[1,:], label='Original')
plt.plot(Area_list[0,:], line_interp, label='Line interpolation', color='g')
plt.plot(Area_list[0,:], cubic_interp, label='Piecewise Cubic interpolation',color='r')
plt.plot(points_list[0,:], points_list[1,:], 'x', label='Points used for interpolation',color='orange')
plt.legend()

plt.xlabel("a")
plt.ylabel("A[r, w]")
plt.savefig("A_a_plot3.png")
plt.show()