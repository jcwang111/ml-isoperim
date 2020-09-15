'''This file runs a few tests, varying the curve of r(theta) a bit
to test if it really is the curve enclosing the largest area with
its given perimeter. Of course, this test does not prove anything,
but only may serve to give some information about them.'''

import numpy as np
import matplotlib.pyplot as plt

from r_theta import *
from functionals import *

def vary_r_and_check_area(r_, r_i, w):
    '''Varies r(theta) and checks the new area.
    The area from the exact r should be greater
    than any of the varied r's.
    Args:
        r_, r_i: r(theta) and first derivative
        w: weight function for the area'''
    exact_area = reg_area(r_, r_i, w)
    print('Exact area:', exact_area)
    #rotate
    for phi in np.linspace(0, 2*np.pi, 100):
        rotated_r = lambda t: r_(t+phi)
        rotated_r_i = lambda t: r_i(t+phi)
        new_area = reg_area(rotated_r, rotated_r_i, w)
        print("Rotation by", phi,'-> area:', new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
    #add epsilon*sin(t) to r(t)
    for eps in np.arange(0.005, 0.3, 0.002):
        new_r = lambda t: r_(t) + eps*np.sin(t)
        new_r_i = lambda t: r_i(t) + eps*np.cos(t)
        new_area = reg_area(new_r, new_r_i, w)
        print('Adding {}sin(t) -> area:'.format(eps), new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            #perimeter = perim(new_r, new_r_i)
            #new_new_r = lambda t: new_r(t) / perimeter
            #new_new_r_i = lambda t: new_r_i(t) / perimeter
            #print('Actual regularized perimeter:', perim(new_new_r, new_new_r_i))

def plot_varied_ellipse():
    a,b = 2,5
    r_ = return_ellipse_reg(a,b)
    r_i = return_ellipse_reg_d1(a,b)
    r_ii = return_ellipse_reg_d2(a,b)
    w = weight_reg(r_, r_i, r_ii)
    #ellipse regularized for perimeter
    orig_perim = perim(r_, r_i)
    rp = lambda t: r_(t)/orig_perim
    rp_i = lambda t: r_i(t)/orig_perim
    print("Confirm that regularized perimeter is 1:", perim(rp, rp_i))

    #ellipse with small change added
    q = lambda t:(r_(t) + 0.1*np.sin(t))
    q_i = lambda t:(r_i(t) + 0.1*np.cos(t))
    q_ii = lambda t:(r_ii(t) - 0.1*np.sin(t))
    orig_q_perim = perim(q, q_i)
    rq = lambda t:q(t)/orig_q_perim
    rq_i = lambda t:q_i(t)/orig_q_perim
    print("Confirm that regularized perimeter is 1:", perim(rq, rq_i))

    print('Area of original ellipse:', area(rp, w))
    print('Area after slight variation:', area(rq, w))
    
    new_weight= weight_reg(q, q_i, q_ii)
    th = np.linspace(0, 2*np.pi, 628)
    plt.polar(th, rp(th), label='Original Ellipse r(θ), a=2 b=5')
    plt.polar(th, rq(th), label='r(θ) + 0.1sin(θ)')
    #plt.polar(th, w(th), label = 'w(θ), the weight function', linestyle='dashed')
    #plt.polar(th, new_weight(th), label='Weight derived from q(θ)')
    #print('Area of q(θ) under its supposedly maximum weight:', area(rq, new_weight))
    plt.legend()
    plt.show()

def alt_vary_r_and_check_area(r_, r_i, w):
    '''Varies r(theta) and checks the new area.
    The area from the exact r should be greater
    than any of the varied r's. For the alternate
    problem, with P[r,w].
    Args:
        r_, r_i: r(theta) and first derivative
        w: weight function for the area'''
    from alt_problem import area, perim, reg_area

    exact_area = reg_area(r_, r_i, w)
    print('Exact area:', exact_area)
    #rotate
    for phi in np.linspace(0, 2*np.pi, 100):
        rotated_r = lambda t: r_(t+phi)
        rotated_r_i = lambda t: r_i(t+phi)
        new_area = reg_area(rotated_r, rotated_r_i, w)
        print("Rotation by", phi,'-> area:', new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
    #add epsilon*sin(t) to r(t)
    for eps in np.arange(0.005, 0.3, 0.002):
        new_r = lambda t: r_(t) + eps*np.sin(t)
        new_r_i = lambda t: r_i(t) + eps*np.cos(t)
        new_area = reg_area(new_r, new_r_i, w)
        print('Adding {}sin(t) -> area:'.format(eps), new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            #perimeter = perim(new_r, new_r_i)
            #new_new_r = lambda t: new_r(t) / perimeter
            #new_new_r_i = lambda t: new_r_i(t) / perimeter
            #print('Actual regularized perimeter:', perim(new_new_r, new_new_r_i))

def september_2_tests_loc(r_, w):
    ###For the local approx, P[r,w] = Int(w(t)*r(t)dt). Solving
    # the Euler-Lagrange equation gets r(t) = μ*w(t). 
    def perim_2(r, w):
        return integrate_quad(lambda t: w(t)*r(t))
    def area_2(r):
        return integrate_quad(lambda t: r(t)**2) / 2
    def reg_area_2(r, w):
        #regularizes perim_2 to 1 and returns area_2
        return area_2(r) / perim_2(r,w)**2

    exact_area = reg_area_2(r_, w)
    print('Exact area:', exact_area)
    #rotate
    for phi in np.linspace(0, 2*np.pi, 100):
        rotated_r = lambda t: r_(t+phi)
        new_area = reg_area_2(rotated_r, w)
        print("Rotation by", phi,'-> area:', new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            perimeter = perim_2(rotated_r, w)
            new_rotated_r = lambda t: rotated_r(t) / perimeter
            print('Actual regularized perimeter:', perim_2(new_rotated_r, w))
    #add epsilon*sin(t) to r(t)
    for eps in np.arange(0.005, 0.3, 0.002):
        new_r = lambda t: r_(t) + eps*np.sin(t)
        new_area = reg_area_2(new_r, w)
        print('Adding {}sin(t) -> area:'.format(eps), new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            perimeter = perim_2(new_r, w)
            new_new_r = lambda t: new_r(t) / perimeter
            print('Actual regularized perimeter:', perim_2(new_new_r, w))

def september_2_tests_r_prime_perim(w, w_prime):
    ###For the local approx, P[r,w] = Int(w(t)*r'^2(t)dt). Isolating
    # r'' gives us el_diffq_3
    from scipy.interpolate import CubicHermiteSpline
    from mirk import MIRK

    def el_diffq_3(w, w_prime, mu=1):
        def diffeq_sys(t, u):
            return np.array([
                u[1], #= du[0]
                u[0]/(2*mu*w(t)) - w_prime(t)*u[1]/w(t) #= du[1]
            ])
        return diffeq_sys
    
    def periodic_bc(Y):
        return np.array([Y[0,0] - Y[0,-1],
                        Y[1,0] - Y[1,-1]
        ])

    sol, t = MIRK(el_diffq_3(w, w_prime), periodic_bc, [1/w(0),0], [0,2*np.pi], 0.01, True, True)
    sol_interpolation = CubicHermiteSpline(t, sol[0,:], sol[1,:], extrapolate='periodic')
    r_ = lambda t: sol_interpolation(t)
    r_i = lambda t: sol_interpolation(t, 1)

    def perim_3(r_prime, w):
        return integrate_quad(lambda t: w(t)*r_prime(t)**2)
    def area_3(r):
        return integrate_quad(lambda t: r(t)**2) / 2
    def reg_area_3(r, r_prime, w):
        #regularizes perim_2 to 1 and returns area_2
        return area_3(r) / perim_3(r_prime, w)

    exact_area = reg_area_3(r_, r_i, w)
    print('Exact area:', exact_area)
    #rotate
    for phi in np.linspace(0, 2*np.pi, 100):
        rotated_r = lambda t: r_(t+phi)
        rotated_r_i = lambda t: r_i(t+phi)
        new_area = reg_area_3(rotated_r, rotated_r_i, w)
        print("Rotation by", phi,'-> area:', new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            perimeter = perim_3(rotated_r_i, w)
            new_rotated_r = lambda t: rotated_r(t) / np.sqrt(perimeter)
            new_rotated_r_i = lambda t: rotated_r_i(t) / np.sqrt(perimeter)
            print(' Actual regularized perimeter:', perim_3(new_rotated_r_i, w))
    #add epsilon*sin(t) to r(t)
    for eps in np.arange(0.005, 0.3, 0.002):
        new_r = lambda t: r_(t) + eps*np.sin(t)
        new_r_i = lambda t: r_i(t) + eps*np.cos(t)
        new_area = reg_area_3(new_r, new_r_i, w)
        print('Adding {}sin(t) -> area:'.format(eps), new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            perimeter = perim_3(new_r_i, w)
            new_new_r = lambda t: new_r(t) / np.sqrt(perimeter)
            new_new_r_i = lambda t: new_r_i(t) / np.sqrt(perimeter)
            print(' Actual regularized perimeter:', perim_3(new_new_r_i, w))
    vary_r_and_check_area(r_, r_i, w)

def sep2_tests_r_prime_perim_known_r(r_, r_i, w):
    def perim_3(r_prime, w):
        return integrate_quad(lambda t: w(t)*r_prime(t)**2)
    def area_3(r):
        return integrate_quad(lambda t: r(t)**2) / 2
    def reg_area_3(r, r_prime, w):
        #regularizes perim_2 to 1 and returns area_2
        return area_3(r) / perim_3(r_prime, w)

    exact_area = reg_area_3(r_, r_i, w)
    print('Exact area:', exact_area)
    #rotate
    for phi in np.linspace(0, 2*np.pi, 100):
        rotated_r = lambda t: r_(t+phi)
        rotated_r_i = lambda t: r_i(t+phi)
        new_area = reg_area_3(rotated_r, rotated_r_i, w)
        print("Rotation by", phi,'-> area:', new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            #perimeter = perim_3(rotated_r_i, w)
            #new_rotated_r = lambda t: rotated_r(t) / np.sqrt(perimeter)
            #new_rotated_r_i = lambda t: rotated_r_i(t) / np.sqrt(perimeter)
            #print(' Actual regularized perimeter:', perim_3(new_rotated_r_i, w))
    #add epsilon*sin(t) to r(t)
    for eps in np.arange(0.005, 0.3, 0.002):
        new_r = lambda t: r_(t) + eps*np.sin(t)
        new_r_i = lambda t: r_i(t) + eps*np.cos(t)
        new_area = reg_area_3(new_r, new_r_i, w)
        print('Adding {}sin(t) -> area:'.format(eps), new_area)
        if (new_area > exact_area):
            print(' New area > exact_area')
            #perimeter = perim_3(new_r_i, w)
            #new_new_r = lambda t: new_r(t) / np.sqrt(perimeter)
            #new_new_r_i = lambda t: new_r_i(t) / np.sqrt(perimeter)
            #print(' Actual regularized perimeter:', perim_3(new_new_r_i, w))

if __name__ == '__main__':
    a, b  = 2, 5
    #ellipse_weight = weight_reg(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a, b)
    '''can use integrate or integrate_quad in functionals.py'''
    #Ellipse: failed test, a=2 b=5
    #vary_r_and_check_area(ellipse_reg, ellipse_reg_d1, ellipse_weight, a, b)
    #Circle: succeeded test
    #vary_r_and_check_area(lambda t:1, lambda t:0, lambda t:1)
    #Third Curve: failed test
    #third_weight = weight_reg(third_reg, third_reg_d1, third_reg_d2)
    #vary_r_and_check_area(third_reg, third_reg_d1, third_weight)
    #Smooth Curve: passed test, for all purposes, n=10 eps=0.1
    n, eps = 10, 0.1
    #smooth_weight = weight_reg(smooth_reg, smooth_reg_d1, smooth_reg_d2, n, eps, 1)
    #vary_r_and_check_area(smooth_reg, smooth_reg_d1, smooth_weight, n, eps, 1)
    #Fourth Curve: passed test
    #fourth_weight = weight_reg(fourth_reg, fourth_reg_d1, fourth_reg_d2)
    #vary_r_and_check_area(fourth_reg, fourth_reg_d1, fourth_weight)
    #plot_varied_ellipse()
    '''Tests for P_loc, September 2'''
    ###Since for P_loc we get w(t)=r(t) from the Euler-Lagrange, we just
    # need to provide any curve and its derivative, and vary it.
    # We might as well confirm that it for sure doesn't solve the actual problem, though.
    ellipse_r = return_ellipse_reg(a, b)
    ellipse_r_i = return_ellipse_reg_d1(a, b)
    ellipse_r_ii = return_ellipse_reg_d2(a, b)
    #w = ellipse_r
    #september_2_tests_loc(ellipse_r, ellipse_r)
    ###Failed for both rotation and adding eps*sin(t)
    #alt_vary_r_and_check_area(ellipse_r, ellipse_r_i, ellipse_r)
    ###Now for P[r,w] = Int(w(t)*r'^2(t)dt)
    #it's not so easy to solve for this, even analytically.
    ellipse_r_iii = return_ellipse_reg_d3(a, b)
    w = weight_function(ellipse_r, ellipse_r_i, ellipse_r_ii)
    w_prime = weight_d1(ellipse_r, ellipse_r_i, ellipse_r_ii, ellipse_r_iii)
    #september_2_tests_r_prime_perim(ellipse_r, ellipse_r_i)
    ###succeeded for rotation, failed for adding eps*sin(t)
    #september_2_tests_r_prime_perim(w, w_prime)
    ###This one fails for some rotations, but succeeds for adding the eps*sin(t)
    #under original test, fails for rotations

    ####
    sep2_tests_r_prime_perim_known_r(lambda t:np.sin(t) + np.cos(t), lambda t:np.cos(t) - np.sin(t), lambda t: 1)