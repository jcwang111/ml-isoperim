'''Sanity checks for various components'''

import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd

from r_theta import *

def w_test():
    '''Test if some w(theta)'s are negative'''
    n, eps = 20,0.1
    w = weight_function(return_smooth_reg(n,eps,1), return_smooth_reg_d1(n,eps,1), return_smooth_reg_d2(n,eps,1))
    t = np.linspace(0, np.pi, 600)

    plt.plot(t, w(t))
    plt.show()
    '''result: some w(theta)'s are negative'''

def derivs_test():
    '''test if the explicitly written derivative functions are correct'''
    a, b = 2, 5
    n, eps = 20,0.1
    r = third_reg
    r_i = third_reg_d1
    r_ii = third_reg_d2
    r_iii = third_reg_d3

    d_i = nd.Derivative(r, n=1)
    d_ii = nd.Derivative(r, n=2)
    d_iii = nd.Derivative(r, n=3)

    t = np.linspace(0, 2*np.pi, 600)
    
    w = weight_reg(r, r_i, r_ii)
    w_i = weight_reg_d1(r, r_i, r_ii, r_iii)

    v_i = nd.Derivative(w, n=1)

    plt.plot(t, w_i(t), linestyle='dashed')
    plt.plot(t, v_i(t))
    plt.show()
    '''result: everything looks good. numdifftools
    is verified to be very accurate too.'''

if __name__ == '__main__':
    derivs_test()