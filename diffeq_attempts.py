'''This file has some common numerical differential
equation solvers written by hand. I will also try
to use pre-written library functions in scipy and
others.'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

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
            u[0] + 2*u[1]**2/u[0] - (w(t)/(mu*u[0]))*(u[0]**2 + u[1]**2)**(3/2) #= du[1]
        ])

    return diffeq_sys

def el_diffq_param(w):
    '''Generates and returns the expanded
    Euler-Lagrange differential
    equation for our weight function, with mu as a parameter
    Args:
        w: weight function, takes in a float and returns a float
        mu: factor of mu for the weight function
    Returns:
        diffeq_sys: system of differential equations that
            can be plugged into a numerical solver
    '''
    def diffeq_sys(t, u, mu):
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
            u[0] + 2*u[1]**2/u[0] - (w(t)/(mu[0]*u[0]))*(u[0]**2 + u[1]**2)**(3/2) #= du[1]
        ])

    return diffeq_sys

def euler_method(diffq, tspan, tstep, u0):
    '''Performs the Euler method on the differential equation.
    Args:
        diffq: function(t, u)->du, where du is an array the same shape as u
        tspan: array-like, shape (2,): [t0, t1], where t0<t1
        tstep: size of each t-step
        u0: array-like, initial values for diffq
    Returns:
        t: time points
        u: value of solution at points t'''
    t = np.arange(*tspan, tstep)
    u = np.zeros((len(u0), len(t)))
    u[:,0] = u0
    for i in range(t.size - 1):
        u[:,i+1] = u[:,i] + tstep*diffq(t[i], u[:,i])
        #print(t[i], u[:,i])
    return t, u

def runge_kutta_four(diffq, tspan, tstep, u0):
    '''Performs fourth-order Runge-Kutta on the differential equation.
    Args:
        diffq: function(t, u)->du, where du is an array the same shape as u
        tspan: array-like, shape (2,): [t0, t1], where t0<t1
        tstep: size of each t-step
        u0: array-like, initial values for diffq
    Returns:
        t: time points
        u: value of solution at points t'''
    t = np.arange(*tspan, tstep)
    u = np.zeros((len(u0), len(t)))
    u[:,0] = u0
    for i in range(t.size - 1):
        k1 = tstep*diffq(t[i], u[:,i])
        k2 = tstep*diffq(t[i]+tstep/2, u[:,i]+k1/2)
        k3 = tstep*diffq(t[i]+tstep/2, u[:,i]+k2/2)
        k4 = tstep*diffq(t[i]+tstep,   u[:,i]+k3  )
        u[:,i+1] = u[:,i] + k1/6 + k2/3 + k3/3 + k4/6
        #print(t[i], u[:,i])
    return t, u

def el_shooting_method_plot(w, mu, u0, init_s=0, scipy_method='Radau', label='Plot', method='scipy'):
    '''Performs the shooting method on the BVP derived from
    the Euler-Lagrange equation, using a scipy method
        Args:
        w: weight function, takes in and returns a float
        mu: value of mu for the euler-lagrange
        u0: value of r(0)
        scipy_method: method to plug into scipy.integrate.solve_ivp
    Returns:
        t: time points
        u: value of solution at points t
        s: derivative of u at first time point, u'(t0)'''
    tstep = 0.01
    tspan = (0, 2*np.pi+tstep)
    diff_eq = el_diffq(w, mu)
    u1 = u0

    if (method == 'scipy'):
        def ivp_solution(s):
            sol = solve_ivp(diff_eq, (0, 2*np.pi+tstep), [u0, s], t_eval=np.arange(0, 2*np.pi+tstep, tstep), method=scipy_method)
            return sol.t, sol.y
    elif (method == 'euler'):
        def ivp_solution(s):
            return euler_method(diff_eq, (0, 2*np.pi), tstep, [u0,s])
    else: #method == 'RK4'
        def ivp_solution(s):
            return runge_kutta_four(diff_eq, (0, 2*np.pi), tstep, [u0,s])
    
    t, ua = ivp_solution(init_s)
    print("Last time point t[-1]:", t[-1])
    #We will use the bisection method: we now need to find our other value of s,
    #such that u(2pi, init_s) and u(2pi, other_s) are different signs

    if np.isclose(ua[0,-1], u1, atol=1.e-5): #u[0,-1]: value of u at last time step
        plt.polar(t, ua[0,:])
        return t, ua
    else:
        print("Searching for s_b...")
        s_a = init_s
        if (ua[0,-1] - u1 > 0):
            s_change = -1
        else:
            s_change = 1
        s_b = s_a + s_change
        t, ub = ivp_solution(s_b)
        print("Last time point t[-1]:", t[-1])
        u = ub
        #if the calculated difference in u1 for both values of s are the same sign
        while ((ua[0,-1] - u1 > 0) == (ub[0,-1] - u1 > 0)):
            print('s_a={}, u_a diff={}, s_b={}, u_b diff={}'.format(s_a, ua[0,-1] - u1, s_b, ub[0,-1] - u1))
            s_change = s_change * 1.5
            s_b = s_b + s_change
            t, ub = ivp_solution(s_b)
            print("Last time point t[-1]:", t[-1])
            u = ub
        ##print(ua[0,-1], ub[0,-1])
    #Perform biscetion method
    print('Running shooting method on initial slope value:')
    while not np.isclose(u[0,-1], u1, atol=1.e-5):
        print(u[0,-1],'=/=', str(u1)+',',"iterating at s-value between", s_a, 'and', s_b)
        #midpoint
        s_m = (s_a + s_b) / 2
        t, u = ivp_solution(s_m)
        print("Last time point t[-1]:", t[-1])
        if (u[0,-1] - u1 > 0):
            s_b = s_m
        else: #u[0,-1] - u1 < 0
            s_a = s_m
        
        if np.isclose(s_a, s_b):
            print('endpoint not perfectly aligned, but halting because s_a == s_b')
            break

    plt.polar(t, u[0,:], label=label)
    #plt.polar(t, u[1,:], label="r'(θ)")


def el_ivp_plot(w, mu, u0, method='scipy', scipy_method='BDF', label='Plot'):
    '''A function for quick testing. Solves the IVP derived from
    the Euler-Lagrange equation with the given weight function.
    Args:
        w: weight function, takes in and returns a float
        mu: value of mu for the euler-lagrange
        u0: values of r(0) and r'(0)
        method: 'euler', 'rk4', or 'scipy'
        scipy_method: method to plug into scipy.integrate.solve_ivp
    Returns: nothing
    '''
    tspan = (0, 2*np.pi)
    tstep = 0.01
    diff_eq = el_diffq(w, mu)
    if method == 'euler':
        t, r = euler_method(diff_eq, tspan, tstep, u0)
    elif method == 'rk4':
        t, r = runge_kutta_four(diff_eq, tspan, tstep, u0)
    else:
        sol = solve_ivp(diff_eq, (0, 2*np.pi+tstep), u0, t_eval=np.arange(0, 2*np.pi+tstep, tstep), method=scipy_method)
        t, r = sol.t, sol.y
    plt.polar(t, r[0,:], label=label)
    #plt.polar(t, r[1,:], label="r'(θ)")
    print(r[0,0], r[0,-1], t[-1])

def el_bvp_plot(w, mu, u0):
    '''A function for quick testing. Solves the BVP using derived from
    the Euler-Lagrange equation with the given weight function,
    using scipy.intergrate.solve_bvp.
    Args:
        w: weight function, takes in and returns a float
        mu: value of mu for the euler-lagrange
        u0: value of both r(0) and r(2pi)
    Returns: nothing
    '''
    tspan = (0, 2*np.pi)
    tstep = 0.01
    diff_eq = el_diffq(w, mu)

    #bc function, to be used in scipy.integrate.solve_bvp
    #boundary conditions: should be 0 for the correct result
    def bc(ya, yb):
        return np.array([ya[0] - yb[0], ya[1] - yb[1]])

    sol = solve_bvp(diff_eq, bc, np.array([0, 2*np.pi]), np.array([[u0, u0], [0, 0]])).sol
    
    theta = np.arange(0, 2*np.pi, tstep)
    print(sol(theta))
    plt.polar(theta, sol(theta)[0,:])
    #print(t, r)

if __name__ == '__main__':
    def const_weight(t):
        return 1

    n, eps, r0 = 10, 0.1, 1#10, 0.1, 1
    smooth_weight = weight_function(return_smooth_reg(n,eps,1), return_smooth_reg_d1(n,eps,1), return_smooth_reg_d2(n, eps, 1))
    r0_smooth = [return_smooth_reg(n, eps, r0)(0), return_smooth_reg_d1(n, eps, r0)(0)]

    third_weight = weight_function(third_reg, third_reg_d1, third_reg_d2)
    r0_third = [third_reg(0), third_reg_d1(0)]

    a, b = 2, 5
    ellipse_weight = weight_function(return_ellipse_reg(a,b), return_ellipse_reg_d1(a,b), return_ellipse_reg_d2(a, b))
    r0_ellipse = [return_ellipse_reg(a, b)(0), return_ellipse_reg_d1(a, b)(0)]
    
    t = np.arange(0, 2*np.pi, 0.02)

    from mirk import MIRK
    def periodic_bc(Y):
        return np.array([Y[0,0] - Y[0,-1],
                         Y[1,0] - Y[1,-1]
        ])
    diffeq = el_diffq(smooth_weight, 1)

    sol, t = MIRK(diffeq, periodic_bc, r0_smooth, (0.0, 2*np.pi), 0.04, endpoint_bc=True, verbose=True)
    plt.polar(t, sol[0,:])
    plt.legend()
    #plt.savefig('BVP_ellipse_2r0.png', bbox='tight')
    plt.show()