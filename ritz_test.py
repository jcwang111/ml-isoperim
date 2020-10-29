'''This file will include an attempt to do optimization on Fourier coefficients. However,
it is not clear it will be very fast.'''

import sympy as sp

from r_theta import *
from fourier import *

def area_fourier(r):
    '''Takes a vector of Fourier coefficients of
    a polar curve and returns the area under it.'''
    #r[0] is already twice the actual constant term
    return np.pi/4 * r[0]**2 + np.pi/2 * r[1:]@r[1:]

def grad_area_fourier(r):
    '''Gradient of the area with respect to a vector
    of the Fourier coefficients of r'''
    grad =  np.pi * r
    grad[0] = grad[0]/2

    return grad

def perim_fourier(r):
    '''Takes a vector of Fourier coefficients of
    a polar curve and returns the calculated area.
    There is no easy way to do this without the
    integral.'''

'''Attempt a solution to the isoperimetric problem using the Rayleigh-Ritz method,
choosing the Fourier basis as our basis functions. I chose to use sympy, a symbolic computation
library, since we will need to take partial derivatives of many variables. However, so far
I haven't got it to work. A more numerical method might help, or just some more debugging.'''

#Create Fourier Series with 2n+1 terms: cos(0t)=1 up to cos(nt) and sin(1t) to sin(nt)
n = 2 # >= 2
n = n+1 #So I can type range(n) instead of range(n+1)

t = sp.Symbol('theta', real=True)
cos_coeffs = sp.symbols('a0:{}'.format(n), real=True)
sin_coeffs = sp.symbols('b1:{}'.format(n), real=True)
#declare variables for these symbols
exec(', '.join('a'+str(i) for i in range(0,n)) + ' = cos_coeffs')
exec(', '.join('b'+str(i) for i in range(1,n)) + '= sin_coeffs')

#Up to 21 terms
R_fourier = a0 #type(R_fourier) is a sympy expression, since a0 is a sympy symbol
#add the cos terms
exec('R_fourier = R_fourier +' + '+'.join('a{0}*sp.cos({0}*t)'.format(str(i)) for i in range(1,n)))
exec('R_fourier = R_fourier +' + '+'.join('b{0}*sp.sin({0}*t)'.format(str(i)) for i in range(1,n)))

'''The above code tries this on a constant weight function, w=1.
Currently I am using 5 Fourier series terms, and the exact solution is supposed
to be a_0 =1 and everything else 0.'''