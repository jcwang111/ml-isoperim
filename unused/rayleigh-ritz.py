# Summary: Attempts to use a form of the Rayleigh-Ritz method
#          to solve the differential equation. We will use
#          the Fourier series basis as our basis functions.
'''Currently not usable'''

import sympy as sp
import numpy as np
import mpmath
from scipy.optimize import root

if __name__=='__main__':
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
    print(R_fourier)
    
    #B = A[r]-μP[r], functional to be minimized
    #Try with constant weight function first
    R = R_fourier

    integrand = sp.Rational(1,2)*R**2 - sp.sqrt(R**2 + R.diff(t))
    
    #Sympy tries hard to evaluate it if you do integrate(), so
    # use Integral instead to make it an integral object
    B = sp.Integral(integrand, (t, 0, 2*sp.pi)) 
    #B = integrand
    #delB/delc_i = 0 for every coefficient c_i
    all_coeffs = cos_coeffs + sin_coeffs
    partial_derivs = [sp.lambdify(all_coeffs, B.diff(a_i),'numpy') for a_i in cos_coeffs] + [sp.lambdify(all_coeffs, B.diff(b_i),'numpy') for b_i in sin_coeffs]
    print('Coefficients to solve for: ', all_coeffs)
    #sol = sp.nsolve(partial_derivs, all_coeffs, (0.1,)*(n*2-1), modules=['sympy', 'mpmath'])
    '''sympy's nsolve didn't work on this function for some reason, so I
    rewrote line 42 to convert the symbolic equations to numerical Python functions, and
    will try to use scipy's root finder.'''
    def partial_derivs_vector(x):
        _x = tuple(x)
        return np.array([dBdc(*_x) for dBdc in partial_derivs])
    root(partial_derivs_vector, [0.1]*(n*2-1))
    print(sol)

    '''The above code tries this on a constant weight function, w=1.
    Currently I am using 5 Fourier series terms, and the exact solution is supposed
    to be a_0 =1 and everything else 0.'''