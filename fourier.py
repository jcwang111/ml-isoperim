'''This file contains the functions written to work
numerically with Fourier Series.'''

import numpy as np
import quadpy

def integrate_2pi(r):
    '''integral from 0 to 2pi of the given function times dtheta
    Exactly the same as my function in r_theta.py, but defined again
    to make this file stand alone.'''
    return quadpy.quad(lambda t: r(t), 0, 2*np.pi, limit=400)[0]
    #scipy.integrate.quad did not support complex integration,
    #but quadpy does!
    '''Using the more precise quad for integration 
    seems to give better Fourier series approximations. However
    the difference may be negligible.'''
    #theta = np.arange(0, (2*np.pi), 0.001)
    #return 0.001*np.sum(r(theta))

def a_n(y, n):
    '''cos(nt) coefficient in the sine-cosine Fourier series, period is 2pi.
    Note: The constant term, cos(0t)=1, has a coefficient of a_0/2
    since int_0^{2\pi}d\theta = 2\pi'''
    return integrate_2pi(lambda t:y(t)*np.cos(n*t)) / np.pi

def b_n(y, n):
    '''sin(nt) coefficient in the sine-cosine Fourier series, period is 2pi'''
    return integrate_2pi(lambda t:y(t)*np.sin(n*t)) / np.pi

def c_n(y, n):
    '''exp(j*nt) coefficent in the complex Fourier series, j = sqrt(-1), period is 2pi'''
    return integrate_2pi(lambda t:y(t)*np.exp(-1j*n*t)) / (2*np.pi)

def fourier_coeffs(y, n, complex=False):
    '''Returns a numpy array of a number of Fourier coefficients of y
    Args:
        y: Function to find basis coefficients for.
        n: int >= 0, largest n to find the coefficient for.
        complex: bool, if False use real sine-cosine series, otherwise use
            the complex exponential form.
    Returns: numpy array of size (2n+1,). Coefficients of sin(kt) and cos(kt)
        for k up to n if using non-complex form, or coefficients of exp(jkt) and exp(-jkt)
        for k up to n. All coefficients of the constant and cosine first, then all coefficients of sine;
        or all coefficients of constant and +j terms first and then all coefficients of -j terms'''

    if not complex:
        coeffs = [a_n(y, k) for k in range(0,n+1)] + [b_n(y, k) for k in range(1,n+1)]
    else:
        coeffs = [c_n(y, k) for k in range(0,n+1)] + [c_n(y, -1*k) for k in range(1,n+1)]
    return np.array(coeffs)

def fourier_series(coeff_array, complex = False):
    '''Uses an array of coefficients, in the form as returned by fourier_coeffs, and
        returns a function of t'''
    n = coeff_array.shape[0] // 2
    #print(n)

    a_k = coeff_array[:n+1]
    b_k = coeff_array[n+1:]
    #print(a_k, b_k)
    if not complex:
        def y(t):
            result = a_k[0]/2 #constant term
            result += np.sum([a_k[k]*np.cos(k*t) for k in range (1,n+1)]) #cosine terms
            result += np.sum([b_k[k-1]*np.sin(k*t) for k in range (1,n+1)]) #sine terms
            return result
    else:
        def y(t):
            result = a_k[0] #constant term
            result += np.sum([a_k[k]*np.exp(1j*k*t) for k in range (1,n+1)]) #exp(jkt) terms
            result += np.sum([b_k[k-1]*np.exp(-1j*k*t) for k in range (1,n+1)]) #exp(-jkt) terms
            return result
    return np.vectorize(y)

if __name__ == '__main__':
    """Some tests to make sure it's working correctly"""
    g = lambda x: 1+ np.cos(x) + np.sin(x)
    h = lambda x: x**2 + 3*x + 5

    f = np.exp
    N = 10 #We will generate 2N+1 terms

    real_coeffs = fourier_coeffs(f, N, complex=False)
    real_fourier = fourier_series(real_coeffs, complex=False)

    complex_coeffs = fourier_coeffs(f, N, complex=True)
    complex_fourier = fourier_series(complex_coeffs, complex=True)

    import matplotlib.pyplot as plt

    t = np.linspace(0, 2*np.pi, 600)

    plt.plot(t, real_fourier(t), label='Sine-Cosine Fourier Series')
    plt.plot(t, np.real(complex_fourier(t)), dashes=[3, 1], label='Complex Exponential Fourier Series')
    plt.plot(t, f(t), label='Original Curve')
    plt.legend()
    plt.show()
