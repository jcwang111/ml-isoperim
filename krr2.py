'''Here, we will try to use Kernel Ridge Regression on our
isoperimetric problem, doing things similar to what was done in
DOI: 10.1002/qua.25040, "Understanding Machine-Learned
Density Functionals".'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import scipy
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
import itertools

from r_theta import *
from fourier import *
from alt_problem import perim, area, reg_perim

def L2_norm(r_i, r_j):
    '''Currently unused.
    The L2 norm of functions r_i and r_j, with the interval of
    integration between 0 to 2pi.
    Args:
        r_i, r_j: Function inputs.
    Returns:
        norm: The inner product of the two arguments,
            from 0 to 2pi'''
    return integrate(lambda t:r_i(t)*r_j(t)) #integrates from 0 to 2pi

def diff_norm_kernel(l=1.0):
    '''Returns Gaussian of the L2 norm of the difference of two functions,
    if we want to use the entire functions instead of
    basis function coefficients.
    Args:
        l: The length scale of the kernel
    Returns:
        kernel: (function, function)->float. The kernel.'''
    def kernel(f_i, f_j):
        '''Assumes the arguments are arrays,
            as scipy's Kernel Ridge requires'''
        return np.exp(-1*integrate(lambda t: (f_i(t) - f_j(t))**2) / (2 * l**2))
    
    return kernel

def read_function(filename):
    '''Reads a file representing points of a function, and returns
    a cubic interpolator for it.
    Args:
        infile: name of a text file with three rows, each with the same
        number of entries, separated by spaces:
            values of t
            values of y
            values of dy/dt
    Returns:
        A scipy PPoly object that can be called to
        return the interpolation for the point x'''

    infile = open(filename)
    t = np.array([float(value) for value in infile.readline().rstrip().split()])
    y = np.array([float(value) for value in infile.readline().rstrip().split()])
    dy = np.array([float(value) for value in infile.readline().rstrip().split()])
    infile.close()

    return CubicHermiteSpline(t, y, dy, extrapolate='periodic')

def generate_data(w, n=10, feature='fourier'):
    '''Generates a data vector out of the weight function.
    Args:
        w: weight function for the area.
        n: length of feature vector. If feature=='fourier' and n is even,
            feature vector will be of length n+1. If feature='function', then
            has no effect (n will always be 1).
        feature: 'fourier', 'function', or 'points'. Decides how to
            represent the w function. Defaults to 'fourier'.
    Returns:
        y: numpy array of floats, shape (n,), or (n+1,) if feature=='fourier' and n is odd.'''
    
    if (feature == 'function'):
        return np.array([w])
    elif (feature == 'points'):
        theta = np.linspace(0, 2*np.pi, n)
        return w(theta)
    elif (feature == 'fourier'):
        return fourier_coeffs(weight, n= n//2)
    else:
        raise ValueError("Invalid argument for feature: must be 'fourier', 'function', or 'points'.")

def cross_evaluate_krr(X, y, alpha_values, gamma_values):
    '''Cross evaluates for the best pairs
    of paramaters for gamma and alpha for the
    Gaussian kernel in KRR.
    Args:
        X: numpy array, training data
        y: exact values for the training data
        alpha_values: list of values to use as alpha, the regularization constant
        gamma_values: list of values to use as the parameter in the RBF kernel
    Returns:
        best_alpha, best_gamma: the optimal pair of parameters chosen'''
    
    best_alpha, best_gamma = None, None
    best_score = np.NINF #negative infinity
    for (alpha, gamma) in itertools.product(alpha_values, gamma_values):
        model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
        score = model.score(X, y)
        if score > best_score:
            best_alpha, best_gamma = alpha, gamma
            best_score = score
        

def grad_krr(r, model)

if __name__ == '__main__':
    scipy.special.seterr(all='raise')

    #Let's first try a data set of the first 21 sine-cosine Fourier coefficients, for various ellipses
    n = 9 #Number of feature vectors; rows of X
    m = 21 #Length of each feature vector; columns of X
    p = 1 #Number of dependent variables

    X = np.zeros((n, m)) #Contains feature vectors
    Y = np.zeros((n, p)) #Values of P[r,w], dependent variables

    i = 0 #current index

    for a,b in [(3,4), (1,1), (2,5), (1,10), (2,1), (5,7)]:
        weight = weight_reg(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a, b)
        X[i,:] = generate_data(weight, m, 'fourier')
        Y[i, 0] = reg_area(ellipse_reg, ellipse_reg_d1, weight, a, b)
        i += 1

    '''The data points will be the A[r] after normalization as well as w(theta)
    in some representation.'''
    model = KernelRidge(kernel='rbf', gamma=1.0)
    model.fit(X, Y)


            