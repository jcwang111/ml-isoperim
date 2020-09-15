'''Here, we will try to use Kernel Ridge Regression on our
isoperimetric problem, doing things similar to what was done in
DOI: 10.1002/qua.25040, "Understanding Machine-Learned
Density Functionals".'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import scipy
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
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

def generate_data(f, n=10, feature='fourier'):
    '''Generates a data vector out of the function.
    Args:
        f: function to turn into a vector.
        n: length of feature vector. If feature=='fourier' and n is even,
            feature vector will be of length n+1. If feature='function', then
            has no effect (n will always be 1).
        feature: 'fourier', 'function', or 'points'. Decides how to
            represent the f function. Defaults to 'fourier'.
    Returns:
        y: numpy array of floats, shape (n,), or (n+1,) if feature=='fourier' and n is odd.'''
    
    if (feature == 'function'):
        return np.array([f])
    elif (feature == 'points'):
        theta = np.linspace(0, 2*np.pi, n)
        return f(theta)
    elif (feature == 'fourier'):
        return fourier_coeffs(f, n= n//2)
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
        score = cross_val_score(model, X, y, cv=4).mean()
        if score > best_score:
            best_alpha, best_gamma = alpha, gamma
            best_score = score
    
    return best_alpha, best_gamma
        

def grad_krr(r, model):
    '''Returns the gradient of the KRR model.
    Args:
        r: input vector
        model: trained KRR model with RBF kernel'''
    X = model.X_fit_
    alpha = model.dual_coef_
    gamma =  model.get_params()['gamma']
    K = rbf_kernel(r.reshape(1,-1), X, gamma)
    
    result = np.zeros(r.shape)
    for i in range(X.shape[0]):
        result += alpha[i]*2*gamma*(X[i,:]-r)*K[0,i]
    
    return result

def area_fourier(r):
    '''Takes a vector of Fourier coefficients of
    a polar curve and returns the area under it.'''
    return np.pi*r[0]**2 + np.pi/2 * r[1:]@r[1:]

def grad_area_fourier(r, model):
    '''Gradient of the area with respect to a vector
    of the Fourier coefficients of r'''
    grad = np.pi * r
    grad[0] = 2*grad[0]
    return grad

def penalty_func(r, model, p, area_func, A_0=1):
    '''Function that penalizes perimeter and deviation
    from the desired area A_0.
    Args:
        r: input vector, shape (m,)
        model: trained KRR model, P_ML[r]
        p: parameter, how much to penalize deviation from the area
        area_func: function that takes in r and returns the area
        A_0: desired area
    Returns:
        cost: float, to be minimized to solve the problem.'''
    
    return model.predict(r.reshape(1,-1))[0] + p*(area_func(r)-A_0)**2

def grad_penalty_krr_fourier(r, model, p, area_func, A_0=1):
    '''Gradient of penalty_func() with respect to r.'''
    return grad_krr(r, model) + 2*p*(area_func(r)-A_0)*grad_area_fourier(r,model)

def perform_gradient_descent(init_guess, cost, cost_grad, n, args, eps=1.0, produce_graph=False):
    '''Implements a naive gradient descent on the cost function.
    Args:
        init_guess: initial guess for the vector r
        cost: cost function to be minimized
        cost_grad: gradient of the cost function with respect to r
        n: number of times to iterate
        args: tuple of extra arguments, beyond r, for cost and cost_grad
        eps: parameter, size of each step
        produce_graph: whether or not to produce a graph of the cost function over every iteration
    Returns:
        result: vector, same shape as init_guess'''
    
    result = init_guess
    if produce_graph:
        iterations = np.zeros(n)
    
    for i in range(n):
        result -= eps * cost_grad(result, *args)
        if produce_graph:
            iterations[i] = cost(result, *args)

    if produce_graph:
        plt.plot(iterations)
        plt.xlabel('Iteration number')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()
    
    return result

if __name__ == '__main__':
    scipy.special.seterr(all='raise')

    #Let's first try a data set of the first 21 sine-cosine Fourier coefficients, for various ellipses
    m = 21 #Length of each feature vector; columns of X
    p = 1 #Number of dependent variables

    param_list = [1, 4/3, 5/3, 2, 4, 5]
    n = len(param_list)**2 #number of feature vectors; rows of X

    X = np.zeros((n, m)) #Contains feature vectors
    y = np.zeros(n) #Values of P[r,w], dependent variables

    weight = return_ellipse(2, 5)
    for (i, (a, b)) in enumerate(itertools.product(param_list, param_list)):
        r_func = return_ellipse(a, b)
        r_func_d1 = return_ellipse_d1(a, b)
        X[i,:] = generate_data(r_func, n=21, feature='fourier')
        y[i] = perim(r_func, r_func_d1, weight)

    alpha_values = [0, 10E-3, 10E-2, 10E-1, 1, 10, 100, 1000]
    gamma_values = [10E-3, 10E-2, 10E-1, 1, 10, 100] #γ = 1/(2σ)^2 in the RBF documentation 
    alpha, gamma = cross_evaluate_krr(X, y, alpha_values, gamma_values)

    '''The data points will be the A[r] after normalization as well as w(theta)
    in some representation.'''
    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    model.fit(X, y)

    '''Now a test of gradient descent'''
    init_r = generate_data(return_ellipse(2, 3), n=21, feature='fourier')
    r_result = perform_gradient_descent(init_r, penalty_func, grad_penalty_krr_fourier, 3200, (model, 5, area_fourier), eps=10E-5, produce_graph=True)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    plt.polar(t, r_series(t))
    plt.show()
