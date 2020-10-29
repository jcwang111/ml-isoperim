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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
import itertools
import numdifftools

from r_theta import *
from fourier import *
from alt_problem import perim, area, reg_perim

def diff_norm_kernel(l=1.0):
    '''Currently unused.
    Returns Gaussian of the L2 norm of the difference of two functions,
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
    #r[0] is already twice the actual constant term
    return np.pi/4 * r[0]**2 + np.pi/2 * r[1:]@r[1:]

def grad_area_fourier(r):
    '''Gradient of the area with respect to a vector
    of the Fourier coefficients of r'''
    grad =  np.pi * r
    grad[0] = grad[0]/2

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
    return grad_krr(r, model) + 2*p*(area_func(r)-A_0)*grad_area_fourier(r)

def perform_gradient_descent(init_guess, cost, cost_grad, steps, args, eps=1.0, produce_graph=False):
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
        iterations = np.zeros(steps)
    
    for i in range(steps):
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

def modified_grad_descent(init_guess, model, steps, eps, l=5):
    '''Performs kPCA and the modified gradient descent algorithm
    on the KRR model that uses 21 fourier coefficients.
    Args:
        init_guess: vector, initial guess for r
        model: trained KRR model
        steps: no. of steps desired
        eps: constant for gradient descent
        l: no. of times to denoise'''
    components = 21 #no. of components to project onto
    X = model.X_fit_
    gamma =  model.get_params()['gamma']
    kpca = KernelPCA(n_components = components, kernel='rbf', gamma=gamma, remove_zero_eig=True, fit_inverse_transform=True)
    kpca.fit(X)

    def magnitude(v):
        '''Magnitude of a vector'''
        return np.sqrt(v@v)

    result = init_guess.reshape(1, -1)
    for k in range(steps):
        result -= eps * grad_krr(result, model) / (1)
        for _ in range(l):
            result = kpca.inverse_transform(kpca.transform(result))

    return result.reshape(-1)
#Old implementation
"""def kPCA_grad_descent_krr_fourier(init_guess, model, steps, eps):
    '''Performs kPCA and the gradient descent on the
    KRR model that uses fourier coefficients as feature
    vectors, doing
    something like the non-linear gradient denoising described
    in the paper.
    Args:
        init_guess: vector, initial guess for r
        model: trained KRR model
        steps: no. of steps desired
        eps: constant for gradient descent'''
    k = None #no. of components to project onto
    X = model.X_fit_
    gamma =  model.get_params()['gamma']
    rbf_feature = Nystroem(gamma = gamma)
    rbf_feature.fit(X)
    kpca = KernelPCA(n_components = k, kernel='rbf', gamma=gamma, remove_zero_eig=True, fit_inverse_transform=True)
    kpca.fit(X)

    def magnitude(v):
        '''Magnitude of a vector'''
        return np.sqrt(v@v)

    def p_q(r):
        '''Kernel projection error function'''
        r_diff = rbf_feature.transform(r.reshape(1,-1)) - rbf_feature.transform(kpca.inverse_transform(kpca.transform(r.reshape(1,-1))))
        return magnitude(r_diff.ravel())

    H = numdifftools.Hessian(p_q)
    
    m = init_guess.shape[0]
    d = m // 2 #no. of eigenvectors from the Hessian of p_q to take as the tangent space

    def P_T(r):
        '''Projection onto the tangent'''
        eigenValues, eigenVectors = np.linalg.eig(H(r))
        idx = eigenValues.argsort()[:d]
        #eigenValues = eigenValues[idx]
        U = eigenVectors
        print(U.shape)
        T_M_basis = eigenVectors[:,idx] #first d eigvectors, basis for the tangent space
        return U@U, T_M_basis.T #Projection matrix, Tangent space projection

    result = init_guess
    for _ in range(steps):
        P, T = P_T(result)
        #projection step
        gradP = grad_krr(result, model)
        print(gradP.shape)
        result -= eps * P@gradP/magnitude(gradP)
        #correction step
        result = T@result

    return result"""

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

    '''test the model'''
    test_param_list = [1.1, 1.4, 1.7, 3, 6]
    n_test = 2*len(test_param_list)**2 #number of feature vectors; rows of X

    X_test = np.zeros((n_test, m)) #Contains feature vectors
    y_exact_test = np.zeros(n_test) #Values of P[r,w], dependent variables

    for (i, (a, b)) in enumerate(itertools.product(test_param_list, test_param_list)):
        r_func = return_ellipse(a, b)
        r_func_d1 = return_ellipse_d1(a, b)
        X_test[2*i,:] = generate_data(r_func, n=21, feature='fourier')
        y_exact_test[2*i] = perim(r_func, r_func_d1, weight)
        #Also do the varied ellipses, r_epsilon
        q_func = lambda t: (r_func(t) + 0.1*np.sin(t))
        q_func_d1 = lambda t: (r_func_d1(t) + 0.1*np.cos(t))
        X_test[2*i+1,:] = generate_data(q_func, n=21, feature='fourier')
        y_exact_test[2*i+1] = perim(q_func, q_func_d1, weight)

    '''y_test = model.predict(X_test)
    plt.plot(y_exact_test, 'x', label='Exact values')
    plt.plot(y_test, '+', label='Predicted values')
    plt.legend()
    #plt.savefig('sep15_perim_comparisons.png')
    plt.show()'''
    #print(area(return_ellipse(1,2)))
    '''Now a test of gradient descent'''
    '''init_r = generate_data(return_ellipse(2, 3), n=21, feature='fourier')
    r_result = perform_gradient_descent(init_r, penalty_func, grad_penalty_krr_fourier, 3200, (model, 10, area_fourier, 2*np.pi), eps=10E-5, produce_graph=True)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    abs_r_series = lambda t: abs(r_series(t))
    plt.polar(t, abs_r_series(t),label=r"First attempt at minimization for #A[r]=1#")
    plt.legend()
    #plt.savefig('sep14_test1.png')
    plt.show()
    '''

    '''Now for gradient descent with PCA part'''
    init_r = generate_data(return_ellipse(1, 2), n=21, feature='fourier')
    r_result = modified_grad_descent(init_r, model, 1600, 10E-5, 1)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    plt.polar(t, r_series(t),label="Attempt minimization with PCA gradient correction for A[r]=1")
    #plt.savefig('sep14_test1.png')
    plt.ylim(top=5)
    plt.show()
    '''Results: for NLGD, stuck at np.linalg.eig(H(r)), which is supposed to find the eigenvalues of the
    Hessian. It says something about a mismatch between dimensions 10 and 21'''