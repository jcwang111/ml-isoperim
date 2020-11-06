"""This file contains my implementation of nonlinear gradient
de-noising: Taken from Snyder, J. C., Rupp, M., Muller, K.,
& Burke, K. (2015). Nonlinear gradient denoising: Finding accurate extrema
from inaccurate functional derivatives. International Journal of Quantum
Chemistry, 115(16), 1102-1114. https://doi.org/10.1002/qua.24937. [1]

I also used information about kernel PCA from Bernhard Schoelkopf, Alexander J. Smola,
and Klaus-Robert Mueller. 1999. Kernel principal component analysis.
In Advances in kernel methods, MIT Press, Cambridge, MA, USA 327-352. [2]
However, kernel PCA is already implemented in sklearn."""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA, KernelPCA
import itertools
import numdifftools

from r_theta import *
from fourier import *
from functionals import perim, area, reg_perim

def read_function(filename):
    """Reads a file with points of a function, and returns a cubic interpolator.

    Args:
        infile: name of a text file with three rows, each with the same
        number of entries, separated by spaces:
            values of t
            values of y
            values of dy/dt
    Returns:
        A scipy PPoly object that can be called to
        return the interpolation for the point x"""

    infile = open(filename)
    t = np.array([float(value) for value in infile.readline().rstrip().split()])
    y = np.array([float(value) for value in infile.readline().rstrip().split()])
    dy = np.array([float(value) for value in infile.readline().rstrip().split()])
    infile.close()

    return scipy.interpolate.CubicHermiteSpline(t, y, dy, extrapolate='periodic')

def generate_data(f, n=10, feature='fourier'):
    """Generates a data vector out of the function.
    
    Args:
        f: function to turn into a vector.
        n: length of feature vector. If feature=='fourier' and n is even,
            feature vector will be of length n+1. If feature='function', then
            has no effect (n will always be 1).
        feature: 'fourier', 'function', or 'points'. Decides how to
            represent the f function. Defaults to 'fourier'.
    Returns:
        y: numpy array of floats, shape (n,), or (n+1,) if feature=='fourier' and n is odd."""
    
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
    """ Finds best pairs of gamma and alpha for the Gaussian kernel in KRR.

    Args:
        X: numpy array, training data
        y: exact values for the training data
        alpha_values: list of values to use as alpha, the regularization constant
        gamma_values: list of values to use as the parameter in the RBF kernel
    Returns:
        best_alpha, best_gamma: the optimal pair of parameters chosen"""
    
    best_alpha, best_gamma = None, None
    best_score = np.NINF #negative infinity
    for (alpha, gamma) in itertools.product(alpha_values, gamma_values):
        model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
        score = cross_val_score(model, X, y, cv=4).mean()
        if score > best_score:
            best_alpha, best_gamma = alpha, gamma
            best_score = score
    
    return best_alpha, best_gamma

def NLGD_krr_fourier(init_guess, model, steps, eps, area_weight_func):
    """Performs NLGD-projected gradient descent on the KRR model.

    Uses gradient descent to minimize the result of the KRR model
    passed in, while keeping the gradient on the desired manifold
    by using non-linear gradient denoising as described in 
    Snyder et al. [1].
    Args:
        init_guess: vector, initial guess for r
        model: trained KRR model
        steps: no. of steps desired
        eps: constant for gradient descent
        area_weight_func: w of A[r,w] for our isoperimetric problem
    Returns:
        result: vector with the same shape as init_guess, minimized with NLGD"""
    components = 25 #no. of components to project onto
    m = init_guess.size
    X_fit = model.X_fit_
    N = X_fit.shape[0]
    gamma =  model.get_params()['gamma']
    sigma = 1/np.sqrt(2*gamma)
    #Note that sklearn's KernelPCA automatically centers the kernel
    kpca = KernelPCA(n_components = components, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
    kpca.fit(X_fit)
    
    def magnitude(v):
        """Magnitude of a vector"""
        return np.sqrt(np.sum(v**2))

    def k(r0, r1):
        """Application of the kernel"""
        return rbf_kernel(r0, r1, gamma)

    def dk_dr0(r0, r1):
        """gradient of the kernel with respect to r0"""
        return (r1 - r0)*k(r0,r1).T / sigma**2

    def d2k_dr0dr1(r0, r1):
        """2nd order grad of kernel with respect to r0 and r1"""
        #if r0 == r1, grad should be zero
        if magnitude(r0-r1) == 0:
            return np.zeros((r0.size,r1.size))
        else:
            return k(r0,r1)*(1/sigma**2 + (r1-r0).T@(r0-r1)/sigma**4)

    def p_q(r):
        """Estimate of square of projection error, reconstruction error"""
        return k(r,r) - np.sum((k(r, X_fit)@kpca.alphas_)**2)

    def grad_p_q(r):
        """Gradient of the square projection error p_q with respect to r"""
        #check this one if the code fails
        term_two = np.zeros(r.shape)
        for i in range(components):
            for j in range(N):
                for l in range(N):
                    term_two += kpca.alphas_[j,i]*kpca.alphas_[l,i]*k(r, X_fit[j,:].reshape(1,-1))*dk_dr0(r, X_fit[l,:].reshape(1,-1))
        return dk_dr0(r,r) - 2*term_two

    def Hess_r(r0, r1):
        """Hessian of the square projection error p_q"""
        term_two = np.zeros((m, m))
        for i in range(components):
            for j in range(N):
                for l in range(N):
                    term_two += kpca.alphas_[j,i] * kpca.alphas_[l,i] * \
                        (-1*dk_dr0(r0, X_fit[j,:].reshape(1,-1)).T @ dk_dr0(r0, X_fit[l,:].reshape(1,-1)) + \
                             k(r0, X_fit[j,:].reshape(1,-1))*d2k_dr0dr1(r0, X_fit[l,:].reshape(1,-1)))
        return d2k_dr0dr1(r0,r1) - 2*term_two
    
    def grad_krr(r):
        """Returns the gradient of the KRR model."""
        alpha = model.dual_coef_.reshape(1, -1)
        return alpha*k(r, X_fit) @ (X_fit - r)
        
    def Projection_T(r):
        """Projection onto the tangent space of the data manifold""" 
        eigenValues, eigenVectors = np.linalg.eigh(Hess_r(r, r))
        #d eigvectors with nonzero eigvalues, basis for the tangent space
        non_zero_eigvals = np.abs(eigenValues) > 1E-17
        U = eigenVectors[:, non_zero_eigvals]
        #Projection onto the tangent space T
        Proj = U@U.T
        #Projection onto T's orthogonal complement
        P_ortho = np.identity(Proj.shape[0]) - Proj

        return Proj, P_ortho

    def g_NLGD(r, P_ortho):
        """Components of p_q's gradient that are orthogonal to the tangent space.

        Squared magnitude of gradient of p_q error projected
        onto orthogonal complement of the tangent space.
        To be minimized in the correction step."""

        v = P_ortho @ grad_p_q(r.reshape(1,-1)).T
        return np.sum(v**2)

    def area_func(r):
        return area(fourier_series(r.reshape(-1)), area_weight_func)
    
    def area_grad(r):
        return numdifftools.Gradient(area_func)(r)

    def penalty_func(r, p=5, A_0=1):
        """Penalizes perimeter and deviation from the desired area A_0."""
        return model.predict(r)[0] + p*(area_func(r)-A_0)**2

    def grad_penalty_func(r, p=5, A_0=1):
        """Gradient of penalty_func() with respect to r."""
        return grad_krr(r) + 2*p*(area_func(r)-A_0)*area_grad(r)

    print("Beginning gradient descent iterations...")
    result = init_guess
    for step_num in range(1, steps+1):
        print("Taking step {}...".format(step_num))
        #Compute NLGD projection Proj_T
        print(" - Computing NLGD projection")
        Proj_T, P_ortho = Projection_T(result)
        #Projection step
        print(" - Performing projection step")
        grad_model = grad_penalty_func(result)
        result -= (eps/step_num) * (Proj_T@grad_model.reshape(-1)).reshape(1,-1)/magnitude(grad_model)
        #Correction step: minimize g_NLGD along P_ortho
        '''Notes: correction step currently does not even iterate once.
        It may be superfluous here, or there may be an issue in the rest
        of my code.'''
        print(" - Performing correction step")
        result = scipy.optimize.fmin_cg(f=g_NLGD, x0=result.reshape(-1), args=(P_ortho,), disp=True, maxiter=10, callback=print).reshape(1,-1)

    return result