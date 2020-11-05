'''Here, we will try to use Kernel Ridge Regression on our
isoperimetric problem, doing things similar to what was done in
DOI: 10.1002/qua.25040, "Understanding Machine-Learned
Density Functionals". In this one, we will place the
weight function in the area functional.

This one also contains the true test of nonlinear gradient
de-noising: Taken from Snyder, J. C., Rupp, M., Muller, K.,
& Burke, K. (2015). Nonlinear gradient denoising: Finding accurate extrema
from inaccurate functional derivatives. International Journal of Quantum
Chemistry, 115(16), 1102-1114. https://doi.org/10.1002/qua.24937.

I also used information about kernel PCA from Bernhard Schoelkopf, Alexander J. Smola,
and Klaus-Robert Mueller. 1999. Kernel principal component analysis.
In Advances in kernel methods, MIT Press, Cambridge, MA, USA 327-352.
However, kernel PCA is already implemented in sklearn.'''

import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA, KernelPCA
import itertools

from r_theta import *
from fourier import *
from functionals import perim, area, reg_perim

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

    return scipy.interpolate.CubicHermiteSpline(t, y, dy, extrapolate='periodic')

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

def grad_penalty_krr_fourier(r, model, p, area_func, area_grad, A_0=1):
    '''Gradient of penalty_func() with respect to r.'''
    return grad_krr(r, model) + 2*p*(area_func(r)-A_0)*area_grad(r)


def NLGD_krr_fourier(init_guess, model, steps, eps):
    '''Performs kPCA and the gradient descent on the
    KRR model that uses fourier coefficients as feature
    vectors, doing something like the non-linear
    gradient denoising described in the paper.
    Args:
        init_guess: vector, initial guess for r
        model: trained KRR model
        steps: no. of steps desired
        eps: constant for gradient descent'''
    components = 25 #no. of components to project onto
    X_fit = model.X_fit_
    N = X_fit.shape[0]
    gamma =  model.get_params()['gamma']
    sigma = 1/np.sqrt(2*gamma)
    #Note that sklearn's KernelPCA automatically centers the kernel
    kpca = KernelPCA(n_components = components, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
    kpca.fit(X)

    kernel = rbf_kernel(gamma)
    
    def magnitude(v):
        '''Magnitude of a vector'''
        return np.sqrt(v@v)

    def k(r0, r1):
        '''Application of the kernel'''
        return rbf_kernel(r0, r1, gamma)

    def dk_dr0(r0, r1):
        '''gradient of the kernel with respect to r0'''
        return (r1 - r0)*k(r0,r1).T / sigma**2

    def d2k_dr0dr1(r0, r1):
        '''2nd order grad of kernel with respect to r0 and r1'''
        #if r0 == r1, grad should be zero
        if magnitude(r0-r1) == 0:
            return np.zeroes((r0.size,r1.size))
        else:
            return k(r0,r1)*(1/sigma**2 + (r1-r0).T@(r0-r1)/sigma**4)

    def p_q(r):
        '''Estimate of square of projection error, reconstruction error'''
        return k(r,r) - np.sum((k(r, X_fit)@kpca.alphas_)**2)

    def grad_p_q(r):
        '''Gradient of the square projection error p_q with respect to r'''
        #check this one if the code fails
        term_two = np.zeros(r.shape)
        for i in range(components):
            for j in range(N):
                for l in range(N):
                    term_two += kpca.alphas[j,i]*kpca.alphas[l,i]*k(r, X_fit[j,:])*dk_dr0(r, X_fit[l,:])
        return dk_dr0(r,r) - 2*term_two

    def Hess_r(r0, r1):
        '''Hessian of the square projection error p_q'''
        term_two = np.zeros((r0.shape[0], r0.shape[0]))
        for i in range(components):
            for j in range(N):
                for l in range(N):
                    term_two += kpca.alphas[j,i] * kpca.alphas[l,i] * \
                        (-1*dk_dr0(r0, X_fit[j,:]) @ dk_dr0(r0, X_fit[l,:]) + k(r0, X_fit[j,:])*d2k_dr0dr1(r0, X_fit[l,:]))
        return d2k_dr0dr1 - 2*term_two
    
    def grad_krr(r):
        '''Returns the gradient of the KRR model.'''
        alpha = model.dual_coef_.reshape(1, -1)
        return alpha*k(r, X_fit) @ (X_fit - r)

    m = init_guess.shape[0]
    d = components // 2 #no. of eigenvectors from the Hessian of p_q to take as the tangent space

    def Projection_T(r):
        '''Projection onto the tangent space of the data manifold''' 
        eigenValues, eigenVectors = np.linalg.eig(Hess_r(r, r))
        idx = eigenValues.argsort()[:d]
        print(eigenValues)
        #eigenValues = eigenValues[idx]
        U = eigenVectors[:, idx] #first d eigvectors, basis for the tangent space
        #Projection onto the tangent space T
        Proj = U@U.T
        #Projection onto T's orthogonal complement
        P_ortho = np.identity(P.shape[0]) - Proj

        return Proj, P_ortho

    def g_NLGD(r, P_ortho):
        '''Squared magnitude of gradient of p_q error projected
            onto orthogonal complement of the tangent space.
            To be minimized in the correction step.'''
        v = P_ortho @ delp_delr(r)
        return v@v

    print("Beginning gradient descent iterations...")
    result = init_guess
    for step_num in range(1, steps+1):
        print("Taking step {}...".format(step_num))
        #Compute NLGD projection Proj_T
        Proj_T, P_ortho = Projection_T(result)
        #Projection step
        grad_model = grad_krr(result).reshape(-1,1)
        result -= (eps/step_num) * Proj_T@grad_model/magnitude(grad_model)
        #Correction step: minimize g_NLGD along P_ortho
        scipy.optimize.fmin_cg

    return result