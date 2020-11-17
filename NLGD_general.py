"""General implementation of nonlinear gradient
de-noising: Taken from Snyder, J. C., Rupp, M., Muller, K.,
& Burke, K. (2015). Nonlinear gradient denoising: Finding accurate extrema
from inaccurate functional derivatives. International Journal of Quantum
Chemistry, 115(16), 1102-1114. https://doi.org/10.1002/qua.24937. [1]

I also used information about kernel PCA from Bernhard Schoelkopf, Alexander J. Smola,
and Klaus-Robert Mueller. 1999. Kernel principal component analysis.
In Advances in kernel methods, MIT Press, Cambridge, MA, USA 327-352. [2]
However, kernel PCA is already implemented in sklearn.

Current status: not 100% confirmed to be working yet."""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA, KernelPCA
import itertools
import numdifftools
from pathos.multiprocessing import ProcessingPool as Pool

def NLGD_KRR(init_guess, model, steps, eps, v=None, correction_step_interval=50):
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
        v: optional, function, v in dT[n]/dn(r) + v(r); generally, gradient of
            anything besides the KRR model that we also want to minimize
        correction_step_interval: steps before taking each correction step. For efficiency purposes.
    Returns:
        result: vector with the same shape as init_guess, minimized with NLGD"""
    
    if correction_step_interval == None or correction_step_interval < 1:
        correction_step_interval = steps 

    m = init_guess.size
    X_fit = model.X_fit_
    N = X_fit.shape[0]
    components = N #no. of components to keep in kPCA. Denoted in the paper as 'q'.
    gamma =  model.get_params()['gamma']
    sigma = 1/np.sqrt(2*gamma)
    #Note that sklearn's KernelPCA automatically centers the kernel
    kpca = KernelPCA(n_components = components, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
    kpca.fit(X_fit)

    #alpha_kPC is the coefficients for the principal components as a lin. combination of feature space vectors
    alpha_kPC = kpca.alphas_
    
    def magnitude(v):
        """Magnitude of a vector"""
        return np.sqrt(np.sum(v**2))

    def k(r0, r1):
        """Application of the kernel"""
        return rbf_kernel(r0, r1, gamma)

    def grad_krr(r):
        """Returns the gradient of the KRR model."""
        alpha = model.dual_coef_.reshape(1, -1)
        return 2*gamma*alpha*k(r, X_fit) @ (X_fit - r)

    def dk_dr0(r0, r1):
        """gradient of the kernel with respect to r0"""
        return (r1 - r0)*k(r0,r1).T / sigma**2

    def p_q(r):
        """Estimate of square of projection error, reconstruction error"""
        r = r.reshape(1,-1)
        return k(r,r)[0,0] - np.sum((k(r, X_fit)@kpca.alphas_)**2)

    def grad_p_q(r):
        """Gradient of the square projection error p_q with respect to r"""
        #check this one if the code fails
        def term_two_term(ijl):
            return alpha_kPC[ijl[1],ijl[0]]*alpha_kPC[ijl[2],ijl[0]] * k(r, X_fit[[ijl[1]]]*dk_dr0(r, X_fit[[ijl[2]]]))
        term_two_to_sum = Pool(3).amap(term_two_term, ((i,j,l) for i in range(components) for j in range(N) for l in range(N))).get()
        '''for i in range(components):
            for j in range(N):
                for l in range(N):
                    term_two += kpca.alphas_[j,i]*kpca.alphas_[l,i]*k(r, X_fit[[j]]*dk_dr0(r, X_fit[[l]]))'''
        return dk_dr0(r,r) - 2*np.sum(term_two_to_sum)

    def d2k_dr0dr1(r0, r1):
        """Hessian of k[r,r] with respect to independent variables of r"""
        #if r0 == r1, grad should be zero
        return k(r0,r1)*( (r1-r0).T@(r1-r0)/sigma**4 - np.identity(m)/sigma**2 )

    def Hess_r(r):
        """Hessian of the square projection error p_q"""
        def term_two_term(ijl):
            return alpha_kPC[ijl[1],ijl[0]] * alpha_kPC[ijl[2],ijl[0]] * \
                            dk_dr0(r, X_fit[[ijl[1]]]).T @ dk_dr0(r, X_fit[[ijl[2]]]) + \
                             k(r, X_fit[[ijl[1]]])*d2k_dr0dr1(r, X_fit[[ijl[2]]])
        term_two_to_sum = Pool(3).amap(term_two_term, ((i,j,l) for i in range(components) for j in range(N) for l in range(N))).get()
        '''for i in range(components):
            for j in range(N):
                for l in range(N):
                    term_two += kpca.alphas_[j,i] * kpca.alphas_[l,i] * \
                            dk_dr0(r, X_fit[[j]]).T @ dk_dr0(r, X_fit[[l]]) + \
                             k(r, X_fit[[j]])*d2k_dr0dr1(r, X_fit[[l]])'''
        return d2k_dr0dr1(r,r) - 2*np.sum(term_two_to_sum)
        
    def Projection_T(r):
        """Projection onto the tangent space of the data manifold"""
        eigenValues, eigenVectors = np.linalg.eigh(Hess_r(r))
        #print(eigenValues)
        #d eigvectors with nonzero eigvalues, basis for the tangent space
        #non_zero_eigvals = np.abs(eigenValues) > 1E-17
        U = eigenVectors[:, np.abs(eigenValues) > 3E-2]
        #print(U)
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

        #v = (P_ortho @ grad_p_q(r.reshape(1,-1)).T)
        return np.sum((P_ortho @ grad_p_q(r.reshape(1,-1)).T)**2)

    if v == None:
        cost_grad = grad_krr
    else:
        cost_grad = lambda r: (grad_krr(r) + v(r))

    print("Beginning gradient descent iterations...")
    result = init_guess
    for step_num in range(1, steps+1):
        print("Taking step {}...".format(step_num))
        #Compute NLGD projection Proj_T
        print(" - Computing NLGD projection")
        Proj_T, P_ortho = Projection_T(result)
        #Projection step
        print(" - Performing projection step")
        result -= (eps/1) * (cost_grad(result)@Proj_T.T)#/magnitude(grad_model)
        #Correction step: minimize p_q. Changed from minimizing g_NLGD along P_ortho, which did not give the desired result.
        if step_num % correction_step_interval == 0:
            print(" - Performing correction step")
            result = scipy.optimize.fmin_cg(f=p_q, x0=result.reshape(-1), gtol=1E-15, disp=True, maxiter=20).reshape(1,-1)
            #plt.polar(np.linspace(0, 2*np.pi, 100), result.reshape(-1))
            #plt.show()

    return result