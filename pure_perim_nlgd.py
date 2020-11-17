'''A copy of the procedure as done in Snyder, J. C. et al. Kernels, Pre-images,
and Optimization. Chapter in Empirical Inference: Festschrift in Honor of Vladimir N. Vapnik 245–259 (2013).
The purpose is to ensure that the code for my own implemented NLGD procedure works correctly, and
so I will be checking my graphs against those in the book chapter/paper to see if they are the same.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from scipy.integrate import quad
import scipy
import itertools
import time

from NLGD_general import NLGD_KRR

def ellipse(theta, a, b):
    """Ellipse polar curve"""
    return a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)

def return_ellipse(a, b):
    """Returns the parameterized ellipse as a function of only theta"""
    def r(theta):
        return a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)
    return r

def ellipse_d1(theta, a, b):
    """first derivative of r(theta) with respect to theta for the ellipse"""
    return (-1/2)*a*b*(a**2-b**2)*np.sin(2*theta)*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 )**(-3/2)

def return_ellipse_d1(a, b):
    def r_i(theta):
        return (-1/2)*a*b*(a**2-b**2)*np.sin(2*theta)*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 )**(-3/2)
    return r_i

def return_ellipse_d2(a, b):
    def ellipse_d2(theta):
        """Second derivative of r(theta) with respect to theta, for the ellipse"""
        c = ( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 ) # b^2cos^2(th) + a^2sin^2(th)
        return (-1)*a*b*(a**2-b**2)*np.cos(2*theta)* c**(-3/2) + (3/4)*a*b*(a**2-b**2)**2 * (np.sin(2*theta))**2 * c**(-5/2)

    return ellipse_d2

def perim(r, r_prime):
    """Exact perimeter functional"""
    return quad(lambda t : np.sqrt(r(t)**2 + r_prime(t)**2), 0, 2*np.pi, limit=400)[0]

def delperim_delr(r_, r_i, r_ii):
    """Returns the exact functional derivative of perim with respect to r"""
    def dPdr(t):
        return (r_(t)**3 + 2*r_(t)*r_i(t)**2 - r_(t)**2*r_ii(t)) / (r_(t)**2 + r_i(t)**2)**(3/2)

    return dPdr

def area(r):
    """Exact area functional. Unused in this file."""
    return 0.5 * integrate_quad( lambda t : r(t)**2 )

def grad_krr(r, model):
    """Returns the gradient of a KRR model."""
    X_fit = model.X_fit_
    alpha = model.dual_coef_.reshape(1, -1)
    gamma =  model.get_params()['gamma']
    r = r.reshape(1,-1)
    return 2*gamma*alpha*rbf_kernel(r, X_fit,gamma) @ (X_fit - r)

if __name__ == '__main__':
    scipy.special.seterr(all='raise')
    
    """Train the model"""
    param_list = [1, 4/3, 5/3, 2]
    n = len(param_list)**2 #number of feature vectors
    m = 100 #Length of each feature vector; columns of X
    
    X = np.zeros((n, m)) #Contains feature vectors
    y = np.zeros(n) #Exact values of P[r]

    theta, step = np.linspace(0, 2*np.pi, m, retstep=True)
    for (i, (a, b)) in enumerate(itertools.product(param_list, param_list)):
        r = return_ellipse(a, b)
        r_d1 = return_ellipse_d1(a, b)
        X[i,:] = r(theta) #100 bins
        y[i] = perim(r, r_d1)

    #We use parameter values as given in the paper
    alpha = 1E-6
    sigma = 6

    gamma = 1/(2*sigma**2)
    print('Alpha:', alpha)
    print('Sigma:', 1/np.sqrt(2*gamma))

    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    model.fit(X, y)
    
    Y = model.predict(X)

    #Note: the book chapter calls their representation of r(theta) a "histogram" representation of
    # r(theta). However, judging by their equation for the gradient of the approximate area,
    # they really have a representation of r(theta) that is the function at certain points in [0, 2pi].
    # (That is not what a histogram representation is.)

    '''
    # Graphs of the functional derivative of the exact perimeter functional and the KRR model
    a, b = 1.2, 1.3
    dPdr = delperim_delr(return_ellipse(a,b), return_ellipse_d1(a,b), return_ellipse_d2(a,b))
    plt.plot(theta, dPdr(theta))
    plt.plot(theta, grad_krr(ellipse(a,b,theta), model).reshape(-1))
    plt.show()
    '''

    def area_approx(R):
        """where R is an array of points representation of r(theta) over [0,2pi]"""
        return 0.5* sum(R**2) * step

    def grad_area_approx(R):
        """gradient of area_approx"""
        return R*step

    p, A_0 = 4, 9*np.pi/4
    def grad_area_penalty(r):
        """Gradient of the area penalty with respect to r."""
        return 2*p*(area_approx(r)-A_0)*grad_area_approx(r)

    """Perform NLGD"""
    start = time.time()

    a, b = 2, 4/3
    init_guess = ellipse(theta,a,b).reshape(1,-1)

    #Note: my implementation uses a multiprocessing Pool
    '''We would pass in v=grad_area_penalty, but the paper's algorithm did not include it, which may
    have been an error only in text and not in the actual method they used to produce the graphs.
    However, not including it gives the same graph as in the paper, which is strange because then the
    area A_0 is irrelevant, and the NLGD solver does not really work as stated.'''
    result = NLGD_KRR(init_guess, model, steps=120, eps=0.022, v=None, correction_step_interval=40).reshape(-1)
    end = time.time()
    print("Elapsed Time:", end - start)
    print(area_approx(result))
    plt.plot(theta, result)
    #plt.savefig("20_p_500steps_NLGD_xy_no_correction.png")
    plt.show()

    plt.polar(theta, result)
    #plt.savefig("20_p_500steps_NLGD_polar_no_correction.png")
    plt.show()
