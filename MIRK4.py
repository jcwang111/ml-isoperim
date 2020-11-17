'''A fourth-order mono-implicit Runge-Kutta solver.
Partly based off the MIRK4 solver in BoundaryValueDiffEq.jl, a Julia package.

Variable and function names based on, and the tableau originally from:
W. H. Enright and P. H. Muir. 1996. Runge-Kutta Software with Defect Control four Boundary Value ODEs.
    SIAM J. Sci. Comput. 17, 2 (March 1996), 479–497. DOI:https://doi.org/10.1137/S1064827593251496'''

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time
from math import ceil

def tableau():
    c = [0, 1, 1/2, 3/4]
    v = [0, 1, 1/2, 27/32]
    b = [1/6, 1/6, 2/3, 0]
    X = [[0,      0,     0,  0],
         [0,      0,     0,  0],
         [1/8,   -1/8,   0,  0],
         [3/64,  -9/64,  0,  0]]
         
    return np.array(c), np.array(v), np.array(b), np.array(X)

def jac_sparse(n, m):
    '''Sparsity structure of the Jacobian, if the
    Boundary conditions are u[0,0]=u[0,-1] and
    u[1,0]=u[1,-1], in that order. This greatly speeds
    up the solver.
    Args:
        n: number of components of the differential equation.
        m: N, number of time points'''
    sparsity = lil_matrix((n*(m-1)+2, n*m))
    for i in range(m-1):
        for c in range(n):
            sparsity[[comp*(m-1)+i for comp in range(n)], [c*m+i]] = 1
            sparsity[[comp*(m-1)+i for comp in range(n)], [c*m+i+1]] = 1

    sparsity[-2,[0,m-1]] = 1
    sparsity[-1,[m,-1]] = 1
    return sparsity

def MIRK(diffeq, bc, init_guess, tspan, dt, endpoint_bc=False, verbose=True):
    '''Uses 4rd order mono-implicit Runge-Kutta to solve 
    a system of differential equations.
    Args:
        diffeq: function(t,y) -> array of size (n,), where y is also (n,)
        bc: function(Y) -> array of size (n,). Y is the matrix with each row
            being values of a component of y.
        init_guess: list/array of size(n,) initial values of y to fill the mesh with
        tspan: (t0, tN), endpoints of the interval to solve for t on
        dt: float, step size. actual step size used will be close
        endpoint_bc: if bc() only depends on values at the endpoints, Y[:,0] and y[:,-1].
            If True, will use jac_sparse() as the sparsity structure, and the solver will
            run much faster.
        verbose: if True, messages will be printed
    Returns:
        Y_sol: array of shape (n+1,N). First row is time points of the mesh,
            and the remaining rows are values for those time points.
        t_mesh: array of shape(N), containing the N time points used.
    '''
    #n: number of components of diffeq
    n = len(init_guess)

    #initialize t_mesh, get num of steps (N) and step size (h)
    assert dt < tspan[-1]-tspan[0], 'Error: Step size must be smaller than interval.'
    N = int(ceil((tspan[-1]-tspan[0])/dt)) + 1
    t_mesh, h = np.linspace(tspan[0], tspan[1], num=N, retstep=True)

    if verbose:
        print('Using {} steps over the interval {}.\nStep size:{}'.format(N-1, tspan, h))

    #Initialize Y mesh with values
    y_init = np.array(init_guess)
    Y = np.tile(y_init, (N,1)).T
    Y_shape = Y.shape
    y_guess = Y.ravel()
    #print(Y)

    #Ensure diffeq and bc return the right number of values
    assert (diffeq(0, y_init)).shape == (n,)
    assert bc(Y).shape == (n,)

    #Get tableau
    c, v, b, X = tableau()
    f = diffeq

    #Initialize K mesh and Φ residual
    order = len(b)
    K = np.zeros((n, N-1, order)) #K matrix for 4th order RK
    Φ = np.zeros((n, N-1)) #Φ carries residuals for the implicit RK equations of each step
    if verbose:
        print('Starting RK process of order {}.'.format(order))

    #Mutating isn't "Pythonic", but I have found it the fastest when dealing with numpy arrays
    def update_K(Y):
            for i in range(N-1):
                for r in range(order):
                    K[:,i,r] = f(t_mesh[i] + c[r]*h, (1-v[r])*Y[:,i] + v[r]*Y[:,i+1] + h*(K[:,i,:r]@X[r,:r]))

    def update_Φ(Y):
        update_K(Y)
        for m in range(n):
            for i in range(N-1):
                Φ[m,i] = Y[m,i+1] - Y[m,i] -h*(b@K[m,i,:])
    
    def fun(y):
        Y = np.reshape(y, Y_shape)
        update_Φ(Y)
        return np.concatenate((Φ.ravel(), bc(Y)))


    if verbose:
        print('Starting minimization.')
        clock = time.process_time()

    #Call scipy.optimize.least_squares()
    sol = least_squares(fun, y_guess, method='trf', verbose=2 if verbose else 0,
                        jac_sparsity=jac_sparse(n,N) if endpoint_bc else None, tr_solver='lsmr')
    Y_sol = np.reshape(sol.x, Y_shape)

    if verbose:
        print("Elapsed time: {} seconds.".format(time.process_time() - clock))

    return Y_sol, t_mesh
        
if __name__ == '__main__':
    """Test on a function"""
    from numpy import pi
    import matplotlib.pyplot as plt

    g = 9.81
    L = 1.0
    tspan = (0.0,pi/2)
    def simplependulum(t, y):
        return np.array([
                y[1],            #= du[0]
                -(g/L)*np.sin(y[0]) #= du[1]
            ])

    def bc_pendulum(Y): # Y[:,0] is the beginning of the time span, and Y[:,-1] is the ending
        middle = Y.shape[1]//2
        return np.array([Y[0, middle] + pi/2, # the solution at the middle of the time span should be -pi/2
                            Y[0,-1] - pi/2  # the solution at the end of the time span should be pi/2
            ])

    y, t = MIRK(simplependulum, bc_pendulum, [pi/2, pi/2], tspan, 0.05, endpoint_bc=False, verbose=True)

    plt.plot(t, y[0,:], label='y0')
    plt.plot(t, y[1,:], label='y1')
    plt.legend()
    plt.show()
    