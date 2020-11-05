'''Since krr2.py has so many functions, I decided to place
all the tests in this file.'''

from krr2 import *

def accuracy_test(model, m):
    '''Generates data and tests our model's accuracy with a graph.'''
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

    y_test = model.predict(X_test)
    plt.plot(y_exact_test, 'x', label='Exact values')
    plt.plot(y_test, '+', label='Predicted values')
    plt.legend()
    #plt.savefig('sep15_perim_comparisons.png')
    plt.show()

def grad_descent_test(model):
    '''Tests naive gradient descent on our KRR model.'''
    init_r = generate_data(return_ellipse(2, 3), n=21, feature='fourier')
    r_result = perform_gradient_descent(init_r, penalty_func, grad_penalty_krr_fourier, 3200, (model, 10, area_fourier, 2*np.pi), eps=10E-5, produce_graph=True)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    abs_r_series = lambda t: abs(r_series(t))
    plt.polar(t, abs_r_series(t),label=r"First attempt at minimization for $A[r]=1$")
    plt.legend()
    #plt.savefig('sep14_test1.png')
    plt.show()

def modified_gd_test(model):
    '''Test the modified gradient descent algorithm.'''
    init_r = generate_data(return_ellipse(1, 2), n=21, feature='fourier')
    r_result = modified_grad_descent(init_r, model, 1600, 10E-5, 1)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    plt.polar(t, r_series(t),label="Attempt minimization with PCA gradient correction for A[r]=1")
    #plt.savefig('sep14_test1.png')
    plt.ylim(top=5)
    plt.show()

def NLGD_test(model):
    init_r = generate_data(return_ellipse(2, 2), n=21, feature='fourier')
    r_result = NLGD_krr_fourier(init_r, model, 25, 10E-3)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    plt.polar(t, r_series(t),label="Attempt minimization with PCA gradient correction for A[r]=1")
    plt.ylim(top=5)
    plt.show()

if __name__ == '__main__':
    scipy.special.seterr(all='raise')
    '''Train the model'''
    #Let's first try a data set of the first 21 sine-cosine Fourier coefficients, for various ellipses
    m = 21 #Length of each feature vector; columns of X
    p = 1 #Number of dependent variables

    param_list = [1, 4/3, 5/3, 2, 4, 5]
    n = 2*len(param_list)**2 #number of feature vectors; rows of X

    X = np.zeros((n, m)) #Contains feature vectors
    y = np.zeros(n) #Values of P[r,w], dependent variables

    weight = return_ellipse(2, 5)
    for (i, (a, b)) in enumerate(itertools.product(param_list, param_list)):
        r_func = return_ellipse(a, b)
        r_func_d1 = return_ellipse_d1(a, b)
        X[2*i,:] = generate_data(r_func, n=21, feature='fourier')
        y[2*i] = perim(r_func, r_func_d1, weight)
        ### Add to the training data r_eps(t) = r_ellipse(t) + 0.1sin(t)
        q_func = lambda t: (r_func(t) + 0.1*np.sin(t))
        q_func_d1 = lambda t: (r_func_d1(t) + 0.1*np.cos(t))
        X[2*i+1,:] = generate_data(q_func, n=21, feature='fourier')
        y[2*i+1] = perim(q_func, q_func_d1, weight)

    alpha_values = [0, 10E-3, 10E-2, 10E-1, 1, 10, 100, 1000]
    gamma_values = [10E-3, 10E-2, 10E-1, 1, 10, 100] #γ = 1/(2σ)^2 in the RBF documentation 
    alpha, gamma = cross_evaluate_krr(X, y, alpha_values, gamma_values)

    '''The data points will be the A[r] after normalization as well as w(theta)
    in some representation.'''
    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    model.fit(X, y)

    accuracy_test(model, m)
    #grad_descent_test(model)
    #modified_gd_test(model)

    '''Try NLGD'''
    #NLGD_test(model)
    '''Results: for NLGD, stuck at np.linalg.eig(H(r)), which is supposed to find the eigenvalues of the
    Hessian. It says something about a mismatch between dimensions 10 and 21'''