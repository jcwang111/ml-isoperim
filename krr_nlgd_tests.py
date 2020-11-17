"""File that tests the functions in krr3.py. This file and that one
attempt to minimize P[r] for a given A[r,w]."""

from krr_nlgd import *

def accuracy_test(model, m):
    """Generates data and tests our model's accuracy with a graph."""
    test_param_list = [1.1, 1.4, 1.7, 3, 6]
    n_test = 2*len(test_param_list)**2 #number of feature vectors; rows of X

    X_test = np.zeros((n_test, m)) #Contains feature vectors
    y_exact_test = np.zeros(n_test) #Values of P[r,w], dependent variables

    for (i, (a, b)) in enumerate(itertools.product(test_param_list, test_param_list)):
        r_func = return_ellipse(a, b)
        r_func_d1 = return_ellipse_d1(a, b)
        X_test[2*i,:] = generate_data(r_func, n=m, feature='fourier')
        y_exact_test[2*i] = perim(r_func, r_func_d1)
        #Also do the varied ellipses, r_epsilon
        q_func = lambda t: (r_func(t) + 0.1*np.sin(t))
        q_func_d1 = lambda t: (r_func_d1(t) + 0.1*np.cos(t))
        X_test[2*i+1,:] = generate_data(q_func, n=m, feature='fourier')
        y_exact_test[2*i+1] = perim(q_func, q_func_d1)

    y_test = model.predict(X_test)
    error_percent = 100* np.divide(y_test - y_exact_test, y_exact_test)
    plt.bar(np.arange(error_percent.size), error_percent)
    plt.xlabel('Test Case')
    plt.ylabel('% Error from Exact')
    
    plt.savefig('krr3_model_prediction_error.png')
    plt.show()
    
def NLGD_test(model, m, area_weight_func):
    init_r = generate_data(return_ellipse(2, 2), n=m, feature='fourier').reshape(1,-1)
    r_result = NLGD_krr_fourier(init_r, model, 50, 1, area_weight_func).reshape(-1)
    print("Final Result for r:", r_result)
    r_series = fourier_series(r_result)

    t = np.linspace(0, 2*np.pi, 300)
    plt.polar(t, r_series(t),label="Attempt minimization with PCA gradient correction for A[r]=1")
    plt.ylim(top=5)
    plt.show()

if __name__ == '__main__':
    scipy.special.seterr(all='raise')
    """Train the model"""
    #Let's first try a data set of the first 21 sine-cosine Fourier coefficients, for various ellipses
    m = 15 #Length of each feature vector; columns of X
    p = 1 #Number of dependent variables

    param_list = [1, 4/3, 5/3, 2, 4, 5]
    n = len(param_list)**2 #number of feature vectors; rows of X

    X = np.zeros((n, m)) #Contains feature vectors
    y = np.zeros(n) #Values of P[r,w], dependent variables

    a, b = 2, 5
    weight = weight_reg(return_ellipse_reg(a, b), return_ellipse_reg_d1(a, b), return_ellipse_reg_d2(a, b))
    for (i, (a, b)) in enumerate(itertools.product(param_list, param_list)):
        r_func = return_ellipse(a, b)
        r_func_d1 = return_ellipse_d1(a, b)
        X[i,:] = generate_data(r_func, n=m, feature='fourier')
        y[i] = perim(r_func, r_func_d1)
        ### Add to the training data r_eps(t) = r_ellipse(t) + 0.1sin(t)
        """q_func = lambda t: (r_func(t) + 0.1*np.sin(t))
        q_func_d1 = lambda t: (r_func_d1(t) + 0.1*np.cos(t))
        X[2*i+1,:] = generate_data(q_func, n=21, feature='fourier')
        y[2*i+1] = perim(q_func, q_func_d1)"""

    #alpha_values = [0, 10E-3, 10E-2, 10E-1, 1, 10, 100, 1000, 10000]
    #gamma_values = [10E-3, 10E-2, 10E-1, 1, 10, 100, 1000] #γ = 1/(2σ^2) in the RBF documentation 
    #alpha, gamma = cross_evaluate_krr(X, y, alpha_values, gamma_values)
    alpha, gamma = 0, 0.01 #Found with cross evaluation of above
    print('Alpha:', alpha)
    print('Gamma:', gamma)

    """The data points will be the A[r] after normalization as well as w(theta)
    in some representation."""
    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    model.fit(X, y)

    #accuracy_test(model, m)

    """Try NLGD"""
    NLGD_test(model, m, weight)