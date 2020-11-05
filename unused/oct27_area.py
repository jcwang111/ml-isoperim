import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def integrate(r, *x):
    '''integral from 0 to 2pi of the given function times dtheta
        *x is any extra arguments for r'''
    return quad(lambda t: r(t, *x), 0, 2*np.pi)[0]
    #scipy.integrate.quad runs slower than this manual rectangle summing
    #theta = np.arange(0, (2*np.pi), 0.001)

    #return 0.001*np.sum(r(theta, *x))

def ellipse(theta, a, b):
    '''Polar function for an ellipse centered at the origin
    Args:
      theta: numpy array of floats, values for theta (preferably [0,2pi])
      a: coefficient for horizontal axis of ellipse
      b: coefficient for vertical axis of ellipse

    Returns:
      radius: numpy array of floats, radius output
    '''
    return a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)

def return_ellipse_reg(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg(theta):
        '''Polar function for an ellipse centered at the origin. Regularized'''
        radius = a*b/( (a*np.sin(theta))**2 + (b*np.cos(theta))**2 )**(1/2)
        return radius/reg_factor

    return ellipse_reg

def return_ellipse_reg_d1(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg_d1(theta):
        '''first derivative of r(theta) with respect to theta for the ellipse'''
        radius = (-1/2)*a*b*(a**2-b**2)*np.sin(2*theta)*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 )**(-3/2)
        return radius/reg_factor

    return ellipse_reg_d1

def return_ellipse_reg_d2(a, b):
    reg_factor = integrate(ellipse, a, b)
    def ellipse_reg_d2(theta):
        '''second derivative of r(theta) with respect to theta for the ellipse'''
        c = ( (b*np.cos(theta))**2 + (a*np.sin(theta))**2 ) # b^2cos^2(th) + a^2sin^2(th)
        radius = (-1)*a*b*(a**2-b**2)*np.cos(2*theta)* c**(-3/2) + (3/4)*a*b*(a**2-b**2)**2 * (np.sin(2*theta))**2 * c**(-5/2)
        return radius/reg_factor

    return ellipse_reg_d2

def perim(r, r_prime):
    '''Perimeter functional for a given r(theta)
    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      r_prime: a function, first derivative with respect to theta of r
    Returns:
      perimeter: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    return integrate(lambda t : np.sqrt( r(t)**2 + r_prime(t)**2 ))

def area(r, w):
    '''Area functional for a given r(theta).
    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      w: a function, the weight function. the first parameter should be theta
    Returns:
      area: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    
    return 0.5 * integrate( lambda t : r(t)**2 * w(t) )

def reg_area(r, r_prime, w):
  '''Returns the weighted area of a curve, after regularizing
  the perimeter to be 1.
  Args:
    r: r(t, *args), the polar curve
    r_prime: r'(t, *args), first derivative of r
    w: w(t), polar weight function to be used for the area
  Returns:
    area: double, the calculated area'''

  #For perimeter to be 1, r(theta) must be divided by the current perimeter. Since we have
  # r^2(theta) in A[r], we can simply divide the original A[r] by P^2
  return 0.5 * integrate_quad(lambda t : r(t)**2 * w(t)) / perim(r, r_prime)**2

def weight_function(r_, r_i, r_ii):
    '''derives w(theta) from the Euler-Lagrange equation, given r and its first and second derivative'''
    def w(t):
      return (r_(t)**2 + 2*r_i(t)**2 - r_(t)*r_ii(t)) / (r_(t)**2 + r_i(t)**2)**(3/2)
    
    return w

def weight_reg(r_, r_i, r_ii):
    '''weight function regularized to integrate to 1'''
    w = weight_function(r_, r_i, r_ii)
    reg_factor = integrate(w)
    return lambda t: w(t)/reg_factor

def plot_varied_ellipse():
    a,b = 2,5
    r_ = return_ellipse_reg(a,b)
    r_i = return_ellipse_reg_d1(a,b)
    r_ii = return_ellipse_reg_d2(a,b)
    w = weight_reg(r_, r_i, r_ii)
    #ellipse regularized for perimeter
    orig_perim = perim(r_, r_i)
    rp = lambda t: r_(t)/orig_perim
    rp_i = lambda t: r_i(t)/orig_perim
    print("Confirm that regularized perimeter is 1:", perim(rp, rp_i))

    #q = r_ε(t) = ellipse(t) + 0.1sin(t)
    q = lambda t:(r_(t) + 0.1*np.sin(t))
    q_i = lambda t:(r_i(t) + 0.1*np.cos(t))
    q_ii = lambda t:(r_ii(t) - 0.1*np.sin(t))
    orig_q_perim = perim(q, q_i)
    rq = lambda t:q(t)/orig_q_perim
    rq_i = lambda t:q_i(t)/orig_q_perim
    print("Confirm that regularized perimeter is 1:", perim(rq, rq_i))

    print('Area of original ellipse :', area(rp, w))
    print('Area of r_eps(theta)     :', area(rq, w))
    
    new_weight= weight_reg(q, q_i, q_ii)
    th = np.linspace(0, 2*np.pi, 628) #values of theta
    plt.polar(th, rp(th), label=r'Original Ellipse $r(\theta), a=2, b=5$')
    plt.polar(th, rq(th), label=r'$r(\theta) + 0.1\sin(\theta)$', color='green')
    #plt.polar(th, w(th), label = 'w(θ), the weight function', linestyle='dashed')
    #plt.polar(th, new_weight(th), label='Weight derived from q(θ)')
    #print('Area of q(θ) under its supposedly maximum weight:', area(rq, new_weight))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_varied_ellipse()