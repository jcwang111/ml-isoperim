# Summary: Functions for r(theta) and for producing the w(theta)
#          that solves the Euler-Lagrange equation

using QuadGK

function ellipse(θ, a, b)
    """Polar function for an ellipse centered at the origin

    Args:
      theta: numpy array of floats, values for theta (preferably [0,2pi])
      a: coefficient for horizontal axis of ellipse
      b: coefficient for vertical axis of ellipse

    Returns:
      radius: numpy array of floats, radius output
    """
    return a*b/sqrt( (a*sin(θ))^2 + (b*cos(θ))^2 )
end

function smooth(θ, n, eps, r0)
    """Polar function for a smooth curve with an oscillating boundary

    Args:
      theta: numpy array of floats, values for theta (preferably [0,2pi])
      n: int, coefficient of theta, number of bumps
      epsilon: float, small number determining height of bumps
      r: base radius

    Returns:
      radius: numpy array of floats, radius output
    """
    return r0 * (1 + eps * cos(n*θ))
end
  
function third(θ)
    """A periodic polar graph"""
    return 1 + 0.1*cos(θ)*sin(2θ)
end

function integrate(r::Function, args...)
  """Integrate r(θ)dθ over [0,2π]"""
  #Quad sum
  return quadgk(t->r(t, args...), 0, 2π, rtol=1e-5)[1]

  #handwritten rectangle summing
  
  #step = 0.01
  #return(step * sum(t->r(t, args...), 0:step:2π))
  
end


"""Defining regularized r(theta) functions"""
##################################
"""The Julia language allows functions to be defined this way, on one line."""
function ellipse_reg(a, b)
  reg_factor = integrate(ellipse, a, b)
  return θ -> ( a*b/sqrt( (a*sin(θ))^2 + (b*cos(θ))^2 ) ) /reg_factor
end

function ellipse_reg_d1(a, b) 
  reg_factor = integrate(ellipse, a, b)
  return θ -> -(1/2)*a*b*(a^2-b^2)*sin(2θ)*( (b*cos(θ))^2 + (a*sin(θ))^2 )^(-3/2) / reg_factor
end

function ellipse_reg_d2(a, b)
  reg_factor = integrate(ellipse, a, b)
  return θ -> ( -a*b*(a^2-b^2)*cos(2θ)*(b^2*cos(θ)^2 + a^2*sin(θ)^2)^(-3/2) + (3/4)*a*b*(a^2-b^2)^2*sin(2θ)^2*(b^2*cos(θ)^2 + a^2*sin(θ)^2)^(-5/2)
                                                         ) /  reg_factor
end

function ellipse_reg_d3(a, b)
  reg_factor = integrate(ellipse, a, b)
  d = a^2 - b^2
  function f(θ)
    c = b^2*cos(θ)^2 + a^2*sin(θ)^2
    return ( 2*a*b*d*c^(-3/2)*sin(2θ) + (9/4)*a*b*d^2*c^(-5/2)*sin(4θ) - (15/8)*a*b*d^3*c^(-7/2)*sin(2θ)^3
                )  / reg_factor
  end
  return f
end
###################################
function smooth_reg(n::Int, eps, r0)
  reg_factor = integrate(smooth, n, eps, r0)
  return θ -> r0 * (1 + eps * cos(n*θ)) / reg_factor
end

function smooth_reg_d1(n::Int, eps, r0)
  reg_factor = integrate(smooth, n, eps, r0)
  return θ -> -n*r0*eps*sin(n*θ) / reg_factor
end

function smooth_reg_d2(n::Int, eps, r0)
  reg_factor = integrate(smooth, n, eps, r0)
  return θ -> -n^2*r0*eps*cos(n*θ) / reg_factor
end

function smooth_reg_d3(n::Int, eps, r0)
  reg_factor = integrate(smooth, n, eps, r0)
  return θ -> n^3*r0*eps*sin(n*θ) / reg_factor
end
###################################
function third_reg()
  reg_factor = integrate(third)
  return θ -> (1 + 0.1*cos(θ)*sin(2θ)) / reg_factor
end

function third_reg_d1()
  reg_factor = integrate(third)
  return θ -> (0.2*cos(θ)*cos(2θ) - 0.1*sin(θ)*sin(2θ)) / reg_factor
end

function third_reg_d2()
  reg_factor = integrate(third)
  return θ -> (-0.4*sin(θ)*cos(2θ) - 0.5*cos(θ)*sin(2θ)) / reg_factor
end

function third_reg_d3()
  reg_factor = integrate(third)
  return θ -> (-1.4*cos(θ)*cos(2θ) + 1.3*sin(θ)*sin(2θ)) / reg_factor
end
####################################
function fourth(θ)
  return 1 + 0.1*sin(3θ)
end

function fourth_reg()
  reg_factor = integrate(fourth)
  return θ -> (1 + 0.1*sin(3θ))/ reg_factor
end

function fourth_reg_d1()
  reg_factor = integrate(fourth)
  return θ -> 0.3*cos(3θ) / reg_factor
end

function fourth_reg_d2()
  reg_factor = integrate(fourth)
  return θ -> -0.9*sin(3θ)/ reg_factor
end

####################################
"""Where μ = 1"""
function weight_function(r_::Function, r_i::Function, r_ii::Function)
  """Returns the weight function that fulfills our
    Euler-Lagrange equation.
    Args: r_, r_i, r_ii: r(theta) and its first and second order derivatives
    Returns: Function, float -> float"""
    function w(θ)
      return ( r_(θ)^2 + 2*r_i(θ)^2 - r_(θ) * r_ii(θ) ) / ( r_(θ)^2 + r_i(θ)^2 )^(3/2)
    end
  end

"""Where μ is chosen to ensure integral[w] = 1"""
function weight_reg(r_::Function, r_i::Function, r_ii::Function)
    weight = weight_function(r_, r_i, r_ii)
    reg_factor = integrate(weight)
    #print(reg_factor, '\n')
    return θ -> weight(θ)/reg_factor
end

function weight_d1(r_, r_i, r_ii, r_iii)
  """First derivative of w(θ)"""
  function w_i(t)
      R = r_(t)
      R_i = r_i(t)
      R_ii = r_ii(t)
      R_iii = r_iii(t)
      return (-R^3*R_i - 4*R*R_i^3 - 3*R_i^3*R_ii - R*R_iii*(R^2+R_i^2) + 3*R*R_i*R_ii*(R+R_ii)) / (R^2 + R_i^2)^(5/2)
  end
  return w_i
end

function weight_reg_d1(r_, r_i, r_ii, r_iii)
    """First derivative of w(θ), regularized"""
    w = weight_function(r_, r_i, r_ii)
    w_i = weight_d1(r_, r_i, r_ii, r_iii)
    reg_factor = integrate(w)
    return t -> w_i(t)/reg_factor
end