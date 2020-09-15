'''My own implementations of linear and cubic Hermite interpolation.
    Moved to "unused" folder: if doing cubic interpolation,
    use scipy.interpolate.CubicHermiteSpline() instead.'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from multiprocessing.pool import Pool

def select_points(A, num_points, randomly = True, return_endpoints = True):
    """Return num_points columns from A, chosen either randomly or evenly spaced
      
    Args:
      A: numpy matrix with m rows and n columns
      num_points: int, number of columns to return. May return less columns 
                  if randomly is False
      randomly: bool, whether the columns are chosen randomly. If False,
                points are evenly spaced based on num_points
      return_endpoints: bool, if True, force endpoints to be returned


    Returns:
      points: matrix of shape (m, num_points)
    """

    assert not (return_endpoints and num_points < 2)
    assert num_points > 0
    
    if randomly:
      if not return_endpoints:
        mask = np.array(sorted(random.sample(range(A.shape[1]), num_points)))
      else: #if return_endpoints
        mask = np.hstack((np.array([0]), \
                          np.array(sorted(random.sample(range(1,A.shape[1]-1), num_points-2))), \
                          np.array([-1]) 
                        ))
      return A[:,mask]

    else: #if not randomly
      points = A[:,np.array(range(0, A.shape[1]-1, A.shape[1]//num_points))]
      if not return_endpoints:
        return points
      else: #if return_endpoints
        return np.hstack((points, np.resize(A[:,-1], (3,1))))


def linear_line_interpolator(p1, p2, x_range):
  '''Uses the slopes and positions of two points to interpolate two connected line segments
     between them, producing y-values based on the passed in range
     Args:
      p1, p2: numpy arrays with shape (3,), describing 2D points as [x,y,dy_dx]
      x_range: numpy array with the domain of x-values on which to find y on
     Returns:
      result: array of the same length as x_range, with the corresponding y-values
  '''

  x1, y1, m1 = p1[0],p1[1],p1[2]
  x2, y2, m2 = p2[0],p2[1],p2[2]

  #Found by solving the two equations y-y_i = m_i(x-x_i) for i in {1,2}
  x_intersect = (y1 - y2 + m2*x2 - m1*x1) / (m2 - m1)

  result = np.zeros((x_range.size,))

  if x_intersect >= x1 and x_intersect <= x2:
    result[x_range <= x_intersect] = y1 + m1*(x_range[x_range <= x_intersect] - x1)
    result[x_range > x_intersect] = y2 + m2*(x_range[x_range > x_intersect] - x2)
  else:
    '''interpolation fails because the tangent lines don't intersect between the two points (ie both slopes are 0),
       so we just draw a line between the two points'''
    result[:] = y1 + ((y2-y1)/(x2-x1)) * (x_range - x1)

  return result
  
def line_interpolate(A, x_range):
  '''Perform linear interpolation on the values in A, produces y values for x_range
    Args:
      A: matrix with shape (3,m), where m is the number of points, and 
        each column being a point [x,y,dy_dx]. 
        The first row must be ascending.
      x_range: numpy array of floats, x-values input

    Returns:
      y: numpy array of floats, y-values for the interpolated graph
  '''

  #assert first row of A is sorted descending to ascending
  assert all(A[0,i] < A[0,i+1] for i in range(A.shape[1]-1))

  y = np.zeros(x_range.size)

  #Iterate through every pair of points
  #Break up for multiprocessing
  interp_list = []
  for i in range(A.shape[1]-1):
    p1, p2 = A[:,i], A[:,i+1]
    mask = (x_range >= p1[0]) & (x_range < p2[0])

    interp_list.append((p1,p2,x_range[mask]))
  #Interpolate y for the x values between those two points, using
  # a multiprocessing pool for each interpolation
  interp_list = Pool().starmap(linear_line_interpolator, interp_list)

  #set our y to the interpolated values
  p_left = A[:,0]
  p_right = A[:,-1]
  y[(x_range >= p_left[0]) & (x_range < p_right[0])] = np.hstack(interp_list)

  #Past the endpoints of A, continue the line with their slop until they hit 0
  p_left_intercept = np.array([p_left[0]-p_left[1]/p_left[2], 0, p_left[2]])
  mask_l = (x_range < p_left[0]) & (x_range >= p_left_intercept[0])
  y[mask_l] = linear_line_interpolator(p_left_intercept,p_left, x_range[mask_l])

  p_right_intercept = np.array([p_right[0]-p_right[1]/p_right[2], 0, p_right[2]])
  mask_r = (x_range >= p_right[0]) & (x_range <= p_right_intercept[0])
  y[mask_r] = linear_line_interpolator(p_right, p_right_intercept, x_range[mask_r])

  return y

def cubic_hermite_interpolator(p0, p1, x_range):
  '''Uses the slopes and positions of two points to interpolate a cubic polynomial between
     them and producing the corresponding y-values for x_range

     Args:
      p0, p1: numpy arrays with shape (3,), describing 2D points as [x,y,dy_dx]
      x_range: numpy array of floats, x-values to be passed in

     Returns:
      y: numpy array of floats, y(x) for each x in x_range
  '''

  x0, y0, m0 = p0[0],p0[1],p0[2]
  x1, y1, m1 = p1[0],p1[1],p1[2]

  '''with two equations y = ax^3 + bx^2 + cx + d and y' = 3ax^2 + 2bx + c for two sets of (x,y) and (x,y'), we have
     a set of four linear equations that can be solved for a, b, c, and d'''
  y = np.array([y0,y1,m0,m1])

  X = np.array([[  x0**3,   x0**2, x0, 1],
                [  x1**3,   x1**2, x1, 1],
                [3*x0**2, 2*x0,     1, 0],
                [3*x1**2, 2*x1,     1, 0]])

  '''X @ c = y'''
  coeffs = np.linalg.inv(X) @ y

  X = np.vstack((x_range**3, x_range**2, x_range, np.ones(x_range.shape)))
  return coeffs @ X

def piecewise_cubic(A, x_range):
  '''Perform piecewise cubic Hermite interpolation on the values in A, produces y values for x_range
    Args:
      A: matrix with shape (3,m), where m is the number of points, and 
        each column being a point [x,y,dy_dx]. 
        The first row must be from descending to ascending.
      x_range: numpy array of floats, x-values input
    Returns:
      y: numpy array of floats, y-values for the interpolated graph
  '''

  #assert first row of A is sorted descending to ascending
  assert all(A[0,i] < A[0,i+1] for i in range(A.shape[1]-1))

  y = np.zeros(x_range.size)

  #Iterate through every pair of points
  #Break up for multiprocessing
  interp_list = []
  for i in range(A.shape[1]-1):
    p1, p2 = A[:,i], A[:,i+1]
    mask = (x_range >= p1[0]) & (x_range < p2[0])
    interp_list.append((p1,p2,x_range[mask]))
  #Interpolate y for the x values between those two points, using
  # a multiprocessing pool for each interpolation
  interp_list = Pool().starmap(cubic_hermite_interpolator, interp_list)

  #set our y to the interpolated values
  p_left = A[:,0]
  p_right = A[:,-1]
  y[(x_range >= p_left[0]) & (x_range < p_right[0])] = np.hstack(interp_list)

  return y
