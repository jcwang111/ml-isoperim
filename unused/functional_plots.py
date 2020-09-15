# Summary: Graphs of P[r] and A[r] for our parameterized equations.
#          Python 3.7
#
# Notes: Written on 2020-07-02. Here, we want to graph A[r] and P[r] for the ellipse and smooth
#               curves, while varying their parameters, a and b for the ellipse, and n, r_0, and ε for the 
#               smooth curve.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from r_theta import *

def perim(r, r_prime, *args):
    '''Perimeter functional for a given r(theta)

    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      r_prime: a function, first derivative with respect to theta of r
      args: positional arguments, after theta, to be passed into r and r_prime

    Returns:
      perimeter: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''
    return integrate(lambda t : np.sqrt( r(t, *args)**2 + r_prime(t, *args)**2 ))

def area(r, w, *args):
    '''Area functional for a given r(theta). Normalization
      of w(theta) is included!

    Args:
      r: a function, r(theta) for the curve. the first parameter must be theta
      w: a function, the weight function. the first parameter should be theta
      args: positional arguments, after theta, to be passed into r and r_prime

    Returns:
      area: the calculated value of the perimeter for the curve for theta in [0,2pi)
    '''

    mu = 1/integrate(w)
    
    return 0.5 * integrate( lambda t : mu * r(t, *args)**2 * w(t) )

def p_loc(r, *args):
    '''Local approximation for the perimeter functional'''
    return integrate(r, *args)

def p_gea(r, r_prime, *args):
  '''Gradient expansion approximation for the perimeter functional'''
  return p_loc(r, *args) + (1/2) * integrate(lambda t : r_prime(t,*args)**2/r(t,*args))

def ellipse_perim_graph():
  fig = plt.figure()
  a = np.arange(1, 101, 1)
  b = np.arange(1, 101, 1)
  a_grid, b_grid = np.meshgrid(a, b)
  Z = np.zeros((100,100))

  for i in range(100):
    for j in range (100):
      Z[i,j] = p_gea(ellipse_reg, ellipse_reg_d1, a_grid[i,j], b_grid[i,j])

  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(a_grid, b_grid, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=1.0, aspect=5)

  ax.set_xlabel('a')
  ax.set_ylabel('b')
  ax.set_zlabel('ellipse perimeter')

  plt.savefig('ellipse_perim_gea.png')
  plt.show()

def ellipse_area_graph():
  fig = plt.figure()
  a = np.arange(1, 101, 1)
  b = np.arange(1, 101, 1)
  a_grid, b_grid = np.meshgrid(a, b)
  Z = np.zeros((100,100))

  for i in range(100):
    for j in range (100):
      w = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a_grid[i,j], b_grid[i,j])
      Z[i,j] = area(ellipse_reg, w, a_grid[i,j], b_grid[i,j])

  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(a_grid, b_grid, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=1.0, aspect=5)

  ax.set_xlabel('a')
  ax.set_ylabel('b')
  ax.set_zlabel('ellipse area')

  plt.savefig('ellipse_area_exact.png')
  plt.show()

def ellipse_perim_graph_one_axis():
  fig = plt.figure()
  a = np.arange(1, 101, 1)
  b = 50
  Z = np.zeros(100)
  G = np.zeros(100)

  for i in range(100):
      Z[i] = perim(ellipse_reg, ellipse_reg_d1, a[i], b)
      G[i] = p_gea(ellipse_reg, ellipse_reg_d1, a[i], b)

  plt.plot(a, Z, label='P_exact')
  plt.plot(a, G, label='P_gea')

  plt.xlabel('a')
  plt.ylabel('ellipse perim')
  plt.legend()

  plt.savefig('ellipse_perim_exact_gea.png')
  plt.show()

def ellipse_area_graph_one_axis():
  fig = plt.figure()
  a = np.arange(1, 101, 1)
  b = 50
  Z = np.zeros(100)

  for i in range(100):
      w = weight_function(ellipse_reg, ellipse_reg_d1, ellipse_reg_d2, a[i], b)
      Z[i] = area(ellipse_reg, w, a[i], b)

  plt.plot(a, Z)

  plt.xlabel('a')
  plt.ylabel('ellipse area')

  plt.savefig('ellipse_area_one_axis.png')
  plt.show()

def smooth_r0_graph_one_axis():
  fig = plt.figure()
  r0 = np.arange(1, 101, 1)
  eps = 0.1
  n = 30
  Area = np.zeros(100)
  Perimeter = np.zeros(100)

  for i in range(100):
      w = weight_function(smooth_reg, smooth_reg_d1, smooth_reg_d2, n, eps, r0[i])
      Perimeter[i] = perim(smooth_reg, smooth_reg_d1, n, eps, r0[i])
      Area[i] = area(smooth_reg, w, n, eps, r0[i])

  plt.plot(r0, Perimeter, label ='P[r]')
  #plt.plot(r0, Area, label = 'A[r]')

  plt.xlabel('r0')

  plt.savefig('smooth_r0_graph_one_axis.png')
  plt.show()

def smooth_perim_graph():
  fig = plt.figure()
  eps = np.arange(0, 1, 0.01)
  n = np.arange(1, 101, 1)
  eps_grid, n_grid = np.meshgrid(eps, n)
  Perim = np.zeros((100,100))

  for i in range(100):
    for j in range (100):
      Perim[i,j] = p_gea(smooth_reg, smooth_reg_d1, int(n_grid[i,j]), eps_grid[i,j], 1)

  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(eps_grid, n_grid, Perim, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=1.0, aspect=5)

  ax.set_xlabel('ε')
  ax.set_ylabel('n')
  ax.set_zlabel('perimeter')

  plt.savefig('smooth_perim_gea.png')
  plt.show()

def smooth_perim_graph_one_axis_vary_eps():
  fig = plt.figure()
  eps = np.arange(0, 1, 0.01)
  n = 10
  Perim = np.zeros(100)
  Pgea = np.zeros(100)

  for i in range(100):
      Perim[i] = perim(smooth_reg, smooth_reg_d1, n, eps[i], 1)
      Pgea[i] = p_gea(smooth_reg, smooth_reg_d1, n, eps[i], 1)

  plt.plot(eps, Perim, label='P_exact')
  plt.plot(eps, Pgea, label='P_gea')

  plt.xlabel('ε')
  plt.ylabel('perimeter')
  plt.legend()
  plt.title('Perimeter of smooth curve, n=10')

  plt.savefig('smooth_perim_exact_gea_vary_eps.png')
  plt.show()

def smooth_perim_graph_one_axis():
  fig = plt.figure()
  eps = 0.05
  n = np.arange(1, 101, 1)
  Perim = np.zeros(100)
  Pgea = np.zeros(100)

  for i in range(100):
      Perim[i] = perim(smooth_reg, smooth_reg_d1, int(n[i]), eps, 1)
      Pgea[i] = p_gea(smooth_reg, smooth_reg_d1, int(n[i]), eps, 1)

  plt.plot(n, Perim, label='P_exact')
  plt.plot(n, Pgea, label='P_gea')

  plt.xlabel('n')
  plt.ylabel('perimeter')
  plt.legend()
  plt.title('Perimeter of smooth curve, ε=0.05')

  plt.savefig('smooth_perim_exact_gea.png')
  plt.show()

def smooth_area_graph():
  fig = plt.figure()
  eps = np.arange(0, 1, 0.01)
  n = np.arange(1, 101, 1)
  eps_grid, n_grid = np.meshgrid(eps, n)
  Area = np.zeros((100,100))

  for i in range(100):
    for j in range (100):
      print(i, j)
      w = weight_function(smooth_reg, smooth_reg_d1, smooth_reg_d2, int(n_grid[i,j]), eps_grid[i,j], 1)
      Area[i,j] = area(smooth_reg, w, int(n_grid[i,j]), eps_grid[i,j], 1)

  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(eps_grid, n_grid, Area, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=1.0, aspect=5)

  ax.set_xlabel('ε')
  ax.set_ylabel('n')
  ax.set_zlabel('area')

  plt.savefig('smooth_area_exact.png')
  plt.show()

def smooth_area_graph_one_axis():
  fig = plt.figure()
  eps = 0.05
  n = np.arange(1, 101, 1)
  Area = np.zeros(100)

  for i in range(100):
      w = weight_function(smooth_reg, smooth_reg_d1, smooth_reg_d2, int(n[i]), eps, 1)
      Area[i] = area(smooth_reg, w, int(n[i]), eps, 1)

  plt.plot(n, Area, label='Area')

  plt.xlabel('n')
  plt.ylabel('Area')
  plt.legend()
  plt.title('Area of smooth curve, ε=0.05')

  plt.savefig('smooth_area_vary_n.png')
  plt.show()

def smooth_area_graph_one_axis_vary_eps():
  fig = plt.figure()
  eps = np.arange(0, 1, 0.01)
  n = 10
  Area = np.zeros(100)

  for i in range(100):
      w = weight_function(smooth_reg, smooth_reg_d1, smooth_reg_d2, n, eps[i], 1)
      Area[i] = area(smooth_reg, w, n, eps[i], 1)

  plt.plot(eps, Area, label='Area')

  plt.xlabel('ε')
  plt.ylabel('Area')
  plt.legend()
  plt.title('Area of smooth curve, n=10')

  plt.savefig('smooth_area_vary_eps.png')
  plt.show()

if __name__ == '__main__':
  smooth_area_graph_one_axis()