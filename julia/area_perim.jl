# Summary: Functionals for area and perimeter in terms of r(θ)

using QuadGK

function twopi_integral(r::Function, args...)
  """Integrate r(θ)dθ over [0,2π]"""
  #Quad sum
  return quadgk(t->r(t, args...), 0, 2π, rtol=1e-5)[1]

  #handwritten rectangle summing
  
  #step = 0.01
  #return(step * sum(t->r(t, args...), 0:step:2π))
end

function area(w::Function, r::Function, args...)
  """Area functional for a radius and weight function
  Args:
    w: weight function as function of the angle, float->float,
    r: radius as a function of the angle, float->float
    args...: extra arguments to be passed into r

  Returns:
    area: float, area of the region
  """
  return 0.5 * twopi_integral(t->w(t)*r(t, args...)^2)
end

function perimeter(r_::Function, r_i::Function, args...)
  """Perimeter functional for a radius and weight function
  Args:
    r_: radius as a function of the angle, float->float
    r_i: first derivative of r as a function of the angle, float->float

  Returns:
    perimeter: float, perimeter of the region
  """
  return twopi_integral(t->sqrt( r_(t, args...)^2 + r_i(t, args...)^2 ))
end

function area_reg(r_::Function, r_i::Function, w::Function)
  """Regularizes the curve to have a perimeter of 1, and calculates
  the weighted area"""
  reg_factor = perimeter(r_, r_i)
  #r(theta) should be divided by the perimeter, and the r^2 factor in the area functional squares it
  return area(w, r_) / (reg_factor^2) 
end