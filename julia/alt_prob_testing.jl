# Summary: Functionals for area and perimeter in terms of r(θ)

using QuadGK

function twopi_integral(r::Function, args...)
  """Integrate r(θ)dθ over [0,2π]"""
  #Quad sum
  return quadgk(t->r(t, args...), 0, 2π, rtol=1e-12)[1]

  #handwritten rectangle summing
  
  #step = 0.01
  #return(step * sum(t->r(t, args...), 0:step:2π))
end

function area(r::Function)
  return 0.5 * twopi_integral(t->r(t)^2)
end

function perimeter(r_::Function, r_i::Function, w::Function)
  return twopi_integral(t-> w(t)*sqrt( r_(t)^2 + r_i(t)^2 ))
end

function area_reg(r_::Function, r_i::Function, w::Function)
  """Regularizes the curve to have a weighted perimeter of 1, and calculates
  the area"""
  reg_factor = perimeter(r_, r_i, w)
  #r(theta) should be divided by the perimeter, and the r^2 factor in the area functional squares it
  return area(r_) / (reg_factor^2) 
end

function vary_r_and_check_area(r_, r_i, w)
  """Varies r(theta) and checks the new area.
  The area from the exact r should be greater
  than any of the varied r's.
  Args:
      r_, r_i: r(theta) and first derivative
      w: weight function for the area
      args: extra args for r_ and r_i"""
  exact_area = area_reg(r_, r_i, w)
  println("Exact area:"*string(exact_area))
  #rotate
  for φ = 0:0.01:2π
      rotated_r = t-> r_(mod2pi(t+φ))
      rotated_r_i = t-> r_i(mod2pi(t+φ))
      new_area = area_reg(rotated_r, rotated_r_i, w)
      println("Rotation by "*string(φ)*"-> area: "*string(new_area))
      if (new_area > exact_area)
          println("New area > exact_area")
      end
  end
  #add epsilon*sin(t) to r(t)
  for ε = 0.005:0.002:0.3
      new_r = t-> r_(t) + ε*sin(t)
      new_r_i = t-> r_i(t) + ε*cos(t)
      new_area = area_reg(new_r, new_r_i, w)
      println("Adding"*string(ε)*"sin(t) -> area: "*string(new_area))
      if (new_area > exact_area)
          println("New area > exact_area")
          #perimeter = perim(new_r, new_r_i)
          #new_new_r = lambda t: new_r(t) / perimeter
          #new_new_r_i = lambda t: new_r_i(t) / perimeter
          #print('Actual regularized perimeter:', perim(new_new_r, new_new_r_i))
      end
  end
end

function vary_r_and_check_area_2(r_, r_i, w)
  """More tests"""
  exact_area = area_reg(r_, r_i, w)
  println("Exact area:"*string(exact_area))
  #add epsilon*10
  for ε = 0.005:0.002:0.3
      new_r = t-> r_(t) + ε*10
      new_r_i = t-> r_i(t)
      new_area = area_reg(new_r, new_r_i, w)
      println("Adding"*string(ε)*"10 -> area: "*string(new_area))
      if (new_area > exact_area)
          println("New area > exact_area")
          #perimeter = perim(new_r, new_r_i)
          #new_new_r = lambda t: new_r(t) / perimeter
          #new_new_r_i = lambda t: new_r_i(t) / perimeter
          #print('Actual regularized perimeter:', perim(new_new_r, new_new_r_i))
      end
  end
end

function vary_sol_and_check_area(sol, w)
  """Same as above function, but pass in
  a solution object instead."""
  r(t) = sol(t, idxs=1)[1]
  r_i(t) = sol(t, idxs=2)[1]
  vary_r_and_check_area(r, r_i, w)
end

function vary_sol_and_check_area_2(sol, w)
  """Same as above function, but pass in
  a solution object instead."""
  r(t) = sol(t, idxs=1)[1]
  r_i(t) = sol(t, idxs=2)[1]
  vary_r_and_check_area_2(r, r_i, w)
end