"""Attempts to use the BVP solver in 
DifferentialEquations.jl, when switching
the location of the weight function
from A[r] to P[r,w]."""

using QuadGK
using DifferentialEquations
using Plots

include("r_theta.jl")
include("alt_prob_testing.jl")

function el_diffq(w, w_prime, mu=1)
    """Generates and returns the expanded
    Euler-Lagrange differential
    equation for our weight function.
    Args:
        w: weight function, takes in a float and returns a float
        w_prime: first derivative of w
        mu: factor of mu for the weight function
    Returns:
        diffeq_sys: system of differential equations that
            can be plugged into a numerical solver
    """
    function diffeq_sys!(du,u,p,t)
        """The differential Equation we are trying to solve.
        Note that the system passed into the solver must mutate
        its arguments.
        Args:
            du: array, derivatives of u
            u: array, dependent variables
            p: array, parameters. not used here
            t: independent variable"""
        du[1] = u[2]
        du[2] = u[1] + 2*u[2]^2/u[1] - w_prime(t)*u[2]/w(t)*(1+u[2]^2/u[1]^2)- mu/w(t)/u[1]*(u[1]^2+u[2]^2)^(3/2)
    end

    return diffeq_sys!
end

function el_bvp_sol(w, w_prime, mu, u0, dt=0.02)
    """Solves and returns the solution to
    the BVP using derived from
    the Euler-Lagrange equation with the given weight function,
    using the GeneralMIRK4() method, then regularizes it to have
    a perimeter of 1.
    Args:
        w: weight function, takes in and returns a float
        mu: value of mu for the euler-lagrange
        u0: value of both r(0) and r(2pi)
    """
    tspan = (0.0, 2π)
    diffeq! = el_diffq(w, w_prime, mu)

    function bc!(residual, u, p, t)
        """boundary condition function, to be passed into the solver.
        The solver will attempt to fulfill the constraint of the
        residual being 0"""
        residual[1] = u[1][1] - u[end][1] #r(0) - r(2π)
        residual[2] = u[1][2] - u[end][2] #r'(0) - r'(2π)
    end

    bvp = BVProblem(diffeq!, bc!, [u0, 0], tspan)
    sol = solve(bvp, GeneralMIRK4(), dt=dt)

    return sol
end

function r0_est_1(w::Function)
    """Estimates a r(0) from the weight function"""
    return 1/w(0)
end

function r0_est_2(w::Function)
    """Estimates a r(0) from the weight function"""
    return 2π/integrate(w)
end

function sol_reg_area(sol, w)
    """Takes something returned from el_bvp_sol
    as an argument, and returns the area when
    regularized to have a perimeter of 1"""

    return area_reg(t->sol(t)[1], t->sol(t)[2], w)
end

function el_bvp_plot(w, w_prime, mu, u0, label="r(θ)", dt=0.02)
    """A function for quick testing. Solves and plots
    the BVP using derived from
    the Euler-Lagrange equation with the given weight function,
    using the GeneralMIRK4() method.
    Args:
        w: weight function, takes in and returns a float
        w_prime: first derivative of w(θ)
        mu: value of mu for the euler-lagrange
        u0: value of both r(0) and r(2pi)
    """
    sol = el_bvp_sol(w, w_prime, mu, u0, dt)

    reg_factor = 1#perimeter(t->sol(t)[1], t->sol(t)[2], w)
    p = plot(sol, proj=:polar)
    #plot!(p, sol.t, sol[1,:], label='1')
    #plot!(p, sol.t, sol[2,:], label='2')
    #sol.t, sol[1,:]./reg_factor
    return p, sol
end

function write_sol(sol, outfile)
    """Writes a DifferentialEquation sol object in
    a text file"""
    open(outfile, "w") do io
        write(io, join(sol.t, " ") * '\n')
        write(io, join(sol[1,:], " ") * '\n')
        write(io, join(sol[2,:], " "))
    end
end

function check_EL_eq(w, w_prime, sol)
  """Evaluates the left side of our euler-lagrange equation.
    It should be uniformly zero."""
  mu = 1
  r(t) = sol(t, idxs=1)[1]
  r_i(t) = sol(t, idxs=2)[1]
  r_ii(t) = sol(t, Val{1}, idxs=2)[1]
  theta = 0:0.008:2π
  resid(t) = w(t)*(r(t)^3 + 2*r(t)*r_i(t)^2 - r(t)^2*r_ii(t))/(r(t)^2+r_i(t)^2)^(3/2) - w_prime(t)*r_i(t)/(r(t)^2 + r_i(t)^2)^(1/2) - mu*r(t)

  p = plot(theta, resid.(theta)) #label="Residual (left side) of the Euler-Lagrange equation.", show=true)
  return p
end

function check_EL_eq_general(w, w_prime, r, r_i, r_ii)
    mu = 1
    theta = 0:0.008:2π
    resid(t) = w(t)*(r(t)^3 + 2*r(t)*r_i(t)^2 - r(t)^2*r_ii(t))/(r(t)^2+r_i(t)^2)^(3/2) - w_prime(t)*r_i(t)/(r(t)^2 + r_i(t)^2)^(1/2) - mu*r(t)
  
    p = plot(theta, resid.(theta), label="Residual (left side) of the Euler-Lagrange equation.", show=true)
    return p
end

#function main()

    n, eps, r0 = 8, 0.01, 1
    a, b = 2,5
    
    s_r = [smooth_reg(n, eps, r0), smooth_reg_d1(n, eps, r0), smooth_reg_d2(n, eps, r0)]
    e_r = [ellipse_reg(a,b), ellipse_reg_d1(a,b), ellipse_reg_d2(a,b)]
    t_r = [third_reg(), third_reg_d1(), third_reg_d2()]
    f_r = [fourth_reg(), fourth_reg_d1(), fourth_reg_d2()]

    smooth_weight = weight_reg(s_r...)
    ellipse_weight = weight_function(e_r...)
    third_weight = weight_reg(t_r...)
    fourth_weight = weight_reg(f_r...)

    ellipse_weight_d1 = weight_d1(e_r..., ellipse_reg_d3(a,b))
    t = 0:0.008:2π
    
    @time p,sol = el_bvp_plot(ellipse_reg(a,b), ellipse_reg_d1(a,b), 1, r0_est_1(ellipse_reg(a,b)), "Solver result for r(θ)", 0.02)
    #savefig(string("dt_0_01_ellipse_w_plot.png"))
    #e = check_EL_eq(ellipse_weight, ellipse_weight_d1, sol)
    println(sol)

    display(p)
    #display(e)
    #savefig("residual.png")
    #r(t) = sol(t, idxs=1)[1]
    #r_i(t) = sol(t, idxs=2)[1]
    #vary_r_and_check_area(r, r_i, ellipse_weight)
#end