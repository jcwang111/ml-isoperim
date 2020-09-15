"""Attempts to use the BVP solver in 
DifferentialEquations.jl, in the
hopes that it may get something different."""

using QuadGK
using DifferentialEquations
using Plots

include("r_theta.jl")
include("area_perim.jl")

function el_diffq(w, mu=1)
    """Generates and returns the expanded
    Euler-Lagrange differential
    equation for our weight function.
    Args:
        w: weight function, takes in a float and returns a float
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
        du[2] = u[1] + 2*u[2]^2/u[1] - (w(t)/(mu*u[1]))*(u[1]^2 + u[2]^2)^(3/2)
    end

    return diffeq_sys!
end

function el_bvp_sol(w, mu, u0, dt=0.02)
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
    diffeq! = el_diffq(w, mu)

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

function el_bvp_plot(w, mu, u0, label="r(θ)", dt=0.02)
    """A function for quick testing. Solves and plots
    the BVP using derived from
    the Euler-Lagrange equation with the given weight function,
    using the GeneralMIRK4() method.
    Args:
        w: weight function, takes in and returns a float
        mu: value of mu for the euler-lagrange
        u0: value of both r(0) and r(2pi)
    """
    sol = el_bvp_sol(w, mu, u0, dt)
    #print(sol(0), sol(2π), '\n')
    print(sol_reg_area(sol,w),'\n')
    reg_factor = integrate(t->sol(t)[1])

    p = plot(sol.t, sol[1,:]./reg_factor, proj=:polar, label=label, lw=2)
    
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

function sol_check(sol)
    """Uses the exact equation for the weight
    function to check a solution"""
    r_(t) = sol(t, idxs=1)[1]
    r_i(t) = sol(t, idxs=2)[1]
    r_ii(t) = sol(t, Val{1}, idxs=2)[1]

    return weight_function(r_, r_i, r_ii) 
end

function main()

    n, eps, r0 = 8, 0.01, 1
    a, b = 2,1
    
    s_r = [smooth_reg(n, eps, r0), smooth_reg_d1(n, eps, r0), smooth_reg_d2(n, eps, r0)]
    e_r = [ellipse_reg(a,b), ellipse_reg_d1(a,b), ellipse_reg_d2(a,b)]
    t_r = [third_reg(), third_reg_d1(), third_reg_d2()]
    f_r = [fourth_reg(), fourth_reg_d1(), fourth_reg_d2()]

    smooth_weight = weight_reg(s_r...)
    ellipse_weight = weight_reg(e_r...)
    third_weight = weight_reg(t_r...)
    fourth_weight = weight_reg(f_r...)

    t = 0:0.008:2π
    
    @time p,sol = el_bvp_plot(ellipse_weight, 1, r0_est_1(ellipse_weight), "Solver Result", 0.005)
    plot!(p, t, e_r[1].(t), ls=:dash, lw=2)
    print(area_reg(e_r[1], e_r[2], ellipse_weight))
    #check_weight = sol_check(sol)
    #check_weight_reg_factor = integrate(check_weight)
    #plot!(p, t, check_weight.(t)./check_weight_reg_factor, ls=:dash, lw=2)
    #plot!(p, t, ellipse_weight.(t))
    
    #write_sol(sol, "cos_sol.txt")
    #print(check_weight.(sol.t))
    display(p)
    #savefig(string("w_is_cos_check_weight.png"))
end