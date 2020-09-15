using CubicHermiteSpline
using ForwardDiff

function write_sol(sol, outfile)
    """Writes a DifferentialEquation sol object in
    a text file"""
    open(outfile, "w") do io
        write(io, join(sol.t, " ") * '\n')
        write(io, join(sol[1,:], " ") * '\n')
        write(io, join(sol[2,:], " "))
    end
end

function read_interpolate_sol(infile)
    """Read points from a file, where the first
    line is t, the second is r(t), and the third is r'(t),
    and return a cubic Hermite interpolation"""
    
    io = open(infile, "r")
    t = parse.(Float64, split(readline(io)))
    u = parse.(Float64, split(readline(io)))
    du_dt = parse.(Float64, split(readline(io)))
    close(io)

    spl = CubicHermiteSplineInterpolation(t, u, du_dt)

    return spl
end

function spl_to_functions(spl)
    """Turns the result of CubicHermiteSplineInterpolation
    into multiple functions"""
    r(t) = spl(t)
    r_i(t) = spl(t; grad=true)
    r_ii(t) = ForwardDiff.derivative(r_i, t)

    return r, r_i, r_ii
end

function r0_est(w::Function)
    """Estimates a r(0) from the weight function"""
    return 1/w(0)
end