"""Things done on September 2nd."""

include("alt_problem.jl")

println("file started")
ε = 0.1
w1(θ) = 1 + ε*cos(θ)
w1_prime(θ) = -ε*sin(θ)

w2(θ) = 1 + ε*sin(θ)
w2_prime(θ) = ε*cos(θ)

w3(θ) = 1 + sin(2θ)/2
w3_prime(θ) = cos(2θ)

@time p,sol = el_bvp_plot(w1, w1_prime, 1, r0_est_2(w1), "Solver Result", 0.01)
display(p)