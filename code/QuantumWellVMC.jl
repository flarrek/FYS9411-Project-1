using LinearAlgebra
using Plots


struct QuantumWell # is a struct for elliptical quantum well systems.
    D::Int64 # is the dimension of the quantum well.
    N::Int64 # is the number of particles in the well.
    a::Float64 # is the characteristic radius of the particles.
    λ::Float64 # is the elliptic parameter of the well.
end
QuantumWell(D::Int64,N::Int64) = QuantumWell(D,N,0.0043,1.0)


mutable struct Move # is a struct for proposed Monte Carlo moves.
    i::Int64 # is the index of the particle to move.
    r::Vector{Float64} # is the new position of the particle.
end
Move() = Move(0,[])


function find_VMC_energy(well::QuantumWell, cycles::Int64; algorithm::String="brute_force",
    initial_α::Float64=0.5, initial_β::Float64=1.0, δr::Float64=0.8, δt::Float64=0.4)
    # Finds the VMC approximate ground state energy of the given quantum well
    # by performing the given number of Monte Carlo cycles based on the given algorithm.

    # CONSTANTS:

    D::Int64 = well.D # is the dimension of the quantum well.
    N::Int64 = well.N # is the number of particles in the well.
    a::Float64 = well.a # is the characteristic radius of the particles.
    λ::Float64 = well.λ # is the elliptic parameter of the well.
    C::Int64 = cycles # is the number of Monte Carlo cycles to be run.

    short_system_description::String = string(D,"D quantum well")
        # is a short description of the quantum well in words.
    long_system_description::String = string("a",((λ==1.0) ? " spherical " : "n elliptical "),
        D,"D quantum well with ",N,((a==0) ? " non-" : " "),"interacting particle",((N>1) ? "s" : ""))
        # is a long description of the quantum well in words.
    system_parameters::String = string("D = ",D,", N = ",N,", a = ",round(a;digits=4),", λ = ",round(λ;digits=4))
        # is a string of the quantum well parameters.

    # VARIABLES:

    α::Float64 = initial_α # is the current variational trial state parameter α.
    β::Float64 = initial_β # is the current variational trial state parameter β.

    R::Vector{Vector{Float64}} = [zeros(D) for i in 1:N] # is the current configuration of the well particles.

    ε::Vector{Float64} = zeros(C) # are the sampled local energies at each Monte Carlo cycle.
    ε²::Vector{Float64} = zeros(C) # are the sampled local energy squares at each Monte Carlo cycle (for calculation of the variance).
    E::Float64 = NaN # is the to be calculated VMC approximate ground state energy of the quantum well.
    ΔE²::Float64 = NaN # is the to be calculated statistical variance of the VMC energy.

    c::Int64 = 0 # is the number of Monte Carlo cycles currently run.
    proposed_move::Move = Move() # is the currently proposed move.
    rejected_moves::Int64 = 0 # is the number of rejected moves because of the random Metropolis acceptance.
    acceptance::Int64 = 0 # is the total percentage of rejected moves because of the random Metropolis acceptance.


    # FUNCTIONS:

    function scatter_particles!()
        # scatters the well particles into an initial box configuration.
        B::Int64 = ceil(N^(1/D))
        for i in 1:N
            R[i] = [1.1*(((i-1)%(B^d))÷(B^(d-1))-(B-1)/2)*a for d in 1:D]
        end
    end

    function propose_move!()
        # proposes a move based on the given algorithm.
        if (algorithm == "brute_force")
            i = rand(1:N)
            Δr = [(2rand()-1) for d in 1:D]*δr
        end
        proposed_move.i = i
        proposed_move.r = R[i]+Δr
    end

    function judge_move!()
        # judges whether the proposed move is accepted based on the characteristic particle radius
        # and the acceptance ratio, and nullifies the move if rejected.

        function acceptance_ratio()
            # returns the Metropolis acceptance ratio for the proposed move based on the given algorithm.
            function transition_ratio()
                # returns the ratio of transition probabilities for the proposed move based on the given algorithm.
                if (algorithm == "brute_force")
                    return 1.0
                end
            end
            g²(r::Vector{Float64}) = exp(-α*([1,1,β][1:D]⋅(r.^2)))^2
            f²(Δr::Float64) = 1-a/Δr
            i = proposed_move.i
            ratio = g²(proposed_move.r)/g²(R[i])
            if (a != 0.0) && (N != 1)
                for j in 1:N
                    if (j == i)
                        continue
                    end
                    ratio *= f²(norm(proposed_move.r-R[j]))/f²(norm(R[i]-R[j]))
                end
            end
            ratio *= transition_ratio()
            return min(1,ratio)
        end

        i = proposed_move.i
        for j in 1:N
            if (j == i)
                continue
            end
            if (norm(proposed_move.r-R[j]) ≤ a)
                # rejects the proposed move definitely if it causes an overlap of particles.
                rejected_moves += 1
                proposed_move = Move()
                return
            end
        end
        if (rand() > acceptance_ratio())
            # rejects the proposed move randomly based on the Metropolis acceptance ratio.
            rejected_moves += 1
            proposed_move = Move()
        end
        # accepts the proposed move if both the above rejection tests fail.
    end

    function move_particles!()
        # moves the well particles based on the proposed move.
        if (proposed_move.i == 0)
            # does not move any particle if the proposed move was rejected.
            return
        end
        i = proposed_move.i
        R[i] = proposed_move.r
    end

    function sample_local_energy!()
        # samples the local energy, as well as the local energy square, at the current particle configuration.
        if (proposed_move.i == 0) && (c > 1)
            # copies the local energy samples from the last cycle if the proposed move was rejected.
            ε[c] = ε[c-1]
            ε²[c] = ε²[c-1]
            return
        end
        U(r::Vector{Float64}) = [1,1,λ^2][1:D]⋅(r.^2)
        q(r::Vector{Float64}) = 4α*([1,1,β][1:D].*r)
        d(Δr::Float64) = Δr^2*(Δr-a)
        s(Δr::Vector{Float64}) = (a/2d(norm(Δr)))*Δr
        ε[c] = α*N*(D+(β-1))
        for i in 1:N
            ε[c] += 1/2*(U(R[i])-1/4*q(R[i])⋅q(R[i]))
            if (a != 0.0) && (N != 1)
                for j in 1:N
                    if (j == i)
                        continue
                    end
                    ε[c] -= (a*(D-3)/2d(norm(R[i]-R[j]))+q(R[i])⋅s(R[i]-R[j]))
                    for k in 1:N
                        if (k == i) || (k == j)
                            continue
                        end
                        ε[c] -= s(R[i]-R[j])⋅s(R[i]-R[k])
                    end
                end
            end
        end
        ε²[c] = ε[c]^2
    end

    function calculate_energy!()
        # calculates the VMC ground state energy approximation of the quantum well, as well as its standard deviation.
        E = sum(ε)/c
        ΔE² = (sum(ε²)/c-E^2)/c
        if (ΔE² < 0)
            if (ΔE²^2 < 1e-20)
                ΔE² = 0.0
            else
                ΔE² = NaN
                error("The statistical variance turned out to be negative!")
            end
        end
    end

    function export_results()
        # exports the results to an external file.
    end

    function plot_particles()
        # plots the well particles in a scatter plot of the right dimension.
        X = [r[1] for r in R]
        Y = (D>1) ? [r[2] for r in R] : 0*X
        Z = (D>2) ? [r[3] for r in R] : 0*X
        plot = scatter(X,Y,Z;title="Particles in "*short_system_description*"<br>("*system_parameters*")",label=false)
        display(plot)
        return
    end


    # EXECUTIONS:

    println()
    println("Finding the VMC energy for ",long_system_description,".")
    println()
    println("Quantum well parameters: ",system_parameters)
    println()
    println("Running ",C," Monte Carlo cycles ...")

    scatter_particles!()
    while (c < C)
        c += 1
        propose_move!()
        judge_move!()
        move_particles!()
        sample_local_energy!()
    end
    calculate_energy!()
    acceptance = round(100*(1-rejected_moves/c))

    println(c," Monte Carlo cycles finished!")
    println()
    println(rejected_moves," moves were rejected.")
    println("Acceptance: ",acceptance,"%")
    println()
    println("Optimal α: ",round(α;digits=4))
    println("Optimal β: ",round(β;digits=4))
    println("VMC energy: ",round(E;digits=4)," ± ",round(√ΔE²;digits=4))
    println()
    plot_particles()
end
