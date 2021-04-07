using LinearAlgebra
using Distributions
using Plots


struct QuantumTrap # is a struct for elliptical quantum trap systems.
    D::Int64 # is the dimension of the quantum trap.
    N::Int64 # is the number of particles in the trap.
    a::Float64 # is the characteristic radius of the particles.
    λ::Float64 # is the elliptic parameter of the trap.
end
QuantumTrap(D::Int64,N::Int64) = QuantumTrap(D,N,0.0043,√8)

function system_parameters(trap::QuantumTrap)
    # returns a string of the quantum trap parameters.
    return string("D = ",trap.D,", N = ",trap.N,
        ((trap.N > 1) ? string(", a = ",round(trap.a;digits=4)) : ""),
        ((trap.D == 3) ? string(", λ = ",round(trap.λ;digits=4)) : ""))
end

function short_system_description(trap::QuantumTrap)
    # returns a short description of the quantum trap in words.
    return string("a ",trap.D,"D quantum trap")
end

function long_system_description(trap::QuantumTrap)
    # returns a long description of the quantum trap in words.
    return string("a",((trap.D == 3) ? ((trap.λ==1.0) ? " spherical " : "n elliptical ") : " "),
        trap.D,"D quantum trap with ",trap.N,
        ((trap.N > 1) ? ((trap.a==0.0) ? " non-interacting particles" : " interacting particles") : " particle"))
end


struct Algorithm # is a struct for VMC algorithms.
    sampling::String # is the sampling method.
    δs::Float64 # is the step size used for proposing moves in the sampling method.
    differentiation::String # is the method of differentiation.
    scattering::String # is the method of scattering the particles initially.
end
Algorithm(sampling::String,δs::Float64=0.08) = Algorithm(sampling,δs,"analytical","normal")

function algorithm_methods(algorithm::Algorithm)
    # returns a string of the algorithm methods.
    return string(uppercasefirst(algorithm.scattering)," scattering / ",uppercasefirst(algorithm.sampling)," sampling with δs = ",algorithm.δs," / ",
        uppercasefirst(algorithm.differentiation)," differentiation")
end


function find_VMC_energy(trap::QuantumTrap, cycles::Int64=1_000_000, algorithm::Algorithm=Algorithm("quantum drift");
        initial_α::Float64=0.5, initial_β::Float64=trap.λ, output::Bool=true)
    # finds the VMC approximate ground state energy of the given quantum trap
    # by performing the given number of Monte Carlo cycles based on the given algorithm.

    # CONSTANTS:

    D::Int64 = trap.D # is the dimension of the quantum trap.
    N::Int64 = trap.N # is the number of particles in the trap.
    a::Float64 = trap.a # is the characteristic radius of the particles.
    λ::Float64 = trap.λ # is the elliptic parameter of the trap.

    δs::Float64 = algorithm.δs # is the step size used for proposing moves in the sampling method.
    C::Int64 = cycles # is the number of Monte Carlo cycles to be run.


    # VARIABLES:

    α::Float64 = initial_α # is the current variational trial state parameter α.
    β::Float64 = initial_β # is the current variational trial state parameter β.

    c::Int64 = 0 # is the number of Monte Carlo cycles currently run.

    R::Vector{Vector{Float64}} = [zeros(D) for i in 1:N] # is the current configuration of the trap particles.
    ΔR::Matrix{Float64} = zeros(N,N) # is a matrix which stores all current and proposed inter-particle distances.
    Δr = Symmetric(ΔR,:L) # are the current inter-particle distances, stored in the lower triangle of ΔR but accessed symmetrically.

    proposed_i::Int64 = 0 # is the particle index of the randomly chosen particle to move at each Monte Carlo cycle.
    current_Qi::Vector{Float64} = zeros(D) # is the current quantum drift for the particle (used in quantum drift sampling).
    δr::Vector{Float64} = zeros(D) # is a randomly drawn position step from the given sampling method.

    proposed_ri::Vector{Float64} = zeros(D) # is the proposed new position for the randomly chosen particle to move.
    proposed_Δri = view(ΔR,1,:) # are the proposed new inter-particle distances by making the move, stored in the first row of ΔR.
    proposed_Qi::Vector{Float64} = zeros(D) # is the quantum drift at the proposed position for the randomly chosen particle.

    rejected_moves::Int64 = 0 # is the number of rejected moves because of the random Metropolis acceptance.
    acceptance::Int64 = 0 # is the total percentage of rejected moves because of the random Metropolis acceptance.



    ε::Vector{Float64} = zeros(C) # are the sampled local energies at each Monte Carlo cycle.
    ε²::Vector{Float64} = zeros(C) # are the sampled local energy squares at each Monte Carlo cycle (for calculation of the variance).
    E::Float64 = 0.0 # is the to be calculated VMC approximate ground state energy of the quantum trap.
    ΔE²::Float64 = 0.0 # is the to be calculated statistical variance of the VMC energy.


    # FUNCTIONS:

    g²(r::Vector{Float64}) = exp(-2α*([1,1,β][1:D]⋅(r.^2)))
    f²(Δr::Float64) = ((Δr > a) ? (1-a/Δr)^2 : 0.0)
        # are the squares of the functions g and f defined in the report, used to calculate the Metropolis acceptance ratio.

    U(r::Vector{Float64})::Float64 = [1,1,λ^2][1:D]⋅(r.^2) # is the elliptical harmonic trap potential energy.

    q(r::Vector{Float64})::Vector{Float64} = 4α*([1,1,β][1:D].*r)
    d(Δr::Float64)::Float64 = Δr^2*(Δr-a)
    s(Δr::Vector{Float64})::Vector{Float64} = (a/2d(norm(Δr)))*Δr
        # are the quantities defined in the report and used to calculate the quantum drift and the local energy.

    function scatter_particles!()
        # scatters the trap particles into an initial configuration based on the given algorithm.
        if (algorithm.scattering == "normal")
            # scatters the trap particles into a normal distribution around the origin with deviation 1/√2.
            placing::Bool = true
            @inbounds for i in 1:N
                placing = true
                while placing
                    R[i] = rand(Normal(),D)/√2
                    placing = false
                    if (a != 0.0) && (N != 1)
                        for j in 1:(i-1)
                            ΔR[i,j] = norm(R[i]-R[j])
                            if (Δr[i,j] ≤ a)
                                placing = true
                            end
                        end
                    end
                end
            end
        elseif (algorithm.scattering == "lattice")
            # scatters the trap particles into a centered L×L×L-point square lattice with size √2 in each spatial direction.
            L::Int64 = ceil(N^(1/D))
            @inbounds for i in 1:N
                R[i] = [√2(((i-1)%(L^d))÷(L^(d-1))-(L-1)/2) for d in 1:D]
                if (a != 0.0) && (N != 1)
                    for j in 1:(i-1)
                        ΔR[i,j] = norm(R[i]-R[j])
                    end
                end
            end
        else
            error("The initial scattering method '",algorithm.scattering,"' is not known.")
        end
    end

    function quantum_drift(i::Int64,r::Vector{Float64})::Vector{Float64}
        # calculates the quantum drift for particle i at position r with the current configuration.
        drift = q(r)
        if (a != 0.0) && (N != 1)
            @inbounds for j in 1:N
                if (j == i)
                    continue
                end
                drift += 4s(r-R[j])
            end
        end
        return drift
    end

    function propose_move!()
        # proposes a move based on the given sampling method.
        i = rand(1:N)
        if (algorithm.sampling == "random step")
            δr = (2rand(D).-1)*δs
        elseif (algorithm.sampling == "quantum drift")
            @inbounds current_Qi = quantum_drift(i,R[i])
            δr = 1/2*current_Qi*δs^2+rand(Normal(),D)*δs
        else
            error("The sampling method '",algorithm,"' is not known.")
        end
        proposed_i = i
        @inbounds proposed_ri = R[i]+δr
    end

    function judge_move!()
        # judges whether the proposed move is accepted based on the characteristic particle radius
        # and the acceptance ratio, and nullifies the move if rejected.

        function acceptance_ratio()::Float64
            # returns the Metropolis acceptance ratio for the proposed move based on the given algorithm.

            function proposal_ratio()::Float64
                # returns the ratio of proposal distributions for the proposed move based on the given algorithm.
                if (algorithm.sampling == "random step")
                    return 1.0
                elseif (algorithm.sampling == "quantum drift")
                    proposed_Qi = quantum_drift(proposed_i,proposed_ri)
                    return exp(-1/2*(proposed_Qi+current_Qi)⋅(δr+1/4*(proposed_Qi-current_Qi)*δs^2))
                else
                    error("The sampling method '",algorithm,"' is not known.")
                end
            end

            i = proposed_i
            ratio = proposal_ratio()
            @inbounds ratio *= g²(proposed_ri)/g²(R[i])
            if (a != 0.0) && (N != 1)
                @inbounds for j in 1:N
                    if (j == i)
                        continue
                    end
                    proposed_Δri[j] = norm(proposed_ri-R[j])
                    ratio *= f²(proposed_Δri[j])/f²(Δr[i,j])
                end
            end
            return min(1.0,ratio)
        end

        if (rand() > acceptance_ratio())
            # rejects the proposed move randomly based on the Metropolis acceptance ratio.
            rejected_moves += 1
            proposed_i = 0
        end
        # accepts the proposed move if both the above rejection tests fail.
    end

    function move_particles!()
        # moves the trap particles based on the proposed move.
        if (proposed_i == 0)
            # does not move any particle if the proposed move was rejected.
            return
        end
        i = proposed_i
        @inbounds R[i] = proposed_ri
    end

    function sample_local_energy!()
        # updates the inter-particle distances Δr related to the accepted move
        # and samples the local energy, as well as the local energy square, at the new particle configuration.
        @inbounds if (proposed_i == 0) && (c > 1)
            # copies the local energy samples from the last cycle if the proposed move was rejected.
            ε[c] = ε[c-1]
            ε²[c] = ε²[c-1]
            return
        end
        @inbounds ε[c] = α*N*(D+(β-1)*(D==3))
        @inbounds for i in 1:N
            ε[c] += 1/2*(U(R[i])-1/4*q(R[i])⋅q(R[i]))
            if (a != 0.0) && (N != 1)
                for j in 1:N
                    if (j == i)
                        continue
                    elseif (i == proposed_i) # updates the inter-particle distances Δr related to the accepted move.
                        if (j < i)
                            ΔR[i,j] = proposed_Δri[j]
                        else
                            ΔR[j,i] = proposed_Δri[j]
                        end
                    end
                    ε[c] -= (a*(D-3)/2d(Δr[i,j])+q(R[i])⋅s(R[i]-R[j]))
                    for k in 1:N
                        if (k == i) || (k == j)
                            continue
                        end
                        ε[c] -= s(R[i]-R[j])⋅s(R[i]-R[k])
                    end
                end
            end
        end
        @inbounds ε²[c] = ε[c]^2
    end

    function calculate_VMC_energy!()
        # calculates the VMC ground state energy approximation of the quantum trap, as well as its statistical variance.
        E = sum(ε)/c
        ΔE² = (sum(ε²)/c-E^2)/c
        if (ΔE² < 0)
            if (ΔE²^2 < 1e-10)
                ΔE² = 0.0
            else
                error("The statistical variance turned out to be negative with value ",ΔE²,"!")
            end
        end
    end

    function export_results()
        # exports the results to an external file.
    end

    function plot_particles()
        # plots the trap particles in a scatter plot of the right dimension.
        @inbounds X::Vector{Float64} = [r[1] for r in R]
        @inbounds Y::Vector{Float64} = ((D>1) ? [r[2] for r in R] : zeros(N))
        @inbounds Z::Vector{Float64} = ((D>2) ? [r[3] for r in R] : zeros(N))
        particles = plot(title="Particles in "*short_system_description(trap)*"<br>("*system_parameters(trap)*")")
        scatter!(particles,X,Y,Z;color="#aa4888",label=false)
        display(particles)
        return
    end


    # EXECUTIONS:

    if output
        println()
        println("Finding the VMC energy for ",long_system_description(trap),".")
        println()
        println("Quantum trap parameters: ",system_parameters(trap))
        println("Algorithm: ",algorithm_methods(algorithm))
        println()
        println("Running ",C," Monte Carlo cycles ...")
    end

    scatter_particles!()
#    plot_particles()
    while (c < C)
        c += 1
        propose_move!()
        judge_move!()
        move_particles!()
        sample_local_energy!()
    end
#    plot_particles()
    calculate_VMC_energy!()
    acceptance = round(100*(1-rejected_moves/c))

    if output
        println(c," MONTE CARLO CYCLES FINISHED!")
        println()
        println(rejected_moves," moves were rejected.")
        println("Acceptance: ",acceptance,"%")
        println()
        println("Optimal α: ",round(α;digits=4))
        if (D == 3)
            println("Optimal β: ",round(β;digits=4))
        end
        println("VMC energy: ",round(E;digits=4)," ± ",round(√ΔE²;digits=4))
        println()
    end

    return E,√ΔE²
end


function compare_VMC_sampling(trap::QuantumTrap,resolution::Int64=100;
        δs::Float64=0.08,initial_α::Float64=0.5,initial_β::Float64=trap.λ,output::Bool=false)
    # compares the two VMC sampling methods by plotting their results against a number of Monte Carlo cycles spanning 1000 to 1_000_000.

    # VARIABLES:

    C::Vector{Int64} = [round(10^e) for e in range(3,7;length=resolution)] # is the vector of Monte Carlo cycles to be run.
    E_RS::Vector{Float64} = zeros(resolution) # is the vector of VMC energies from the random step sampling method.
    ΔE_RS::Vector{Float64} = zeros(resolution) # is the vector of statistical error from the random step sampling method.
    E_QD::Vector{Float64} = zeros(resolution) # is the vector of VMC energies from the quantum drift sampling method.
    ΔE_QD::Vector{Float64} = zeros(resolution) # is the vector of statistical error from the quantum drift sampling method.

    E::Float64 = 0.0 # is the to be calculated reference VMC energy of the quantum trap.


    # EXECUTIONS:

    println()
    println("Comparing VMC sampling methods for ",long_system_description(trap)*".")
    println()
    println("Quantum trap parameters: ",system_parameters(trap))
    println()
    println("Running ",BigInt(2sum(C))," Monte Carlo cycles ...")
    @inbounds for i in 1:resolution
        E_RS[i],ΔE_RS[i] = find_VMC_energy(trap,Algorithm("random step",δs),C[i];initial_α=initial_α,initial_β=initial_β,output=output)
        E_QD[i],ΔE_QD[i] = find_VMC_energy(trap,Algorithm("quantum drift",δs),C[i];initial_α=initial_α,initial_β=initial_β,output=output)
    end
    println("VMC SAMPLING COMPARISON FINISHED!")
    println()
    E,_ = find_VMC_energy(trap,Algorithm("quantum drift",δs),10^8;initial_α=initial_α,initial_β=initial_β,output=true)
    println()

    println("Plotting results.")
    comparison1 = plot(title="Comparison of VMC sampling from "*short_system_description(well)*"<br>("*system_parameters(well)*", δs = "*string(δs)*")",
        legend=:bottomright,xlabel="Monte Carlo cycles",xaxis=:log,ylabel="VMC energy [ħω]")
    plot!(comparison1,C,E_RS;ribbon=ΔE_RS,fillalpha=.5,width=2,color="#4aa888",label="random step sampling")
    plot!(comparison1,C,E_QD;ribbon=ΔE_QD,fillalpha=.5,width=2,color="#aa4888",label="quantum drift sampling")
    plot!(comparison1,C,[E for i in 1:resolution];style=:dash,width=2,color="#fdce0b",label="reference VMC energy")
    display(comparison1)
    comparison2 = plot(title="Comparison of VMC sampling error from "*short_system_description(well)*"<br>("*system_parameters(well)*", δs = "*string(δs)*")",
        legend=:right,xlabel="Monte Carlo cycles",xaxis=:log,ylabel="VMC energy error [ħω]")
    plot!(comparison2,C,ΔE_RS;width=2,color="#4aa888",label="random step sampling")
    plot!(comparison2,C,ΔE_QD;width=2,color="#aa4888",label="quantum drift sampling")
    display(comparison2)
    println()
    return
end
