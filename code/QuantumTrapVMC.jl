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
QuantumTrap(D::Int64,N::Int64,a::Float64) = QuantumTrap(D,N,a,√8)

function system_parameters(trap::QuantumTrap)
    # returns a string of the quantum trap parameters.
    return string("D = ",trap.D," / N = ",trap.N,
        ((trap.N > 1) ? string(" / a = ",round(trap.a;digits=4)) : ""),
        ((trap.D == 3) ? string(" / λ = ",round(trap.λ;digits=4)) : ""))
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


struct VMCAlgorithm # is a struct for VMC algorithms.
    variation::String # is the method of variation of the parameters α and β.
    sampling::String # is the method of sampling from the probability distribution.
    δs::Float64 # is the step size used for proposing moves in the sampling method.
    differentiation::String # is the method of differentiation for calculating quantum drift and the local energy.
    δd::Float64 # is the step size used in numerical differentiation.
    scattering::String # is the method of scattering for the initial trap particle configuration.
end
VMCAlgorithm(variation::String,sampling::String,δs::Float64=√0.1,
    differentiation::String="analytical",δd::Float64=0.01) = VMCAlgorithm(variation,sampling,δs,differentiation,δd,"normal")

function algorithm_parameters(algorithm::VMCAlgorithm)
    # returns a string of the algorithm parameters.
    return string("δs = ",round(algorithm.δs;digits=4),
        ((algorithm.differentiation=="numerical") ? string(" / δd = ",round(algorithm.δd;digits=4)) : ""))
end

function algorithm_methods(algorithm::VMCAlgorithm)
    # returns a string of the algorithm methods.
    return string(uppercasefirst(algorithm.variation)," variation / ",uppercasefirst(algorithm.sampling)," sampling / ",
        uppercasefirst(algorithm.differentiation)," differentiation / ", uppercasefirst(algorithm.scattering)," scattering")
end


function find_VMC_energy(trap::QuantumTrap, algorithm::VMCAlgorithm=VMCAlgorithm("range","quantum drift"), cycles::Vector{Int64}=[1_000_000];
        αs::Vector{Float64}=[0.5], βs::Vector{Float64}=[trap.λ], output::Bool=true)
    # finds the VMC approximate ground state energy of the given quantum trap by varying the parameters α and β
    # using the given method of variation, and performing the given number of Monte Carlo cycles at each variational point
    # using the given methods of scattering, differentiation and sampling.

    # CONSTANTS:

    D::Int64 = trap.D # is the dimension of the quantum trap.
    N::Int64 = trap.N # is the number of particles in the trap.
    a::Float64 = trap.a # is the characteristic radius of the particles.
    λ::Float64 = trap.λ # is the elliptic parameter of the trap.

    δs::Float64 = algorithm.δs # is the step size used for proposing moves in the sampling method.
    δd::Float64 = algorithm.δd # is the step size used in numerical differentiation.

    U::Int64 = length(αs) # is the number of α values to be considered (if range variation).
    V::Int64 = length(βs) # is the number of β values to be considered (if range variation).
    M::Int64 = length(cycles) # is the number of VMC energies to calculate and store at each variational point
        # (used to plot convergence of the sampling methods).
    C::Int64 = cycles[end] # is the number of Monte Carlo cycles to be run in total at each variational point.


    # VARIABLES:

    α::Float64 = αs[1] # is the current variational trial state parameter α.
    β::Float64 = βs[1] # is the current variational trial state parameter β.
    c::Int64 = 0 # is the number of Monte Carlo cycles run at the current variational point.

    R::Vector{Vector{Float64}} = [zeros(D) for i in 1:N] # is the current configuration of the trap particles.
    ΔR::Matrix{Float64} = zeros(N,N) # is a matrix which stores all current and proposed inter-particle distances.
    Δr = Symmetric(ΔR) # are the current inter-particle distances, stored in the upper triangle of ΔR but accessed symmetrically.

    s::Matrix{Vector{Float64}} = [zeros(D) for i in 1:N,j in 1:N]
        # is a matrix which stores all current values of the vector quantities q and s as defined in the report.
    q = view(s,diagind(s)) # are the current values of q, stored along the diagonal of the matrix s.

    proposed_i::Int64 = 0 # is the particle index of the randomly chosen particle to move at each Monte Carlo cycle.
    current_Qi::Vector{Float64} = zeros(D) # is the current quantum drift for the particle (used in quantum drift sampling).
    δr::Vector{Float64} = zeros(D) # is a randomly drawn position step for the particle to move using the given sampling method.
    proposed_ri::Vector{Float64} = zeros(D) # is the proposed new position for the particle to move.
    proposed_Δri = view(ΔR,:,1) # are the proposed new inter-particle distances by making the move, stored in the first column of ΔR.
    proposed_Qi::Vector{Float64} = zeros(D) # is the quantum drift at the proposed new position for the particle to move
        # (used in quantum drift sampling).

    rejected_moves::Int64 = 0 # is the number of rejected moves at the current variational point because of random Metropolis acceptance.
    A::Int64 = 0# is the total percentage of rejected moves at the current variational point because of random Metropolis acceptance.

    ε::Vector{Float64} = zeros(C) # are the sampled local energies from each Monte Carlo cycle at the current variational point.
    ε²::Vector{Float64} = zeros(C) # are the sampled local energy squares from each Monte Carlo cycle at the current variational point.

    E::Float64 = 0.0 # is the calculated energy at the current variational point,
        # as well as the final VMC ground state energy of the quantum trap.
    ΔE²::Float64 = 0.0 # is the estimated statistical variance of the energy at the current variational point.
    ΔE::Float64 = 0.0 # is the estimated statistical error for the energy at the current variational point,
        # as well as the final statistical error of the VMC ground state energy of the quantum trap.

    Es::Array{Float64,3} = zeros(U,V,M) # is the matrix of calculated energies at each variational point.
    ΔEs::Array{Float64,3} = zeros(U,V,M) # is the matrix of statistical error for the calculated energies.


    # FUNCTIONS:

    function _g²(r::Vector{Float64})::Float64
        tmp = 0.0
        @inbounds for d in 1:D
            if (d < 3)
                tmp += r[d]^2
            else
                tmp += β*r[d]^2
            end
        end
        return exp(-2α*tmp)
    end
    _f²(Δr::Float64)::Float64 = ((Δr > a) ? (1-a/Δr)^2 : 0.0)
        # are the squares of the functions g and f defined in the report, used to calculate the Metropolis acceptance ratio.

    function _U(r::Vector{Float64})::Float64
        # calculates the elliptical harmonic potential energy for a particle at position r.
        tmp = 0.0
        @inbounds for d in 1:D
            if (d < 3)
                tmp += r[d]^2
            else
                tmp += λ^2*r[d]^2
            end
        end
        return tmp
    end

    function _q(r::Vector{Float64})::Vector{Float64}
        tmp = zeros(D)
        @inbounds for d in 1:D
            if (d < 3)
                tmp[d] = r[d]
            else
                tmp[d] = β*r[d]
            end
        end
        return -4α*tmp
    end
    _d(Δr::Float64)::Float64 = Δr^2*(Δr-a)
    _s(Δr::Vector{Float64})::Vector{Float64} = a*Δr/(2*_d(norm(Δr)))
        # are the quantities defined in the report and used to calculate the quantum drift and the local energy.

    function _Q(i::Int64,ri::Vector{Float64})::Vector{Float64}
        # calculates the quantum drift for particle i at the proposed position ri with the current configuration.
        Qi = _q(ri)
        if (a != 0.0) && (N != 1)
            @inbounds for j in 1:N
                if (j == i)
                    continue
                end
                Qi += 4*_s(ri-R[j])
            end
        end
        return Qi
    end

    function _Q(i::Int64)::Vector{Float64}
        # calculates the quantum drift for particle i with the current configuration.
        @inbounds Qi = q[i]
        if (a != 0.0) && (N != 1)
            @inbounds for j in 1:N
                if (j == i)
                    continue
                end
                Qi += 4*s[j,i]
            end
        end
        return Qi
    end


    function reset_variables!()
        # resets all variables for the next Monte Carlo simulation.
        c = 0
        rejected_moves = 0
        ε = zeros(C)
        ε² = zeros(C)
        E = 0.0
        ΔE² = 0.0
        ΔE = 0.0
    end


    function scatter_particles!()
        # scatters the trap particles into an initial configuration based on the given method.
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
                            ΔR[j,i] = norm(R[i]-R[j]) # calculates and stores the initial values of Δr.
                            if (Δr[j,i] ≤ a) # replaces the particle if it overlaps with one already placed.
                                placing = true
                                break
                            else # calculates and stores the initial values of s if the particle does not overlap.
                                s[j,i] = _s(R[i]-R[j])
                                s[i,j] = -s[j,i]
                            end
                        end
                    end
                end
                q[i] = _q(R[i]) # calculates and stores the initial values of q.
            end
        elseif (algorithm.scattering == "lattice")
            # scatters the trap particles into a centered L×L×L-point square lattice with size √2 in each direction.
            L::Int64 = ceil(N^(1/D))
            @inbounds for i in 1:N
                R[i] = [√2(((i-1)%(L^d))÷(L^(d-1))-(L-1)/2) for d in 1:D]
                if (a != 0.0) && (N != 1)
                    for j in 1:(i-1) # calculates and stores the initial values of Δr and s.
                        ΔR[j,i] = norm(R[i]-R[j])
                        s[j,i] = _s(R[i]-R[j])
                        s[i,j] = -s[j,i]
                    end
                end
                q[i] = _q(R[i]) # calculates and stores the initial values of q.
            end
        else
            error("The initial scattering method '",algorithm.scattering,"' is not known.")
        end
    end

    function propose_move!()
        # proposes a move based on the given sampling method.
        i = rand(1:N)
        if (algorithm.sampling == "random step")
            @inbounds for d in 1:D
                δr[d] = (2rand()-1)*δs
            end
        elseif (algorithm.sampling == "quantum drift")
            current_Qi = _Q(i)
            dist = Normal()
            @inbounds for d in 1:D
                δr[d] = 1/2*current_Qi[d]*δs^2+rand(dist)*δs
            end
        else
            error("The sampling method '",algorithm.sampling,"' is not known.")
        end
        proposed_i = i
        @inbounds proposed_ri = R[i]+δr
    end

    function judge_move!()
        # judges whether the proposed move is accepted based on the Metropolis acceptance ratio,
        # and nullifies the move if rejected.

        function acceptance_ratio()::Float64
            # returns the Metropolis acceptance ratio for the proposed move based on the given sampling method.

            function proposal_ratio()::Float64
                # returns the ratio of proposal probabilities for the proposed move based on the given sampling method.
                if (algorithm.sampling == "random step")
                    return 1.0
                elseif (algorithm.sampling == "quantum drift")
                    proposed_Qi = _Q(proposed_i,proposed_ri)
                    tmp = 0.0
                    @inbounds for d in 1:D
                        tmp += (proposed_Qi[d]+current_Qi[d])*(δr[d]+1/4*(proposed_Qi[d]-current_Qi[d])*δs^2)
                    end
                    return exp(-1/2*tmp)
                else
                    error("The sampling method '",algorithm.sampling,"' is not known.")
                end
            end

            i = proposed_i
            ratio = proposal_ratio()
            @inbounds ratio *= _g²(proposed_ri)/_g²(R[i])
            if (a != 0.0) && (N != 1)
                @inbounds for j in 1:N
                    if (j == i)
                        continue
                    end
                    proposed_Δri[j] = norm(proposed_ri-R[j])
                    ratio *= _f²(proposed_Δri[j])/_f²(Δr[j,i])
                end
            end
            return min(1.0,ratio)
        end

        if (rand() > acceptance_ratio())
            # rejects the proposed move randomly based on the Metropolis acceptance ratio.
            rejected_moves += 1
            proposed_i = 0
        end
        # accepts the proposed move if the above rejection test fails.
    end

    function move_particles!()
        # moves the trap particles based on the proposed move.
        if (proposed_i == 0)
            # does not move any particle if the proposed move was rejected.
            return
        end
        i = proposed_i
        @inbounds R[i] = proposed_ri # updates the position of the particle.
        @inbounds q[i] = _q(proposed_ri) # calculates and updates the vector trait q of the particle.
        @inbounds for j in 1:N # calculates and updates the inter-particle distances Δr and the vector traits s related to the particle.
            if (j == i)
                continue
            elseif (j < i)
                ΔR[j,i] = norm(R[i]-R[j])
                s[j,i] = _s(R[i]-R[j])
                s[i,j] = -s[j,i]
            else
                ΔR[i,j] = norm(R[j]-R[i])
                s[i,j] = _s(R[j]-R[i])
                s[j,i] = -s[i,j]
            end
        end
    end

    function sample_local_energy!()
        # samples the local energy, as well as the local energy square, at the new particle configuration.
        @inbounds if (proposed_i == 0) && (c > 1)
            # copies the local energy samples from the last cycle if the proposed move was rejected.
            ε[c] = ε[c-1]
            ε²[c] = ε²[c-1]
            return
        end
        @inbounds ε[c] = α*N*(D+(β-1)*(D==3))
        @inbounds for i in 1:N
            ε[c] += 1/2*_U(R[i])
            for d in 1:D
                ε[c] -= 1/8*q[i][d]^2
            end
            if (a != 0.0) && (N != 1)
                for j in 1:N
                    if (j == i)
                        continue
                    end
                    ε[c] -= a*(D-3)/(2*_d(Δr[j,i]))
                    for d in 1:D
                        ε[c] -= q[i][d]*s[j,i][d]
                    end
                    for k in 1:N
                        if (k == i) || (k == j)
                            continue
                        end
                        for d in 1:D
                            ε[c] -= 2*s[j,i][d]*s[k,i][d]
                        end
                    end
                end
            end
        end
        @inbounds ε²[c] = ε[c]^2
    end

    function calculate_energy!()
        # calculates the energy at the given variational point, as well as its statistical variance and error.
        E = sum(ε)/c
        ΔE² = (sum(ε²)/c-E^2)/c
        if (ΔE² < 0)
            if (ΔE²^2 < 1e-10)
                ΔE² = 0.0
            else
                error("The statistical variance turned out to be negative with value ",ΔE²,"!")
            end
        end
        ΔE = √ΔE²
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
        println("Algorithm methods: ",algorithm_methods(algorithm))
        println("Algorithm parameters: ",algorithm_parameters(algorithm))
        println()
    end
    for u in 1:U, v in 1:V
        α = αs[u]
        β = βs[v]
        if output
            println()
            println("Running ",C," Monte Carlo cycles at the variational point (",
            "α = ",round(α;digits=4),((D == 3) ? string(" / β = ",round(β;digits=4)) : ""),") ... ")
        end
        reset_variables!()
        scatter_particles!()
        while (c < C)
            c += 1
            propose_move!()
            judge_move!()
            move_particles!()
            sample_local_energy!()
            for m in 1:M
                if (c == cycles[m])
                    calculate_energy!()
                    Es[u,v,m] = E
                    ΔEs[u,v,m] = ΔE
                end
            end
        end
#        plot_particles()
        A = round(100*(1-rejected_moves/c))
        if output
            println(c," Monte Carlo cycles finished!")
            println()
            println("Acceptance: ",A,"%")
            println("Energy: ",round(E;digits=4)," ± ",round(ΔE;digits=4))
            println()
        end
    end

    if output # prints the optimal parameters α and β as well as the VMC energy if output is turned on.
        E,uv = findmin(Es[:,:,M])
        u = uv[1]
        v = uv[2]
        α = αs[u]
        β = βs[v]
        ΔE = ΔEs[u,v,M]
        println()
        println("Optimal α: ",round(α;digits=4))
        if (D == 3)
            println("Optimal β: ",round(β;digits=4))
        end
        println("VMC energy: ",round(E;digits=4)," ± ",round(ΔE;digits=4))
        println()
    end

    return Es,ΔEs
end


function compare_VMC_sampling(trap::QuantumTrap,resolution::Int64=10;δs::Float64=√0.1,α::Float64=0.5,β::Float64=trap.λ)
    # compares the two VMC sampling methods by plotting their results against a number of Monte Carlo cycles spanning from 1000 to 10_000000.

    # CONSTANTS:

    cycles::Vector{Int64} = [round(10^e) for e in range(3,7;length=resolution)]
        # is the vector of Monte Carlo cycles for which to store and plot the energies.


    # EXECUTIONS:

    println()
    println("Comparing VMC sampling methods for ",long_system_description(trap),".")
    println()
    println("Quantum trap parameters: ",system_parameters(trap))
    println("Algorithm parameters: δs = ",round(δs;digits=4))
    println("Variational parameters: α = ",round(α;digits=4),
        ((trap.D == 3) ? string(" / β = ",round(β;digits=4)) : ""))
    println()
    println("Running ",cycles[end]," Monte Carlo cycles with random step sampling ...")
    Es_RS,ΔEs_RS = find_VMC_energy(trap,VMCAlgorithm("range","random step",δs),cycles;αs=[α],βs=[β],output=false)
    Es_RS = Es_RS[1,1,:]
    ΔEs_RS = ΔEs_RS[1,1,:]
    println("Running ",cycles[end]," Monte Carlo cycles with quantum drift sampling ...")
    Es_QD,ΔEs_QD = find_VMC_energy(trap,VMCAlgorithm("range","quantum drift",δs),cycles;αs=[α],βs=[β],output=false)
    Es_QD = Es_QD[1,1,:]
    ΔEs_QD = ΔEs_QD[1,1,:]
    println("VMC sampling comparison finished!")
    println()

    E = Es_QD[end]
    println("Plotting results.")
    comparison = plot(title="Comparison of VMC sampling for "*short_system_description(trap)*
        "<br>("*system_parameters(trap)*" / δs = "*string(round(δs;digits=4))*
        " / α = "*string(round(α;digits=4))*((trap.D == 3) ? string(" / β = ",round(β;digits=4)) : "")*")",
        legend=:bottomright,xlabel="Monte Carlo cycles",xaxis=:log,ylabel="VMC energy [ħω]")
    plot!(comparison,cycles,Es_RS;ribbon=ΔEs_RS,fillalpha=.5,width=2,color="#4aa888",label="random step sampling")
    plot!(comparison,cycles,Es_QD;ribbon=ΔEs_QD,fillalpha=.5,width=2,color="#aa4888",label="quantum drift sampling")
    plot!(comparison,cycles,[E for i in 1:resolution];style=:dash,width=2,color="#fdce0b",label="reference VMC energy")
    display(comparison)
    println()
    return
end


function plot_VMC_variation(trap::QuantumTrap;αs::Vector{Float64}=[0.1n for n in 1:9],βs::Vector{Float64}=[trap.λ])
    # plots the energies at the given values for the variational parameters α and β.

    # CONSTANTS:
    U::Int64 = length(αs) # is the number of α values to be considered.
    V::Int64 = length(βs) # is the number of β values to be considered.
    C::Int64 = 1_000000 # is the number of Monte Carlo cycles to be run at each variational point.

    # EXECUTIONS:

    println()
    println("Plotting VMC variation for ",long_system_description(trap),".")
    println()
    println("Quantum trap parameters: ",system_parameters(trap))
    println()
    println("Running ",U*V*C," Monte Carlo cycles ...")
    Es,ΔEs = find_VMC_energy(trap,VMCAlgorithm("range","quantum drift"),[C];αs=αs,βs=βs,output=false)
    Es = Es[:,:,1]
    ΔEs = ΔEs[:,:,1]
    println(U*V*C," Monte Carlo cycles finished!")
    println()
    println("Plotting results.")
    if (trap.D != 3) || (V == 1)
        β::Float64 = βs[1]
        variation = plot(title="VMC variation for "*short_system_description(trap)*
            "<br>("*system_parameters(trap)*((trap.D == 3) ? string(" / β = ",round(β;digits=4)) : "")*")",
            xlabel="α",ylabel="energy [ħω]")
            plot!(variation,αs,Es[:,1];ribbon=ΔEs[:,1],fillalpha=.5,width=2,color="#fdce0b",label=false)
    else
        variation = plot(title="VMC variation for "*short_system_description(trap)*
            "<br>("*system_parameters(trap)*")",xlabel="α",ylabel="β",zlabel="energy [ħω]")
        plot!(variation,αs,βs,Es;st=:surface,seriescolor=:sun,label=false)
    end
    display(variation)
    println()
    return
end
