using Distributed
@everywhere using LinearAlgebra
@everywhere using Distributions
using Plots


@everywhere struct QuantumTrap # is a struct for elliptical quantum trap systems.
    D::Int64 # is the dimension of the quantum trap.
    N::Int64 # is the number of particles in the trap.
    a::Float64 # is the characteristic radius of the particles.
    λ::Float64 # is the elliptic parameter of the trap.
end
QuantumTrap(D::Int64,N::Int64) = QuantumTrap(D,N,0.0043,√8)
QuantumTrap(D::Int64,N::Int64,a::Float64) = QuantumTrap(D,N,a,√8)

function system_parameters(trap::QuantumTrap)::String
    # returns a string of the quantum trap parameters.
    return string("D = ",trap.D," / N = ",trap.N,
        (trap.N > 1 ? string(" / a = ",round(trap.a;digits=4)) : ""),
        (trap.D == 3 ? string(" / λ = ",round(trap.λ;digits=4)) : ""))
end

function short_system_description(trap::QuantumTrap)::String
    # returns a short description of the quantum trap in words.
    return string("a ",trap.D,"D quantum trap")
end

function long_system_description(trap::QuantumTrap)::String
    # returns a long description of the quantum trap in words.
    return string("a",(trap.D == 3 ? (trap.λ == 1.0 ? " spherical " : "n elliptical ") : " "),
        trap.D,"D quantum trap with ",trap.N,
        (trap.N > 1 ? (trap.a == 0.0 ? " non-interacting particles" : " interacting particles") : " particle"))
end


@everywhere function simulate_n_sample(trap::QuantumTrap, cycles::Int64; α::Float64, β::Float64,
        variation::String, scattering::String, sampling::String, δv::Float64, δg::Float64, δs::Float64)
    # performs the given number of Monte Carlo cycles and takes as many samples of local energy and variational derivatives
    # using the given methods of scattering and sampling.

    # CONSTANTS:

    D::Int64 = trap.D # is the dimension of the quantum trap.
    N::Int64 = trap.N # is the number of particles in the trap.
    a::Float64 = trap.a # is the characteristic radius of the particles.
    λ::Float64 = trap.λ # is the elliptical parameter of the trap.

    C::Int64 = cycles # is the number of Monte Carlo cycles to be run in total.


    # VARIABLES:

    c::Int64 = 0 # is the number of Monte Carlo cycles currently run.

    R::Vector{Vector{Float64}} = [zeros(D) for i in 1:N] # is the current configuration of the trap particles.
    ΔR::Matrix{Float64} = zeros(N,N) # is a matrix which stores all current and proposed inter-particle distances.
    Δr = Symmetric(ΔR)
        # are the current inter-particle distances, stored in the upper triangle of ΔR but accessed symmetrically.

    s::Matrix{Vector{Float64}} = [zeros(D) for i in 1:N,j in 1:N]
        # is a matrix which stores all current values of the vector quantities q and s as defined in the report.
    q = view(s,diagind(s)) # are the current values of q, stored along the diagonal of the matrix s.

    proposed_i::Int64 = 0 # is the particle index of the randomly chosen particle to move at each Monte Carlo cycle.
    δr::Vector{Float64} = zeros(D)
        # is a randomly drawn position step for the particle to move using the given sampling method.
    proposed_ri::Vector{Float64} = zeros(D) # is the proposed new position for the particle to move.
    proposed_Δri = view(ΔR,:,1)
        # are the proposed new inter-particle distances by making the move, stored in the first column of ΔR.

    if sampling == "quantum drift"
        current_Qi::Vector{Float64} = zeros(D)
            # is the current quantum drift for the particle to move.
        proposed_Qi::Vector{Float64} = zeros(D)
            # is the quantum drift at the proposed new position for the particle to move.
    end

    ε::Vector{Float64} = zeros(C)
        # are the sampled local energies from each Monte Carlo cycle.
    ε²::Vector{Float64} = zeros(C)
        # are the sampled local energy squares from each Monte Carlo cycle.

    if variation == "gradient descent"
        ∂lnΨ∂α::Vector{Float64} = zeros(C)
        ∂lnΨ∂αε::Vector{Float64} = zeros(C)
        ∂lnΨ∂β::Vector{Float64} = zeros(C)
        ∂lnΨ∂βε::Vector{Float64} = zeros(C)
            # are the sampled variational derivative quantities from each Monte Carlo cycle.
    end

    rejected_moves::Int64 = 0
        # is the number of rejected moves at the current variational point because of random Metropolis acceptance.


    # FUNCTIONS:

    function _q(r::Vector{Float64})::Vector{Float64}
        tmp = zeros(D)
        @inbounds for d in 1:D
            if d < 3
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
        # calculates the quantum drift for particle i at the proposed position ri with the current configuration
        # (used for quantum drift sampling).
        Qi = _q(ri)
        if N > 1 && a > 0.0
            @inbounds for j in 1:N
                if j == i
                    continue
                end
                Qi += 4*_s(ri-R[j])
            end
        end
        return Qi
    end

    function _Q(i::Int64)::Vector{Float64}
        # calculates the quantum drift for particle i with the current configuration
        # (used for quantum drift sampling).
        @inbounds Qi = q[i]
        if N > 1 && a > 0.0
            @inbounds for j in 1:N
                if j == i
                    continue
                end
                Qi += 4*s[j,i]
            end
        end
        return Qi
    end

    function scatter_particles!()
        # scatters the trap particles into an initial configuration based on the given method.
        if scattering == "normal"
            # scatters the trap particles into a normal distribution around the origin with deviation 1/√2.
            placing = true
            dbn = Normal()
            @inbounds for i in 1:N
                placing = true
                while placing
                    R[i] = rand(dbn,D)/√2
                    placing = false
                    if N > 1 && a > 0.0
                        for j in 1:(i-1)
                            ΔR[j,i] = norm(R[i]-R[j]) # calculates and stores the initial values of Δr.
                            if Δr[j,i] ≤ a # replaces the particle if it overlaps with one already placed.
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
        elseif scattering == "lattice" && N > 1
            # scatters the trap particles into a centered L×L×L-point square lattice with size √2 in each direction.
            L = ceil(Int,N^(1/D))
            @inbounds for i in 1:N
                for d in 1:D
                    R[i][d] = (((i-1)%(L^d))÷(L^(d-1))-(L-1)/2)*(√2/(L-1))
                end
                if N > 1 && a > 0.0
                    for j in 1:(i-1) # calculates and stores the initial values of Δr and s.
                        ΔR[j,i] = norm(R[i]-R[j])
                        s[j,i] = _s(R[i]-R[j])
                        s[i,j] = -s[j,i]
                    end
                end
                q[i] = _q(R[i]) # calculates and stores the initial values of q.
            end
        end
    end

    function propose_move!()
        # proposes a move based on the given sampling method.
        i = rand(1:N)
        if sampling == "random step"
            # proposes a uniformly distributed random step with size δs.
            @inbounds for d in 1:D
                δr[d] = (2rand()-1)*δs
            end
        elseif sampling == "quantum drift"
            # proposes a move by quantum drift and a normally distributed random step with deviation δs.
            dbn = Normal()
            current_Qi = _Q(i)
            @inbounds for d in 1:D
                δr[d] = 1/2*current_Qi[d]*δs^2+rand(dbn)*δs
            end
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
                if sampling == "random step"
                    return 1.0
                elseif sampling == "quantum drift"
                    proposed_Qi = _Q(proposed_i,proposed_ri)
                    tmp = 0.0
                    @inbounds for d in 1:D
                        tmp += (proposed_Qi[d]+current_Qi[d])*(δr[d]+1/4*(proposed_Qi[d]-current_Qi[d])*δs^2)
                    end
                    return exp(-1/2*tmp)
                end
            end

            function _g²(r::Vector{Float64})::Float64
                tmp = 0.0
                @inbounds for d in 1:D
                    if d < 3
                        tmp += r[d]^2
                    else
                        tmp += β*r[d]^2
                    end
                end
                return exp(-2α*tmp)
            end
            _f²(Δr::Float64)::Float64 = (Δr > a ? (1-a/Δr)^2 : 0.0)
                # are the squares of the functions g and f defined in the report.

            i = proposed_i
            ratio = proposal_ratio()
            @inbounds ratio *= _g²(proposed_ri)/_g²(R[i])
            if N > 1 && a > 0.0
                @inbounds for j in 1:N
                    if j == i
                        continue
                    end
                    proposed_Δri[j] = norm(proposed_ri-R[j])
                    ratio *= _f²(proposed_Δri[j])/_f²(Δr[j,i])
                end
            end
            return min(1.0,ratio)
        end

        if rand() > acceptance_ratio() # rejects the proposed move randomly based on the Metropolis acceptance ratio.
            rejected_moves += 1
            proposed_i = 0
        end
    end

    function move_particles!()
        # moves the trap particles based on the proposed move.
        if proposed_i == 0 # does not move any particle if the proposed move was rejected.
            return
        end
        i = proposed_i
        @inbounds R[i] = proposed_ri # updates the position of the particle.
        @inbounds q[i] = _q(proposed_ri) # calculates and updates the vector trait q of the particle.
        if N > 1 && a > 0.0
            @inbounds for j in 1:N
                # calculates and updates the inter-particle distances Δr and the vector traits s related to the particle.
                if j == i
                    continue
                elseif j < i
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
    end

    function sample_quantities!()
        # samples the local energy, the local energy square as well as potentially the variational derivative quantities
        # at the new particle configuration.

        function _U(r::Vector{Float64})::Float64
            # calculates the elliptical harmonic potential energy for a particle at position r.
            tmp = 0.0
            @inbounds for d in 1:D
                if d < 3
                    tmp += r[d]^2
                else
                    tmp += λ^2*r[d]^2
                end
            end
            return tmp
        end

        @inbounds if proposed_i == 0 && c > 1 # copies the samples from the last cycle if the proposed move was rejected.
            ε[c] = ε[c-1]
            ε²[c] = ε²[c-1]
            if variation == "gradient descent"
                ∂lnΨ∂α[c] = ∂lnΨ∂α[c-1]
                ∂lnΨ∂αε[c] = ∂lnΨ∂αε[c-1]
                if D == 3
                    ∂lnΨ∂β[c] = ∂lnΨ∂β[c-1]
                    ∂lnΨ∂βε[c] = ∂lnΨ∂βε[c-1]
                end
            end
            return
        end
        @inbounds ε[c] = α*N*(D+(β-1)*(D==3))
        @inbounds for i in 1:N
            ε[c] += 1/2*_U(R[i])
            for d in 1:D
                ε[c] -= 1/8*q[i][d]^2
                if variation == "gradient descent"
                    if d < 3
                        ∂lnΨ∂α[c] -= R[i][d]^2
                    else
                        ∂lnΨ∂α[c] -= β*R[i][d]^2
                        ∂lnΨ∂β[c] -= α*R[i][d]^2
                    end
                end
            end
            if N > 1 && a > 0.0
                for j in 1:N
                    if j == i
                        continue
                    end
                    ε[c] -= a*(D-3)/(2*_d(Δr[j,i]))
                    for d in 1:D
                        ε[c] -= q[i][d]*s[j,i][d]
                    end
                    for k in 1:N
                        if k == i || k == j
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
        if variation == "gradient descent"
            @inbounds ∂lnΨ∂αε[c] = ∂lnΨ∂α[c]*ε[c]
            if D == 3
                @inbounds ∂lnΨ∂βε[c] = ∂lnΨ∂β[c]*ε[c]
            end
        end
    end


    # EXECUTIONS:

    scatter_particles!() # sets up the quantum trap particles using the given method of scattering.
    while c < C # runs the given amount of Monte Carlo cycles with the given method of sampling.
        c += 1
        propose_move!()
        judge_move!()
        move_particles!()
        sample_quantities!()
    end

    if variation == "gradient descent" # returns energy and variational derivative samples if gradient descending.
        return ε , ε² , rejected_moves , ∂lnΨ∂α , ∂lnΨ∂αε , ∂lnΨ∂β , ∂lnΨ∂βε
    else
        return ε , ε² , rejected_moves # returns only energy samples if not gradient descending.
    end
end


function find_VMC_energy(trap::QuantumTrap, Ms::Vector{Int64}=[10^4,10^6];
        αs::Vector{Float64}=[0.5], βs::Vector{Float64}=[trap.λ],
        variation::String="gradient descent", scattering="normal", sampling::String="quantum drift",
        δv::Float64=0.001, δg::Float64=0.01, δs::Float64=√0.4, text_output::String="some", plot_output::String="none")
    # finds the VMC approximate ground state energy of the given quantum trap by varying the parameters α and β
    # using the given method of variation, and taking the given number of samples at each variational point
    # using the given methods of scattering and sampling.

    # CONSTANTS:

    D::Int64 = trap.D # is the dimension of the quantum trap.
    N::Int64 = trap.N # is the number of particles in the trap.
    a::Float64 = trap.a # is the characteristic radius of the particles.
    λ::Float64 = trap.λ # is the elliptical parameter of the trap.

    U::Int64 = length(αs) # is the number of α values to be considered (if range variation).
    V::Int64 = length(βs) # is the number of β values to be considered (if range variation).
    W::Int64 = length(Ms) # is the number of energies to calculate and store at each variational point
        # (used to plot convergence of the Monte Carlo sampling methods).

    αs = sort(αs) # sorts the vector of α values to be considered.
    βs = sort(βs) # sorts the vector of β values to be considered.

    T::Int64 = nworkers() # is the number of available threads for parallell calculation.

    Ms = sort(Ms) # sorts the vector of sample numbers to be considered.
    Ms = [(m-m%T) for m in Ms]
        # adjusts the numbers of sample numbers to be considered at each variational point to match the number of threads.
    M::Int64 = Ms[end] # is the number of samples to be taken in total at each variational point.
    B::Int64 = ceil(Int,log2(M))
        # is the 2-logarithm of the number of Monte Carlo cycles to be run in total at each variational point
        # (used to plot block resampling of statistical error for the final VMC energy.)


    # ASSERTIONS:

    if D < 1 || D > 3
        error("Provide a number of dimensions between 1 and 3 for the quantum trap.")
    end

    if N < 1
        error("Provide a positive number of particles in the quantum trap.")
    end

    if a < 0.0
        error("The characteristic radius for the particles in the quantum trap cannot be negative. ",
            "Provide a positive or zero value.")
    end

    if λ ≤ 0.0
        error("Provide a positive elliptical parameter for the quantum trap.")
    end

    if any(Ms .< 1)
        error("Provide only positive integers for the numbers of Monte Carlo cycles to be considered.")
    end

    if any(αs .≤ 0.0) || any(βs .≤ 0.0)
        error("Provide only positive values for the variational parameters α and β.")
    end

    if text_output ∉ ("full","some","none")
        error("The text output choice '",text_output,"' is not valid. Choose either 'full', 'some' or 'none'.")
    end

    if plot_output ∉ ("variation","convergence","resampling","none")
        error("The plot output choice '",plot_output,"' is not valid. ",
            "Choose either 'variation', 'convergence', 'configurations', 'resampling' or 'none'.")
    end

    if plot_output == "variation" && U == 1
        error("Too few values for α were provided. For variation plot output, provide at least 2 values for the parameter.")
    end

    if plot_output == "convergence" && W == 1
        error("Too few Monte Carlo cycle values were provided. ",
            "For convergence plot output, provide at least 2 numbers of Monte Carlo cycles to be considered.")
    end

    if variation ∉ ("range","gradient descent")
        error("The variation method '",variation,"' is not known. Choose either 'range' or 'gradient descent'.")
    end

    if scattering ∉ ("normal","lattice")
        error("The initial scattering method '",scattering,"' is not known. Choose either 'normal' or 'lattice'.")
    end

    if sampling ∉ ("random step","quantum drift")
        error("The sampling method '",sampling,"' is not known. Choose either 'random step' or 'quantum drift'.")
    end

    if variation == "gradient descent"
        if U > 1 || V > 1
            error("Too many values for α or β were provided. ",
                "For gradient descent variation, provide only one initial value for each parameter.")
        end
        if W < 2
            error("Too few Monte Carlo cycle values were provided. ",
                "For gradient descent variation, ",
                "provide at least one lowest number of Monte Carlo cycles to be considered during gradient descent ",
                "and one highest number of Monte Carlo cycles to be considered at the final variational point.")
        end
        if δv ≤ 0.0
            error("Provide a positive value for the gradient descent step size δv.")
        end
        if δg ≤ 0.0
            error("Provide a positive value for the gradient descent convergence threshold δg.")
        end
    end

    if δs ≤ 0.0
        error("Provide a positive value for the sampling step size δs.")
    end


    # VARIABLES:

    C::Int64 = round(M/T) # is the number of cycles to be run in total on each thread.

    α::Float64 = αs[1]
        # is the current variational trial state parameter α, as well as the final optimal value for α to be returned.
    β::Float64 = βs[1]
        # is the current variational trial state parameter β, as well as the final optimal value for β to be returned.

    ε::Vector{Float64} = zeros(M)
        # are the sampled local energies from each Monte Carlo cycle at the current variational point.
    ε²::Vector{Float64} = zeros(M)
        # are the sampled local energy squares from each Monte Carlo cycle at the current variational point.

    E::Float64 = 0.0 # is the calculated energy at the current variational point.
    ΔE²::Float64 = 0.0 # is the estimated statistical variance of the energy at the current variational point.
    ΔE::Float64 = 0.0 # is the estimated statistical error for the energy at the current variational point.

    if variation == "gradient descent"
        ∂lnΨ∂α::Vector{Float64} = zeros(M)
        ∂lnΨ∂αε::Vector{Float64} = zeros(M)
        ∂lnΨ∂β::Vector{Float64} = zeros(M)
        ∂lnΨ∂βε::Vector{Float64} = zeros(M)
            # are the sampled variational derivative quantities from each Monte Carlo cycle at the current variational point.

        ∂E∂α::Float64 = 0.0
        ∂E∂β::Float64 = 0.0
            # are the calculated variational derivatives at the current variational point.
    end

    results::Vector{Future} = [Future(t+1) for t in 1:T]

    rejected_moves::Vector{Int64} = zeros(T)
        # are the numbers of rejected moves at the current variational point because of random Metropolis acceptance.
    acceptance::Int64 = 0 # is the total percentage of Metropolis accepted moves at the current variational point.

    optimal_E::Float64 = NaN # is the currently optimal energy (used to store samples for block resampling).
    optimal_ε::Vector{Float64} = zeros(M)
        # are the sampled local energies from each Monte Carlo cycle at the currently optimal variational point
        # (used for block resampling).

    Es = zeros(U,V,W) # is the matrix of calculated energies at each variational point,
        # as well as the final vector of calculated VMC energies to be returned.
    ΔEs = zeros(U,V,W) # is the matrix of statistical error for the calculated energies,
        # as well as the final vector of statistical error for the VMC energies to be returned.

    if plot_output != "none"
        plout = plot() # is the plot to be displayed.
    end


    # FUNCTIONS:

    function flush_threads!()
        # flushes the caches of all threads.
        @everywhere GC.gc()
        GC.gc()
        @everywhere GC.gc()
        GC.gc()
    end

    function find_energy!(u::Int64=0,v::Int64=0)
        # runs a Monte Carlo simulation at the current variational point and stores the results in Es[u,v] and ΔEs[u,v]
        # (if not currently gradient descending, for which u and v is zero and no values are stored).

        # FUNCTIONS:

        function calculate_statistics!(m::Int64)
            # calculates the energy, the statistical variance and error of the energy as well as the variational derivatives
            # at the current variational point, including samples up to the given sample index.
            E = sum(ε[1:m])/m
            ΔE² = (sum(ε²[1:m])/m-E^2)/m
            if ΔE² < 0 # sets the statistical variance to zero if negative, but prints a warning.
                println("– Warning: The statistical variance turned out to be negative with value ",ΔE²,"! It was set to zero. –")
                ΔE² = 0.0
                ΔE = 0.0
            else
                ΔE = √ΔE²
            end
            if m == M
                if isnan(optimal_E) || E < optimal_E
                    # stores the energy and energy samples as optimal if they are lower than the currently optimal values
                    # (used for block resampling).
                    optimal_E = E
                    optimal_ε = ε
                end
            end
            if variation == "gradient descent"
                ∂E∂α = 2sum(∂lnΨ∂αε)/m-2sum(∂lnΨ∂α)/m*E
                if D == 3
                    ∂E∂β = 2sum(∂lnΨ∂βε)/m-2sum(∂lnΨ∂β)/m*E
                end
            end
        end


        # EXECUTIONS:

        if text_output == "full"
            println()
            println("Running ",T*C," Monte Carlo cycles at the variational point (",
            "α = ",round(α;digits=4),(D == 3 ? string(" / β = ",round(β;digits=4)) : ""),") ...")
        end
        if T > 1
            # if the main thread has access to other threads, sets up all these to run an equal number of Monte Carlo cycles,
            # collects the results and stores them in the sample vectors.
            for t in 1:T
                results[t] = @spawnat (t+1) simulate_n_sample(trap,C;α=α,β=β,
                    variation=variation,scattering=scattering,sampling=sampling,δv=δv,δg=δg,δs=δs)
            end
            for t in 1:T
                ms = (t-1)*C+1:t*C # are the sample indices at which to store the results.
                if variation == "gradient descent"
                    ε[ms] , ε²[ms] , rejected_moves[t] , ∂lnΨ∂α[ms] , ∂lnΨ∂αε[ms] , ∂lnΨ∂β[ms] , ∂lnΨ∂βε[ms] = fetch(results[t])
                else
                    ε[ms] , ε²[ms] , rejected_moves[t] = fetch(results[t])
                end
                finalize(results[t]) # releases the results from thread caches.
            end
        else
            # if the main thread has access to one or zero other threads, sets up to run the Monte Carlo cycles on its own.
            if variation == "gradient descent"
                ε , ε² , rejected_moves[1] , ∂lnΨ∂α , ∂lnΨ∂αε , ∂lnΨ∂β , ∂lnΨ∂βε = simulate_n_sample(trap,C;α=α,β=β,
                    variation=variation,scattering=scattering,sampling=sampling,δv=δv,δg=δg,δs=δs)
            else
                ε , ε² , rejected_moves[1] = simulate_n_sample(trap,C;α=α,β=β,
                    variation=variation,scattering=scattering,sampling=sampling,δv=δv,δg=δg,δs=δs)
            end
        end
        if u > 0 && v > 0 # calculates and stores energy values if indices u and v are specified.
            @inbounds for w in 1:W
                calculate_statistics!(Ms[w])
                Es[u,v,w] = E
                ΔEs[u,v,w] = ΔE
            end
            acceptance = round(100*(1-sum(rejected_moves)/M))
        elseif u == 0 || v == 0 # calculates energy values and variational derivatives if indices u and v are not specified.
            calculate_means!(Ms[1])
            acceptance = round(100*(1-sum(rejected_moves)/Ms[1]))
        end
        if acceptance == 0 # prints a warning if the acceptance turned out to be zero.
            println("– Warning: The acceptance turned out to be zero! –")
        end
        flush_threads!()
        if text_output == "full"
            println(T*M," Monte Carlo cycles finished!")
            println()
            println("Acceptance: ",acceptance,"%")
            println("Energy: ",round(E;digits=4)," ± ",round(ΔE;digits=4))
            println()
        end
    end

    function resample_energy!()
        # calculates a better estimate of the statistical error for the final VMC energy through block resampling.

        function van(samples::Vector{Float64})::Float64 # calculates the variance of the given samples.
            variance = sum(samples.^2)/length(samples)-mean(samples)^2
            if variance < 0 # sets the sample variance to zero if negative, but prints a warning.
                println("– Warning: The sample variance turned out to be negative with value ",variance,"! It was set to zero. –")
                return 0.0
            else
                return variance
            end
        end

        function block(samples::Vector{Float64})::Vector{Float64}
            # blocks the given samples into pairs and returns the mean values of each pair.
            N_samples = length(samples)
            if N_samples == 1
                return samples # returns the same sample if there is only one sample left.
            end
            blocked_samples = zeros(floor(Int,N_samples/2))
            @inbounds for i in 1:length(blocked_samples)
                blocked_samples[i] = (samples[2i-1]+samples[2i])/2
            end
            return blocked_samples
        end

        errors = zeros(B) # are the resampled statistical errors for the final VMC energy.
        b = 0 # is the number of blockings currently done.
        for w in 1:W
            samples = optimal_ε[1:Ms[w]]
            errors[1] = √(van(samples)/length(samples))
            b = 1
            samples = block(samples)
            errors[b+1] = √(van(samples)/length(samples))
            while errors[b+1] > errors[b]
                b += 1
                samples = block(samples)
                errors[b+1] = √(van(samples)/length(samples))
            end
            ΔEs[w] = errors[b]
        end
        ΔE = ΔEs[end]
        if plot_output == "resampling"
            plout = plot(title="Block resampling of VMC error for "*short_system_description(trap)*
                "<br>("*system_parameters(trap)*")<br>"*
                "α = "*string(round(α;digits=4))*(D == 3 ? string(" / β = ",round(β;digits=4)) : ""),
                xlabel="blockings",ylabel="VMC energy error [ħω]")
            @inbounds plot!(plout,0:b,errors[1:(b+1)];fillrange=zeros(b+1),fillalpha=.5,
                width=2,color="#fdce0b",alpha=.2,label=false)
        end
    end


    # EXECUTIONS:

    if text_output == "full" || text_output == "some" # prints parameters and methods if text output is set to 'full' or 'some'.
        println()
        println("Finding the VMC energy for ",long_system_description(trap),".")
        println()
        println("Quantum trap parameters: ",system_parameters(trap))
        println("Algorithm methods: ",uppercasefirst(variation)," variation / ",
            uppercasefirst(scattering)," scattering / ",uppercasefirst(sampling)," sampling")
        println("Algorithm parameters: ",
            (variation == "gradient descent" ? string("δv = ",round(δv;digits=4)," / δg = ",round(δg;digits=4)," / ") : ""),
            "δs = ",round(δs;digits=4))
        println()
    end
    if text_output == "some" # prints only variational parameter information if text output is set to 'some'.
        println()
        if variation == "range"
            println("Running ",U*V*M," Monte Carlo cycles at the variational point",
                (U == 1 && V == 1 ? string(" (α = ",round(α;digits=4),
                (D == 3 ? string(" / β = ",round(β;digits=4)) : "")) : string("s (α ∈ ",round.(αs;digits=4),
                (D == 3 ? string(" / β ∈ ",round.(βs;digits=4)) : "" ))),") ...")
        elseif variation == "gradient descent"
            println("Gradient descending from the initial variational point (",
            "α = ",round(α;digits=4),(D == 3 ? string(" / β = ",round(β;digits=4)) : ""),") ...")
        end
    end

    if variation == "range"
        @inbounds for u in 1:U, v in 1:V # scans through the given range of values for α (and β) and stores the results.
            α = αs[u]
            if D == 3
                β = βs[v]
            end
            find_energy!(u,v)
        end
        E,uv = findmin(Es[:,:,W])
        u = uv[1]
        v = uv[2]
        α = αs[u]
        β = βs[v]
        ΔE = ΔEs[u,v,W]
        if plot_output == "variation" # plots energies at the given values for α (and β) if plot output is set to 'variation'.
            if D < 3 || V == 1
                plout = plot(title="VMC variation for "*short_system_description(trap)*
                    "<br>("*system_parameters(trap)*(D == 3 ? string(" / β = ",round(β;digits=4)) : "")*")",
                    xlabel="α",ylabel="energy [ħω]")
                    plot!(plout,αs,Es[:,1,W];width=2,color="#fdce0b",label=false)
            else
                plout = plot(title="VMC variation for "*short_system_description(trap)*
                    "<br>("*system_parameters(trap)*")",xlabel="α",ylabel="β",zlabel="energy [ħω]")
                plot!(plout,αs,βs,Es[:,:,W];st=:surface,seriescolor=:sun,label=false)
            end
        end
        Es = Es[u,v,:]
        ΔEs = ΔEs[u,v,:]
    elseif variation == "gradient descent"
        C = round(Ms[1]/T) # sets the number of samples to be taken per thread to the minimum during gradient descent.
        find_energy!()
        while ∂E∂α^2 > δg^2 || D == 3 && ∂E∂β^2 > δg^2
            # continues gradient descending as long as the variational derivatives are larger than the given threshold.
            while ∂E∂α*δv ≥ α || D == 3 && ∂E∂β*δv ≥ β # ensures that negative values for α (and β) are never considered.
                ∂E∂α *= 0.5
                if D == 3
                    ∂E∂β *= 0.5
                end
            end
            α -= ∂E∂α*δv
            if D == 3
                β -= ∂E∂β*δv
            end
            find_energy!()
        end
        C = round(M/T) # sets the numbers of samples per thread back to maximum at the final variational point.
        variation = "range"
            # switches to range variation so that variational derivatives are not calculated at this final variational point.
        if text_output == "some"
            println("Running ",M," Monte Carlo cycles at the final variational point (",
            "α = ",round(α;digits=4),(D == 3 ? string(" / β = ",round(β;digits=4)) : ""),") ...")
        end
        find_energy!(1,1)
        Es = Es[1,1,:]
        ΔEs = ΔEs[1,1,:]
    end
    resample_energy!()

    if text_output == "some"
        println("VMC simulation finished!")
        println()
    end
    if text_output == "full" || text_output == "some"
            # prints the optimal parameters α and β as well as the VMC energy if text output is set to 'full' or 'some'.
        println()
        println("Optimal α: ",round(α;digits=4))
        if D == 3
            println("Optimal β: ",round(β;digits=4))
        end
        println("VMC energy: ",round(E;digits=4)," ± ",round(ΔE;digits=4))
        println()
    end

    if plot_output == "convergence"
        if sampling == "random step" # sets the plot colour to teal for random step sampling.
            sampling_colour = "#4aa888"
        elseif sampling == "quantum drift" # sets the plot colour to purple for quantum drift sampling.
            sampling_colour = "#aa4888"
        end
        plout = plot(title="Convergence of VMC sampling for "*short_system_description(trap)*
            "<br>("*system_parameters(trap)*" / δs = "*string(round(δs;digits=4))*")<br>"*
            "α = "*string(round(α;digits=4))*(D == 3 ? string(" / β = ",round(β;digits=4)) : ""),
            legend=:bottomright,xlabel="Monte Carlo cycles",xaxis=:log,ylabel="VMC energy [ħω]")
        plot!(plout,cycles,Es;ribbon=ΔEs,fillalpha=.5,width=2,color=sampling_colour,label=sampling*" sampling")
        plot!(plout,cycles,[E for i in 1:W];style=:dash,width=2,color="#fdce0b",label="reference VMC energy")
    end
    if plot_output != "none"
        display(plout)
    end

    return α,β,Es,ΔEs
end


function compare_VMC_sampling(trap::QuantumTrap, Ms::Vector{Int64}=[10^e for e in 1:7];
        α::Float64=0.5, β::Float64=trap.λ,δs::Float64=√0.4)
    # compares the two VMC sampling methods by plotting their results against given numbers of Monte Carlo cycles.

    # CONSTANTS:

    D::Int64 = trap.D # is the dimension of the quantum trap.

    W::Int64 = length(Ms) # is the number of energies to calculate and store in order to compare convergence
        # of the Monte Carlo sampling methods.

    T::Int64 = nworkers() # is the number of available threads for parallell calculation.

    Ms = [(m-m%T) for m in Ms] # adjusts the numbers of samples to be considered to match threads.
    M::Int64 = Ms[end] # is the number of samples to be taken in total for each sampling method.


    # VARIABLES

    Es_RS::Vector{Float64} = zeros(W) # is the vector of calculated energies from random step sampling.
    ΔEs_RS::Vector{Float64} = zeros(W) # is the vector of statistical error for the calculated energies
        # from random step sampling.
    Es_QD::Vector{Float64} = zeros(W) # is the vector of calculated energies from quantum drift sampling.
    ΔEs_QD::Vector{Float64} = zeros(W) # is the vector of statistical error for the calculated energies
        # from quantum drift sampling.


    # EXECUTIONS:

    println()
    println("Comparing VMC sampling methods for ",long_system_description(trap),".")
    println()
    println("Quantum trap parameters: ",system_parameters(trap))
    println("Algorithm methods: Normal scattering / Random step & Quantum drift sampling")
    println("Algorithm parameters: δs = ",round(δs;digits=4))
    println("Variational parameters: α = ",round(α;digits=4),(D == 3 ? string(" / β = ",round(β;digits=4)) : ""))
    println()
    println()
    println("Running ",M," Monte Carlo cycles with random step sampling ...")
    _,_,Es_RS,ΔEs_RS = find_VMC_energy(trap,Ms;αs=[α],βs=[β],sampling="random step",δs=δs,text_output="none")
    println("Running ",M," Monte Carlo cycles with quantum drift sampling ...")
    _,_,Es_QD,ΔEs_QD = find_VMC_energy(trap,Ms;αs=[α],βs=[β],sampling="quantum drift",δs=δs,text_output="none")
    println("VMC sampling comparison finished!")
    println()
    E = Es_QD[end]
    comparison = plot(title="Comparison of VMC sampling for "*short_system_description(trap)*
        "<br>("*system_parameters(trap)*" / δs = "*string(round(δs;digits=4))*")<br>"*
        "α = "*string(round(α;digits=4))*(D == 3 ? string(" / β = ",round(β;digits=4)) : ""),
        legend=:bottomright,xlabel="Monte Carlo cycles",xaxis=:log,ylabel="energy [ħω]")
    plot!(comparison,Ms,Es_RS;ribbon=ΔEs_RS,fillalpha=.5,width=2,color="#4aa888",label="random step sampling")
    plot!(comparison,Ms,Es_QD;ribbon=ΔEs_QD,fillalpha=.5,width=2,color="#aa4888",label="quantum drift sampling")
    plot!(comparison,Ms,fill(E,W);style=:dash,width=2,color="#fdce0b",label="reference energy")
    display(comparison)
    return
end

; # suppresses inclusion output.
