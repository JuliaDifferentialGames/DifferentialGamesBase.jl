# PR 4: chance-constraint risk allocation, guarantee classes, information patterns.

@testset "Stochastic overhaul PR4" begin
    @testset "standard-normal CDF" begin
        cdf = DifferentialGamesBase._norm_cdf
        q = DifferentialGamesBase._norm_quantile
        @test cdf(0.0) ≈ 0.5 atol = 1e-15
        @test cdf(1.6448536269514722) ≈ 0.95 rtol = 1e-12
        @test cdf(-3.090232306167813) ≈ 0.001 rtol = 1e-10
        @test cdf(-6.0) ≈ 9.865876450376946e-10 rtol = 1e-8   # continued-fraction branch
        for p in (1e-8, 1e-3, 0.2, 0.5, 0.9, 0.999)
            @test cdf(q(p)) ≈ p rtol = 1e-9
        end
    end

    @testset "exact affine-Gaussian violation probability matches the quantile residual" begin
        ev = GaussianQuantileEvaluator()
        for ε in (0.01, 0.05, 0.2), σ in (0.3, 2.0)
            # residual exactly 0 <=> violation probability exactly ε
            μ = -DifferentialGamesBase._norm_quantile(1 - ε) * σ
            @test chance_residual(ev, μ, σ^2, ε) ≈ 0 atol = 1e-12
            @test gaussian_violation_probability(μ, σ^2) ≈ ε rtol = 1e-9
        end
        @test gaussian_violation_probability(0.1, 0.0) == 1.0
        @test gaussian_violation_probability(-0.1, 0.0) == 0.0
    end

    @testset "joint risk budget: union bound over every (constraint, step, component)" begin
        c1 = ChanceConstraint(StateBounds(1; state_offset = 0, state_dim = 2, lower = [-1.0, -1.0], upper = [1.0, 1.0]), 0.05)
        c2 = ChanceConstraint(ControlBounds(1; control_offset = 0, control_dim = 1, lower = [-2.0], upper = [2.0]), 0.05)
        budget = JointRiskBudget(0.1)
        eps = allocate_horizon_risk(UniformRiskAllocation(), [c1, c2], 10, budget)
        n_events = (constraint_output_dim(c1) + constraint_output_dim(c2)) * 10
        @test all(e -> all(==(0.1 / n_events), e), eps)
        @test joint_risk_bound(eps) ≈ 0.1
        @test check_risk_budget(eps, budget)
        @test !check_risk_budget([e .* 1.5 for e in eps], budget)
        @test_throws AssertionError JointRiskBudget(1.5)
    end

    @testset "guarantee classes" begin
        aff = ChanceConstraint(StateBounds(1; state_offset = 0, state_dim = 1, lower = [-1.0], upper = [1.0]), 0.05)
        nl = ChanceConstraint(PrivateInequality(1; func = (x, u, p, t) -> [x[1]^2 - 1.0], dim = 1), 0.05)
        @test is_affine_constraint(aff) && !is_affine_constraint(nl)
        ev = GaussianQuantileEvaluator()
        @test guarantee_class(aff, ev) === :exact_affine_gaussian
        @test guarantee_class(nl, ev) === :linearized_approximation
    end

    @testset "checks on Gaussian moments agree with Monte Carlo (affine ⇒ exact)" begin
        cc = ChanceConstraint(StateBounds(1; state_offset = 0, state_dim = 2, lower = [-5.0, -5.0], upper = [1.0, 5.0]), 0.05)
        dim = constraint_output_dim(cc)
        steps = 2
        xm = [0.2 0.4; 0.0 0.0]; um = zeros(1, steps)
        Σx = [[0.09 0.0; 0.0 0.01], [0.16 0.0; 0.0 0.01]]
        Σu = [zeros(1, 1) for _ in 1:steps]; Σxu = [zeros(2, 1) for _ in 1:steps]
        checks = check_chance_constraints([cc], xm, um, Σx, Σu, Σxu)
        @test all(c -> c.guarantee === :exact_affine_gaussian, checks)
        # component 3 is x₁ - 1 ≤ 0; first-state upper bound at step k: P(x1 > 1) = 1 - Φ((1-μ)/σ)
        for c in filter(c -> c.component == 3, checks)
            μ = xm[1, c.step]; σ = sqrt(Σx[c.step][1, 1])
            @test c.violation_probability ≈ 1 - DifferentialGamesBase._norm_cdf((1 - μ) / σ) rtol = 1e-8
        end
        # Monte Carlo of the same events
        rng = Random.MersenneTwister(5); ns = 200_000
        g = [xm[1, k] + sqrt(Σx[k][1, 1]) * randn(rng) - 1.0 for k in 1:steps, _ in 1:ns]
        emp = empirical_violation_rates(g)
        for k in 1:steps
            p = filter(c -> c.component == 3 && c.step == k, checks)[1].violation_probability
            lo, hi = emp.per_event_interval[k]
            @test lo - 1e-3 <= p <= hi + 1e-3
        end
        @test emp.any_violation >= maximum(emp.per_event)
        @test joint_violation_bound(filter(c -> c.component == 3, checks)) >= emp.any_violation - 5e-3
    end

    @testset "StochasticGameProblem carries information, chance constraints, budget" begin
        A = [1.0 0.1; 0.0 1.0]; B = [reshape([0.0, 0.1], 2, 1)]
        g = LQGameProblem(A, B, [Matrix(1.0I, 2, 2)], [reshape([1.0], 1, 1)],
                          [Matrix(1.0I, 2, 2)], [0.0, 0.0], 3.0; dt = 1.0)
        cc = ChanceConstraint(StateBounds(1; state_offset = 0, state_dim = 2, lower = [-5.0, -5.0], upper = [1.0, 5.0]), 0.05)
        sg0 = StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.01I, 2, 2)))
        @test sg0.information isa PerfectStateInformation
        @test isempty(all_chance_constraints(sg0))
        sg = StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.01I, 2, 2));
                                   chance_constraints = [cc], joint_risk = JointRiskBudget(0.1))
        @test all_chance_constraints(sg) == [cc]
        bad = ChanceConstraint(StateBounds(5; state_offset = 0, state_dim = 2, lower = [-5.0, -5.0], upper = [1.0, 5.0]), 0.05)
        @test_throws AssertionError StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.01I, 2, 2));
                                                          chance_constraints = [bad])
        # determinize compiles the declared constraint and applies the per-step budget
        dg = determinize(sg)
        @test length(dg.private_constraints) == 1
        z = vcat([0.0, 0.0], vec(zeros(2, 2)))
        r1 = evaluate_constraint(dg.private_constraints[1], z, [0.0], nothing, 1)
        @test all(isfinite, r1)
    end
end

@testset "Stochastic overhaul PR6: Kalman helpers" begin
    @testset "Joseph form equals the standard update when K is optimal" begin
        Σ⁻ = [0.5 0.1; 0.1 0.3]; H = [1.0 0.0]; V = reshape([0.2], 1, 1)
        K = kalman_gain(Σ⁻, H, V)
        std = Σ⁻ - K * H * Σ⁻
        @test kalman_joseph(Σ⁻, K, H, V) ≈ std atol = 1e-14
        # a suboptimal gain still gives a valid (PSD, symmetric) covariance via Joseph form
        J = kalman_joseph(Σ⁻, 0.7 .* K, H, V)
        @test issymmetric(J) && isposdef(J)
    end
    @testset "observation helpers" begin
        obs = linear_observation([1.0 0.0; 0.0 2.0], Matrix(0.1I, 2, 2))
        @test observation_matrix(obs, zeros(2), nothing, 1) ≈ [1.0 0.0; 0.0 2.0]
        @test is_affine_observation(obs, 2)
        sq = SharedObservation((x, v, p, t) -> x[1:1] .^ 2 .+ v, 1, reshape([0.1], 1, 1))
        @test !is_affine_observation(sq, 2)
    end
    @testset "common-information moments: no information leaves open-loop covariance growth" begin
        A = [[1.0 0.1; 0.0 1.0] for _ in 1:3]; B = [reshape([0.0, 0.1], 2, 1) for _ in 1:3]
        P = [zeros(1, 2) for _ in 1:3]; W = [0.01 * Matrix(1.0I, 2, 2) for _ in 1:3]
        H = [[1.0 0.0] for _ in 1:4]
        m = common_information_moments(A, B, P, W, H, reshape([1e12], 1, 1), zeros(2, 2))
        Σ = zeros(2, 2)
        for k in 1:3
            Σ = A[k] * Σ * A[k]' + W[k]
        end
        @test m.state_covariances[end] ≈ Σ rtol = 1e-6   # P = 0: physical covariance is open-loop
    end
end

@testset "Stochastic API docstrings" begin
    names = [:JointLQStageCost, :JointLQGameProblem, :DiscreteGaussianNoise, :StochasticGameProblem,
             :all_chance_constraints, :JointRiskBudget, :allocate_horizon_risk, :joint_risk_bound,
             :check_risk_budget, :is_affine_constraint, :guarantee_class, :gaussian_violation_probability,
             :ChanceCheck, :check_chance_constraints, :joint_violation_bound, :empirical_violation_rates,
             :linear_observation, :observation_matrix, :is_affine_observation, :kalman_gain,
             :kalman_joseph, :common_information_moments, :ChanceConstraint, :determinize]
    missing_docs = [n for n in names if !Base.Docs.hasdoc(DifferentialGamesBase, n)]
    @test isempty(missing_docs)
end
