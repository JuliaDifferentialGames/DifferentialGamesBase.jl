# ============================================================================
# Phase 6: Stochastic dynamics core, chance constraints, determinize()
# ============================================================================

@testset "Phase 6a: ItoSDEDynamics / StochasticDiscreteApproximation (exact linear path)" begin
    A = [1.0 1.0; 0.0 1.0]
    B = [reshape([0.0, 1.0], 2, 1)]
    drift = LinearDynamics(A, B)
    G = [0.1 0.0; 0.0 0.1]
    sde = ItoSDEDynamics(drift, constant_diffusion(G), 2)

    @test total_state_dim(sde) == 2
    @test total_control_dim(sde) == 1
    @test is_constant_diffusion(sde)
    @test !is_constant_diffusion(ItoSDEDynamics(drift, (x,u,p,t)->G, 2))

    sd = discretize_stochastic(sde, 1.0)
    x = [1.0, 0.5]; u = [0.2]
    @test mean_step(sd, x, u, nothing, 1) ≈ A*x + B[1]*u
    Jx, Ju = jacobian(sd, x, u, nothing, 1)
    @test Jx ≈ A
    Q = process_covariance(sd, x, u, nothing, 1)
    @test Q ≈ 1.0 .* (G*G')
end

@testset "Phase 6b: ItoSDEDynamics nonlinear drift — linearized covariance path" begin
    f = (x, u, p, t) -> [x[2]; u[1]]
    drift = CoupledNonlinearDynamics(f, 2, 1)
    G = [0.0; 0.1]
    sde = ItoSDEDynamics(drift, constant_diffusion(reshape(G, 2, 1)), 1)
    sd = discretize_stochastic(sde, 0.1)
    x = [1.0, 0.5]; u = [0.2]
    xn = mean_step(sd, x, u, nothing, 0.0)
    @test length(xn) == 2
    Jx, Ju = jacobian(sd, x, u, nothing, 0.0)
    @test size(Jx) == (2, 2)
    Q = process_covariance(sd, x, u, nothing, 0.0)
    @test size(Q) == (2, 2)
end

@testset "Phase 6c: StochasticGameProblem / ProcessNoiseModel rework (backward-compatible)" begin
    A = [1.0 1.0; 0.0 1.0]
    B = [reshape([0.0, 1.0], 2, 1)]
    Q = diagm([1.0, 1.0]); R = reshape([0.1], 1, 1); Qf = diagm([10.0, 10.0])
    g = LQGameProblem(A, B, [Q], [R], [Qf], [1.0, 0.0], 2.0; dt = 1.0)

    Σ = diagm([0.01, 0.01])
    sg = StochasticGameProblem(g, ProcessNoiseModel(Σ))

    @test sg.stochastic_dynamics isa ItoSDEDynamics
    @test is_certainty_equivalent(sg)
    @test !has_multiplicative_noise(sg)
    @test get_W(sg) ≈ Σ
    @test n_players(sg) == 1
end

@testset "Phase 6d: ChanceConstraint + GaussianQuantileEvaluator + LinearizedGaussianPropagation" begin
    inner = PrivateInequality(1; func = (x,u,p,t) -> [x[1] - 2.0], dim = 1, depends_on_control = false)
    cc = ChanceConstraint(inner, 0.05)
    @test is_private(cc)
    @test get_player(cc) == 1
    @test evaluate_constraint(cc, [1.0, 0.0], Float64[], nothing, 0.0) == [-1.0]

    ev = GaussianQuantileEvaluator()
    # Nominal value 0, variance 1, risk 0.5 -> Φ⁻¹(0.5) = 0 -> residual = 0
    @test isapprox(chance_residual(ev, 0.0, 1.0, 0.5), 0.0; atol = 1e-6)
    # Smaller risk -> larger (more conservative, positive) quantile contribution
    @test chance_residual(ev, 0.0, 1.0, 0.05) > chance_residual(ev, 0.0, 1.0, 0.3)

    policy = UniformRiskAllocation()
    @test allocate_risk(policy, cc) == [0.05]
end

@testset "Phase 6e: determinize() round-trip" begin
    A = [1.0 1.0; 0.0 1.0]
    B = [reshape([0.0, 1.0], 2, 1)]
    Qc = diagm([1.0, 1.0]); R = reshape([0.1], 1, 1); Qf = diagm([10.0, 10.0])
    g = LQGameProblem(A, B, [Qc], [R], [Qf], [1.0, 0.0], 2.0; dt = 1.0)

    cc = ChanceConstraint(
        PrivateInequality(1; func = (x,u,p,t) -> [x[1] - 2.0], dim = 1, depends_on_control = false),
        0.05)
    g2 = remake(g; private_constraints = [cc])

    Σ = diagm([0.01, 0.01])
    sg = StochasticGameProblem(g2, ProcessNoiseModel(Σ))
    detg = determinize(sg)

    @test detg isa GameProblem
    @test total_state_dim(detg.dynamics) == 2 + 2^2
    @test length(detg.initial_state) == 2 + 2^2
    @test length(detg.private_constraints) == 1
    # The compiled constraint's residual is evaluable on the augmented state.
    c = detg.private_constraints[1]
    v = evaluate_constraint(c, detg.initial_state, [0.0], nothing, 0.0)
    @test length(v) == 1
end
