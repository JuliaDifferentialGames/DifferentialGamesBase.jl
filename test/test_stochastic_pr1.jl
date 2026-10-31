# PR 1 regression tests for the stochastic-games overhaul: PSD covariance,
# normal-quantile accuracy, and the expected-cost trace term in `determinize`.

@testset "Stochastic overhaul PR1" begin
    @testset "normal quantile matches reference values" begin
        q = DifferentialGamesBase._norm_quantile
        @test q(0.5) ≈ 0.0 atol = 1e-15
        @test q(0.95) ≈ 1.6448536269514722 rtol = 1e-12
        @test q(0.975) ≈ 1.959963984540054 rtol = 1e-12
        @test q(0.999) ≈ 3.090232306167813 rtol = 1e-12
        @test q(1e-10) ≈ -6.361340902404056 rtol = 1e-12
        @test q(0.05) ≈ -q(0.95) rtol = 1e-14
        @test q(0.0) == -Inf && q(1.0) == Inf
    end

    @testset "chance residual reduces to deterministic constraint at zero variance" begin
        ev = GaussianQuantileEvaluator()
        @test chance_residual(ev, -0.3, 0.0, 0.05) == -0.3
        @test chance_residual(ev, 0.0, 4.0, 0.05) ≈ 2 * 1.6448536269514722 rtol = 1e-12
    end

    @testset "singular and zero process covariance are accepted" begin
        @test ProcessNoiseModel(zeros(2, 2)) isa ProcessNoiseModel
        @test ProcessNoiseModel([1.0 0.0; 0.0 0.0]) isa ProcessNoiseModel
        @test_throws AssertionError ProcessNoiseModel([1.0 0.0; 0.0 -1.0])
        @test_throws AssertionError ProcessNoiseModel([1.0 2.0; 0.0 1.0])

        A = [1.0 0.1; 0.0 1.0]
        B = [reshape([0.0, 0.1], 2, 1)]
        g = LQGameProblem(A, B, [Matrix(1.0I, 2, 2)], [reshape([1.0], 1, 1)],
                          [Matrix(1.0I, 2, 2)], [1.0, 0.0], 2.0; dt = 1.0)
        W = [0.04 0.0; 0.0 0.0]
        sg = StochasticGameProblem(g, ProcessNoiseModel(W))
        @test get_W(sg) ≈ W atol = 1e-14
        sd = discretize_stochastic(sg.stochastic_dynamics, 1.0)
        @test process_covariance(sd, zeros(2), zeros(1), nothing, 1) ≈ W atol = 1e-14
        @test all(isfinite, sample_step(sd, zeros(2), zeros(1), nothing, 1))
    end

    @testset "determinize adds the ½tr(QΣ) expected-cost term" begin
        A = [1.0 0.0; 0.0 1.0]
        B = [reshape([1.0, 0.0], 2, 1)]
        Q = [2.0 0.5; 0.5 1.0]
        Qf = [3.0 0.0; 0.0 1.0]
        g = LQGameProblem(A, B, [Q], [reshape([1.0], 1, 1)], [Qf], [0.0, 0.0], 2.0; dt = 1.0)
        sg = StochasticGameProblem(g, ProcessNoiseModel(Matrix(0.1I, 2, 2)))
        dg = determinize(sg)
        Σ = [0.3 0.1; 0.1 0.2]
        μ = [0.4, -0.2]
        z = vcat(μ, vec(Σ))
        obj = dg.objectives[1]
        stage = evaluate_stage_cost(obj.stage_cost, z, [0.0], nothing, 1)
        @test stage ≈ 0.5 * μ' * Q * μ + 0.5 * tr(Q * Σ)
        term = evaluate_terminal_cost(obj.terminal_cost, z, nothing)
        @test term ≈ 0.5 * μ' * Qf * μ + 0.5 * tr(Qf * Σ)
    end
end
