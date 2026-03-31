# ============================================================================
# test_dynamics_and_problem.jl
#
# Test hierarchy:
#   L1 — Unit: LinearDynamics type construction and accessors
#   L2 — Unit: LQStageCost LTI/LTV construction and accessors
#   L3 — Unit: GameProblem structural queries
#   L4 — Integration: LTVLQGameProblem construction and consistency
#   L5 — Regression: LTI path produces identical results to old LinearDynamics
#   L6 — Type stability: @inferred checks on all hot-path accessors
# ============================================================================

using Test
using LinearAlgebra

# Helpers — build canonical LTI problem data
function lti_data(T=Float64; n=4, m=2, n_players=2)
    A  = Matrix{T}(I(n)) .+ T(0.1) * randn(T, n, n)
    B  = [randn(T, n, m) for _ in 1:n_players]
    Q  = [Matrix{T}(I(n)) for _ in 1:n_players]
    R  = [Matrix{T}(I(m)) for _ in 1:n_players]
    Qf = [T(10) * Matrix{T}(I(n)) for _ in 1:n_players]
    x0 = randn(T, n)
    (A=A, B=B, Q=Q, R=R, Qf=Qf, x0=x0, n=n, m=m, n_players=n_players)
end

# Helpers — build canonical LTV problem data by perturbing LTI per timestep
function ltv_data(T=Float64; n=4, m=2, n_players=2, N=10)
    d = lti_data(T; n=n, m=m, n_players=n_players)
    A_seq  = [d.A .+ T(0.01) * randn(T, n, n) for _ in 1:N]
    B_seq  = [[d.B[i] .+ T(0.01) * randn(T, n, m) for _ in 1:N] for i in 1:n_players]
    Q_seq  = [[Matrix{T}(I(n)) for _ in 1:N] for _ in 1:n_players]
    R_seq  = [[Matrix{T}(I(m)) for _ in 1:N] for _ in 1:n_players]
    (A_seq=A_seq, B_seq=B_seq, Q_seq=Q_seq, R_seq=R_seq, Qf=d.Qf, x0=d.x0,
     n=n, m=m, n_players=n_players, N=N)
end

@testset "DifferentialGamesBase — Dynamics and Problem Types" begin

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L1 — LinearDynamics construction" begin

        @testset "LTI: basic construction" begin
            d = lti_data()
            dyn = LinearDynamics(d.A, d.B)
            @test dyn.state_dim == d.n
            @test dyn.control_dims == [d.m, d.m]
            @test isnothing(dyn.n_steps)
            @test !is_ltv(dyn)
        end

        @testset "LTV: basic construction" begin
            d = ltv_data(N=10)
            dyn = LinearDynamics(d.A_seq, d.B_seq)
            @test dyn.state_dim == d.n
            @test dyn.control_dims == [d.m, d.m]
            @test dyn.n_steps == 10
            @test is_ltv(dyn)
        end

        @testset "LTI: A not square — should error" begin
            A_bad = randn(4, 3)
            B = [randn(4, 2)]
            @test_throws AssertionError LinearDynamics(A_bad, B)
        end

        @testset "LTI: B row mismatch — should error" begin
            A = Matrix{Float64}(I(4))
            B = [randn(3, 2)]   # wrong number of rows
            @test_throws AssertionError LinearDynamics(A, B)
        end

        @testset "LTV: sequence length mismatch — should error" begin
            d = ltv_data(N=10)
            B_bad = deepcopy(d.B_seq)
            push!(B_bad[1], randn(d.n, d.m))  # player 1 has N+1 matrices
            @test_throws AssertionError LinearDynamics(d.A_seq, B_bad)
        end

        @testset "LTV: inconsistent control dim — should error" begin
            d = ltv_data(N=5)
            B_bad = deepcopy(d.B_seq)
            B_bad[1][3] = randn(d.n, d.m + 1)  # player 1, step 3 has wrong dim
            @test_throws AssertionError LinearDynamics(d.A_seq, B_bad)
        end

    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L1 — LinearDynamics accessors" begin

        @testset "LTI: get_A ignores k" begin
            d = lti_data()
            dyn = LinearDynamics(d.A, d.B)
            for k in [1, 5, 100]
                @test get_A(dyn, k) === dyn.A  # identity: same object, no copy
            end
        end

        @testset "LTI: get_B(i, k) ignores k" begin
            d = lti_data()
            dyn = LinearDynamics(d.A, d.B)
            for k in [1, 5]
                @test get_B(dyn, 1, k) === dyn.B[1]
                @test get_B(dyn, 2, k) === dyn.B[2]
            end
        end

        @testset "LTV: get_A(k) returns correct matrix" begin
            d = ltv_data(N=10)
            dyn = LinearDynamics(d.A_seq, d.B_seq)
            for k in 1:10
                @test get_A(dyn, k) === d.A_seq[k]
            end
        end

        @testset "LTV: get_B(i, k) returns correct matrix" begin
            d = ltv_data(N=10)
            dyn = LinearDynamics(d.A_seq, d.B_seq)
            for i in 1:d.n_players, k in 1:10
                @test get_B(dyn, i, k) === d.B_seq[i][k]
            end
        end

        @testset "get_B_concatenated: correct dimensions" begin
            d = lti_data()
            dyn = LinearDynamics(d.A, d.B)
            B_cat = get_B_concatenated(dyn, 1)
            @test size(B_cat) == (d.n, d.n_players * d.m)
        end

    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L2 — LQStageCost construction and accessors" begin

        @testset "LTI: basic construction" begin
            n, m = 4, 2
            Q  = Matrix{Float64}(I(n))
            R  = Matrix{Float64}(I(m))
            M  = zeros(n, m)
            q  = zeros(n)
            r  = zeros(m)
            c  = 0.0
            cost = LQStageCost(Q, R, M, q, r, c)
            @test !is_ltv(cost)
            # Accessors return the same object for all k
            for k in [1, 5, 100]
                @test get_Q(cost, k) === Q
                @test get_R(cost, k) === R
            end
        end

        @testset "LTI convenience: omit optional terms" begin
            n, m = 4, 2
            cost = LQStageCost(Matrix{Float64}(I(n)), Matrix{Float64}(I(m)))
            @test !is_ltv(cost)
            @test get_Q(cost, 1) == I(n)
        end

        @testset "LTI: non-symmetric Q — should error" begin
            Q = [1.0 2.0; 3.0 4.0]
            R = Matrix{Float64}(I(2))
            @test_throws AssertionError LQStageCost(Q, R)
        end

        @testset "LTI: non-posdef R — should error" begin
            Q = Matrix{Float64}(I(2))
            R = -Matrix{Float64}(I(2))
            @test_throws AssertionError LQStageCost(Q, R)
        end

        @testset "LTV: basic construction and accessors" begin
            n, m, N = 4, 2, 8
            Q_seq = [Matrix{Float64}(I(n)) for _ in 1:N]
            R_seq = [Matrix{Float64}(I(m)) for _ in 1:N]
            cost = LQStageCost(Q_seq, R_seq)
            @test is_ltv(cost)
            for k in 1:N
                @test get_Q(cost, k) === Q_seq[k]
                @test get_R(cost, k) === R_seq[k]
            end
        end

        @testset "LTV: sequence length mismatch — should error" begin
            n, m, N = 4, 2, 5
            Q_seq = [Matrix{Float64}(I(n)) for _ in 1:N]
            R_seq = [Matrix{Float64}(I(m)) for _ in 1:N+1]  # wrong length
            @test_throws AssertionError LQStageCost(Q_seq, R_seq,
                [zeros(n, m) for _ in 1:N], [zeros(n) for _ in 1:N], [zeros(m) for _ in 1:N])
        end

    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L3 — GameProblem structural queries" begin

        @testset "LQGameProblem: is_lq_game, is_unconstrained, !is_ltv" begin
            d = lti_data()
            game = LQGameProblem(d.A, d.B, d.Q, d.R, d.Qf, d.x0, 1.0; dt=0.1)
            @test is_lq_game(game)
            @test is_unconstrained(game)
            @test !is_ltv(game.dynamics)
            @test !is_pd_gnep(game)
            @test state_dim(game) == d.n
            @test control_dim(game) == d.n_players * d.m
            @test n_steps(game) == 10
        end

        @testset "LTVLQGameProblem: is_lq_game, is_ltv" begin
            d = ltv_data(N=10)
            tf = 1.0; dt = 0.1
            game = LTVLQGameProblem(d.A_seq, d.B_seq, d.Q_seq, d.R_seq, d.Qf, d.x0, tf; dt=dt)
            @test is_lq_game(game)
            @test is_ltv(game.dynamics)
            @test is_unconstrained(game)
            @test n_steps(game) == 10
        end

    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L4 — LTVLQGameProblem consistency" begin

        @testset "N / dt consistency check" begin
            d = ltv_data(N=10)
            # tf/dt = 5/0.1 = 50 ≠ 10 — should error
            @test_throws AssertionError LTVLQGameProblem(
                d.A_seq, d.B_seq, d.Q_seq, d.R_seq, d.Qf, d.x0, 5.0; dt=0.1
            )
        end

        @testset "Dynamics accessor round-trip through GameProblem" begin
            d = ltv_data(N=10)
            tf = 1.0; dt = 0.1
            game = LTVLQGameProblem(d.A_seq, d.B_seq, d.Q_seq, d.R_seq, d.Qf, d.x0, tf; dt=dt)
            dyn = game.dynamics
            for k in 1:10
                @test get_A(dyn, k) === d.A_seq[k]
                for i in 1:d.n_players
                    @test get_B(dyn, i, k) === d.B_seq[i][k]
                end
            end
        end

        @testset "Cost accessor round-trip through GameProblem" begin
            d = ltv_data(N=10)
            tf = 1.0; dt = 0.1
            game = LTVLQGameProblem(d.A_seq, d.B_seq, d.Q_seq, d.R_seq, d.Qf, d.x0, tf; dt=dt)
            for i in 1:d.n_players
                sc = game.objectives[i].stage_cost
                for k in 1:10
                    @test get_Q(sc, k) === d.Q_seq[i][k]
                    @test get_R(sc, k) === d.R_seq[i][k]
                end
                tc = game.objectives[i].terminal_cost
                @test tc.Qf === d.Qf[i]
            end
        end

    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L5 — Regression: LTI path identical to previous LinearDynamics" begin
        #
        # Construct the same problem as the old LQGameProblem and verify all
        # fields and accessor outputs are numerically identical. This guards
        # against the unified type introducing any value change on the LTI path.
        #
        d = lti_data()
        game = LQGameProblem(d.A, d.B, d.Q, d.R, d.Qf, d.x0, 1.0; dt=0.1)
        dyn = game.dynamics

        # Accessor values must match input matrices exactly (not copies)
        @test get_A(dyn, 1) == d.A
        @test get_A(dyn, 7) == d.A   # k ignored for LTI
        for i in 1:d.n_players
            @test get_B(dyn, i, 1) == d.B[i]
            @test get_B(dyn, i, 7) == d.B[i]
        end

        # Cost matrices
        for i in 1:d.n_players
            sc = game.objectives[i].stage_cost
            @test get_Q(sc, 1) == d.Q[i]
            @test get_R(sc, 1) == d.R[i]
            tc = game.objectives[i].terminal_cost
            @test tc.Qf == d.Qf[i]
        end

        # Metadata preserved
        @test game.n_players == d.n_players
        @test state_dim(game) == d.n
        @test control_dim(game) == d.n_players * d.m
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "L6 — Type stability (@inferred)" begin
        #
        # Every hot-path accessor must be type-stable. Failures here surface
        # the Union-field failure mode before it causes convergence regressions.
        #
        d_lti = lti_data()
        d_ltv = ltv_data(N=10)
        dyn_lti = LinearDynamics(d_lti.A, d_lti.B)
        dyn_ltv = LinearDynamics(d_ltv.A_seq, d_ltv.B_seq)

        @testset "get_A LTI" begin
            @test @inferred(get_A(dyn_lti, 1)) isa Matrix{Float64}
        end
        @testset "get_A LTV" begin
            @test @inferred(get_A(dyn_ltv, 1)) isa Matrix{Float64}
        end
        @testset "get_B LTI" begin
            @test @inferred(get_B(dyn_lti, 1, 1)) isa Matrix{Float64}
        end
        @testset "get_B LTV" begin
            @test @inferred(get_B(dyn_ltv, 1, 1)) isa Matrix{Float64}
        end
        @testset "is_ltv LTI (compile-time)" begin
            @test @inferred(is_ltv(dyn_lti)) === false
        end
        @testset "is_ltv LTV (compile-time)" begin
            @test @inferred(is_ltv(dyn_ltv)) === true
        end

        # Cost accessors
        n, m, N = 4, 2, 8
        cost_lti = LQStageCost(Matrix{Float64}(I(n)), Matrix{Float64}(I(m)))
        cost_ltv = LQStageCost(
            [Matrix{Float64}(I(n)) for _ in 1:N],
            [Matrix{Float64}(I(m)) for _ in 1:N]
        )
        @testset "get_Q LTI" begin
            @test @inferred(get_Q(cost_lti, 1)) isa Matrix{Float64}
        end
        @testset "get_Q LTV" begin
            @test @inferred(get_Q(cost_ltv, 1)) isa Matrix{Float64}
        end
        @testset "get_R LTI" begin
            @test @inferred(get_R(cost_lti, 1)) isa Matrix{Float64}
        end
        @testset "get_R LTV" begin
            @test @inferred(get_R(cost_ltv, 1)) isa Matrix{Float64}
        end
    end

end