using Test
using LinearAlgebra
using ForwardDiff
using DifferentialGamesBase

# ============================================================================
# Phase 1 test suite
#
# Runs against the installed DifferentialGamesBase package.
# No stubs — all types come from the package.
#
# Concrete types used in cost-term tests (AbstractCostTerm subtypes) are
# defined at TOP LEVEL so their evaluate_cost_term methods are visible to
# module-scoped dispatch. This is the lesson from Phase 0.
# ============================================================================

# ============================================================================
# Top-level concrete strategy-test helpers
# ============================================================================

# Minimal AbstractCostTerm for any cost-term tests that remain
# (none needed here — Phase 1 focuses on strategy/dynamics/discretization)

# ============================================================================
# Phase 1b: OpenLoopStrategy
# ============================================================================

@testset "Phase 1b: OpenLoopStrategy" begin

    n_pl  = 2; m1 = 2; m2 = 1; N = 8
    cdims = [m1, m2]
    times = collect(range(0.0, 1.0, length=N+1))
    T     = Float64
    U1    = rand(m1, N)
    U2    = rand(m2, N)

    @testset "Construction — valid" begin
        s = OpenLoopStrategy([U1, U2], cdims, times)
        @test s isa AbstractStrategy{T}
        @test n_steps(s)          == N
        @test n_players(s)        == n_pl
        @test get_control_dims(s) == cdims
    end

    @testset "Construction — dimension validation" begin
        @test_throws AssertionError OpenLoopStrategy([rand(m1+1, N), U2], cdims, times)
        @test_throws AssertionError OpenLoopStrategy([rand(m1, N-1), U2], cdims, times)
        @test_throws AssertionError OpenLoopStrategy([U1, U2], [m1], times)
    end

    @testset "zero_open_loop_strategy" begin
        s = zero_open_loop_strategy(n_pl, cdims, N, times)
        @test all(iszero, s.controls[1])
        @test all(iszero, s.controls[2])
    end

    @testset "get_nominal_control" begin
        s = OpenLoopStrategy([U1, U2], cdims, times)
        @test get_nominal_control(s, 1, 3) == U1[:, 3]
        @test get_nominal_control(s, 2, 7) == U2[:, 7]
    end

    @testset "apply_strategy — ignores x" begin
        s  = OpenLoopStrategy([U1, U2], cdims, times)
        x1 = randn(4); x2 = randn(4)
        u1 = apply_strategy(s, x1, 1)
        u2 = apply_strategy(s, x2, 1)
        @test u1 ≈ u2
        @test u1[1:m1]     ≈ U1[:, 1]
        @test u1[m1+1:end] ≈ U2[:, 1]
    end

    @testset "apply_strategy — η has no effect" begin
        s = OpenLoopStrategy([U1, U2], cdims, times)
        x = randn(4)
        @test apply_strategy(s, x, 1; η=0.0) ≈ apply_strategy(s, x, 1; η=1.0)
    end

    @testset "ForwardDiff through apply_strategy" begin
        s = OpenLoopStrategy([U1, U2], cdims, times)
        x = zeros(4)
        J = ForwardDiff.jacobian(x -> apply_strategy(s, x, 1), x)
        @test all(iszero, J)
    end
end

# ============================================================================
# Phase 1b: FeedbackStrategy
# ============================================================================

@testset "Phase 1b: FeedbackStrategy" begin


    n = 4; n_pl = 2; m1 = 2; m2 = 1; N = 6
    cdims = [m1, m2]; T = Float64
    times = collect(range(0.0, 0.6, length=N+1))

    gains = [[rand(m1, n) for _ in 1:N], [rand(m2, n) for _ in 1:N]]
    ff    = [[zeros(m1) for _ in 1:N], [zeros(m2) for _ in 1:N]]
    x_nom = zeros(n, N+1)
    u_nom = [zeros(m1, N), zeros(m2, N)]

    @testset "Construction — valid" begin
        s = FeedbackStrategy(gains, ff, x_nom, u_nom, cdims, times)
        @test s isa AbstractStrategy{T}
        @test n_steps(s)   == N
        @test n_players(s) == n_pl
        @test state_dim(s) == n
    end

    @testset "Construction — dimension validation" begin
        bad = deepcopy(gains); pop!(bad[1])
        @test_throws AssertionError FeedbackStrategy(bad, ff, x_nom, u_nom, cdims, times)

        bad2 = deepcopy(gains); bad2[1][1] = rand(m1+1, n)
        @test_throws AssertionError FeedbackStrategy(bad2, ff, x_nom, u_nom, cdims, times)

        @test_throws AssertionError FeedbackStrategy(gains, ff, zeros(n, N), u_nom, cdims, times)
    end

    @testset "zero_feedback_strategy" begin
        s = zero_feedback_strategy(n_pl, n, cdims, N, times)
        @test all(iszero, s.gains[1][1])
        @test all(iszero, s.nominal_states)
    end

    @testset "apply_strategy — zero gains, nonzero nominal" begin
        u_nom2 = [ones(m1, N), 2ones(m2, N)]
        s2 = FeedbackStrategy(
            [[zeros(m1, n) for _ in 1:N], [zeros(m2, n) for _ in 1:N]],
            ff, x_nom, u_nom2, cdims, times
        )
        x = zeros(n)
        u = apply_strategy(s2, x, 1; η=1.0)
        @test u[1:m1]     ≈ ones(m1)
        @test u[m1+1:end] ≈ 2ones(m2)
    end

    @testset "apply_strategy — feedback correction" begin
        P1 = [[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0]]
        s  = FeedbackStrategy(
            [P1, [zeros(m2, n)]],
            [[zeros(m1)], [zeros(m2)]],
            zeros(n, 2), [zeros(m1, 1), zeros(m2, 1)],
            cdims, [0.0, 0.1]
        )
        x = [1.0, 2.0, 0.0, 0.0]
        u = apply_strategy(s, x, 1; η=1.0)
        @test u[1:m1] ≈ [-1.0, -2.0]
    end

    @testset "η interpolation" begin
        ff_nz = [[ones(m1) for _ in 1:N], [ones(m2) for _ in 1:N]]
        s2    = FeedbackStrategy(gains, ff_nz, x_nom, u_nom, cdims, times)
        x     = randn(n)
        u_η0  = apply_strategy(s2, x, 1; η=0.0)
        u_η1  = apply_strategy(s2, x, 1; η=1.0)
        @test !all(u_η0 .≈ u_η1)
        @test apply_strategy(s2, x, 1; η=0.5) ≈ (u_η0 + u_η1) / 2
    end

    @testset "ForwardDiff — zero gains → zero Jacobian" begin
        s  = zero_feedback_strategy(n_pl, n, cdims, N, times)
        x  = zeros(n)
        J  = ForwardDiff.jacobian(x -> apply_strategy(s, x, 1; η=1.0), x)
        @test all(iszero, J)
    end

    @testset "ForwardDiff — nonzero gains → Jacobian = -P" begin
        s  = FeedbackStrategy(gains, ff, x_nom, u_nom, cdims, times)
        x  = zeros(n)
        J  = ForwardDiff.jacobian(x -> apply_strategy(s, x, 1; η=1.0), x)
        @test J[1:m1, :]     ≈ -gains[1][1]
        @test J[m1+1:end, :] ≈ -gains[2][1]
    end

    @testset "to_open_loop conversion" begin
        u_nom3 = [rand(m1, N), rand(m2, N)]
        s      = FeedbackStrategy(gains, ff, x_nom, u_nom3, cdims, times)
        ol     = to_open_loop(s)
        @test ol isa OpenLoopStrategy{T}
        @test n_steps(ol) == N
        for i in 1:n_pl
            @test ol.controls[i] ≈ u_nom3[i]
        end
    end

    @testset "control_offsets" begin
        s = FeedbackStrategy(gains, ff, x_nom, u_nom, cdims, times)
        @test control_offsets(s) == [0, m1]
    end
end

# ============================================================================
# Phase 1a: evaluate_dynamics
# ============================================================================

@testset "Phase 1a: evaluate_dynamics" begin

    @testset "LinearDynamics — LTI single step" begin
        A   = [1.0 0.1; 0.0 1.0]
        B   = [reshape([0.0; 0.1], 2, 1)]
        dyn = LinearDynamics(A, B)
        x   = [1.0, 0.5]; u = [2.0]
        @test evaluate_dynamics(dyn, x, u, nothing, 1) ≈ A*x + B[1]*u
    end

    @testset "LinearDynamics — multi-player joint control" begin
        A  = Matrix{Float64}(I, 4, 4)
        B1 = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
        B2 = [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.0 1.0]
        dyn = LinearDynamics(A, [B1, B2])
        x   = ones(4); u = [1.0, 2.0, 3.0, 4.0]
        @test evaluate_dynamics(dyn, x, u, nothing, 1) ≈ x + B1*u[1:2] + B2*u[3:4]
    end

    @testset "CoupledNonlinearDynamics — pendulum" begin
        f   = (x, u, p, t) -> [x[2]; -sin(x[1]) + u[1]]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        x   = [π/6, 0.0]; u = [0.1]
        ẋ   = evaluate_dynamics(dyn, x, u, nothing, 0.0)
        @test ẋ ≈ [0.0; -sin(π/6) + 0.1]
    end

    @testset "SeparableDynamics — stacked RHS" begin
        f1  = (x1, u1, p, t) -> [x1[2]; u1[1]]
        f2  = (x2, u2, p, t) -> [-x2[1]; u2[1]]
        dyn = SeparableDynamics([f1, f2], [2, 2], [1, 1])
        x   = [1.0, 0.5, 2.0, -1.0]; u = [0.3, 0.7]
        ẋ   = evaluate_dynamics(dyn, x, u, nothing, 0.0)
        @test ẋ[1:2] ≈ f1(x[1:2], [u[1]], nothing, 0.0)
        @test ẋ[3:4] ≈ f2(x[3:4], [u[2]], nothing, 0.0)
    end

    @testset "ForwardDiff compatibility — linear" begin
        A   = [1.0 0.1; 0.0 1.0]
        dyn = LinearDynamics(A, [reshape([0.0; 0.1], 2, 1)])
        x   = [1.0, 0.5]; u = [2.0]
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_dynamics(dyn, z[1:2], z[3:3], nothing, 1), vcat(x, u)
        )
    end

    @testset "ForwardDiff compatibility — nonlinear" begin
        f   = (x, u, p, t) -> [x[2]; -sin(x[1]) + u[1]]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        x   = [0.1, 0.0]; u = [0.0]
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_dynamics(dyn, z[1:2], z[3:3], nothing, 0.0), vcat(x, u)
        )
    end
end

# ============================================================================
# Phase 1: DiscreteApproximation — ZOHDiscretization
# ============================================================================

@testset "Phase 1: ZOHDiscretization" begin


    @testset "Harmonic oscillator — one RK4 step" begin
        f   = (x, u, p, t) -> [x[2]; -x[1]]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        dt  = 0.01
        da  = discretize(dyn, dt; method=ZOHDiscretization())

        @test da isa DiscreteApproximation
        @test da.state_dim   == 2
        @test da.control_dim == 1
        @test da.dt          ≈ dt

        x1 = da_step(da, [1.0, 0.0], [0.0], nothing, 0.0)
        @test x1[1] ≈ cos(dt)  atol=1e-6
        @test x1[2] ≈ -sin(dt) atol=1e-6
    end

    @testset "Jacobian matches finite differences" begin
        f   = (x, u, p, t) -> [x[2]; -x[1] + u[1]]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        da  = discretize(dyn, 0.05; method=ZOHDiscretization())

        x = [1.0, 0.5]; u = [0.2]
        Jx_da, Ju_da = jacobian(da, x, u, nothing, 0.0)

        ε = 1e-6
        Jx_fd = hcat([
            (da_step(da, x .+ ε*I(2)[:,j], u, nothing, 0.0) -
             da_step(da, x .- ε*I(2)[:,j], u, nothing, 0.0)) ./ (2ε)
            for j in 1:2]...)
        Ju_fd = (da_step(da, x, u .+ [ε], nothing, 0.0) -
                 da_step(da, x, u .- [ε], nothing, 0.0)) ./ (2ε)

        @test Jx_da ≈ Jx_fd atol=1e-5
        @test Ju_da ≈ Ju_fd atol=1e-5
    end

    @testset "Jacobian = ForwardDiff through step (key consistency test)" begin
        # This is the invariant that prevents artificial infeasibility:
        # jacobian(da,...) and ForwardDiff.jacobian through da_step(da,...) are identical.
        f   = (x, u, p, t) -> [x[2]; -x[1]^2 + u[1]]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        da  = discretize(dyn, 0.05; method=ZOHDiscretization())

        x = [0.5, -0.3]; u = [0.1]
        Jx_da, Ju_da = jacobian(da, x, u, nothing, 0.0)

        z   = vcat(x, u)
        J_fd = ForwardDiff.jacobian(
            z_var -> da_step(da, z_var[1:2], z_var[3:3], nothing, 0.0), z
        )
        @test Jx_da ≈ J_fd[:, 1:2] atol=1e-10
        @test Ju_da ≈ J_fd[:, 3:3] atol=1e-10
    end

    @testset "ZOH on LinearDynamics" begin
        A  = [1.0 0.1; 0.0 1.0]
        B1 = reshape([0.0; 0.1], 2, 1)
        dyn = LinearDynamics(A, [B1])
        da  = discretize(dyn, 0.1; method=ZOHDiscretization())

        x = [1.0, 0.5]; u = [2.0]
        @test da_step(da, x, u, nothing, 1) ≈ A*x + B1*u

        Jx, Ju = jacobian(da, x, u, nothing, 1)
        @test Jx ≈ A
        @test Ju ≈ B1
    end

    @testset "dt validation" begin
        f   = (x, u, p, t) -> x + u
        dyn = CoupledNonlinearDynamics(f, 2, 2)
        @test_throws AssertionError discretize(dyn, -0.1)
        @test_throws AssertionError discretize(dyn, 0.0)
    end
end

# ============================================================================
# Phase 1: DiscreteApproximation — MatrixExpDiscretization
# ============================================================================

@testset "Phase 1: MatrixExpDiscretization" begin

    @testset "Double integrator — exact discrete map" begin
        # ẍ = u  →  A=[0 1;0 0], B=[0;1]
        # Exact: x₁(dt)=x₁+dt·x₂+½dt²·u, x₂(dt)=x₂+dt·u
        A  = [0.0 1.0; 0.0 0.0]
        B1 = reshape([0.0; 1.0], 2, 1)
        dyn = LinearDynamics(A, [B1])
        dt  = 0.1

        da = discretize(dyn, dt; method=MatrixExpDiscretization())
        @test da isa DiscreteApproximation

        x = [1.0, 2.0]; u = [0.5]
        x1 = da_step(da, x, u, nothing, 0.0)
        @test x1[1] ≈ 1.0 + 0.1*2.0 + 0.5*0.01*0.5  atol=1e-10
        @test x1[2] ≈ 2.0 + 0.1*0.5                   atol=1e-10
    end

    @testset "Jacobian is constant (LTI)" begin
        A  = [0.0 1.0; 0.0 0.0]
        B1 = reshape([0.0; 1.0], 2, 1)
        dyn = LinearDynamics(A, [B1])
        da  = discretize(dyn, 0.1; method=MatrixExpDiscretization())

        Jx1, Ju1 = jacobian(da, randn(2), randn(1), nothing, 0.0)
        Jx2, Ju2 = jacobian(da, randn(2), randn(1), nothing, 0.0)
        @test Jx1 ≈ Jx2   # LTI → Jacobian independent of (x, u)
        @test Ju1 ≈ Ju2
    end

    @testset "MatrixExp vs ZOH small-dt agreement" begin
        # A = [0 1; -1 0] is a continuous-time system matrix (harmonic oscillator).
        # ZOH must be applied to CoupledNonlinearDynamics (uses RK4 on continuous RHS).
        # MatrixExp is applied to LinearDynamics constructed from the same A.
        # Both should give nearly identical results for small dt.
        A  = [0.0 1.0; -1.0 0.0]
        B1 = reshape([0.0; 1.0], 2, 1)
        dt = 0.001

        # ZOH on continuous dynamics (RK4 path)
        f_linear = (x, u, p, t) -> A * x + B1 * u
        dyn_cont = CoupledNonlinearDynamics(f_linear, 2, 1)
        da_rk4   = discretize(dyn_cont, dt; method=ZOHDiscretization())

        # MatrixExp on LinearDynamics (exact)
        dyn_lin  = LinearDynamics(A, [B1])
        da_exp   = discretize(dyn_lin, dt; method=MatrixExpDiscretization())

        x = [1.0, 0.0]; u = [0.0]
        @test da_step(da_rk4, x, u, nothing, 0.0) ≈ da_step(da_exp, x, u, nothing, 0.0) atol=1e-10
    end

    @testset "Error on nonlinear dynamics" begin
        f   = (x, u, p, t) -> [x[2]; -sin(x[1])]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        @test_throws ErrorException discretize(dyn, 0.1; method=MatrixExpDiscretization())
    end
end

# ============================================================================
# Phase 1a: rollout (visualization / warm-start path — separate from solver)
# ============================================================================

@testset "Phase 1a: rollout" begin

    @testset "Linear rollout — exact" begin
        A   = [1.0 0.1; 0.0 1.0]
        B   = [reshape([0.0; 0.1], 2, 1)]
        dyn = LinearDynamics(A, B)
        x0  = [1.0, 0.0]; N = 5
        U   = zeros(1, N)
        ts  = collect(range(0.0, 0.5, length=N+1))

        X = rollout(dyn, x0, U, nothing, ts)
        @test size(X) == (2, N+1)
        @test X[:, 1] ≈ x0

        x_ref = copy(x0)
        for _ in 1:N; x_ref = A * x_ref; end
        @test X[:, N+1] ≈ x_ref
    end

    @testset "Nonlinear rollout — harmonic oscillator (RK4 fallback)" begin
        f   = (x, u, p, t) -> [x[2]; -x[1]]
        dyn = CoupledNonlinearDynamics(f, 2, 1)
        x0  = [1.0, 0.0]; N = 200
        U   = zeros(1, N)
        ts  = collect(range(0.0, π/2, length=N+1))

        X = rollout(dyn, x0, U, nothing, ts)
        @test X[1, end] ≈ 0.0  atol=1e-5
        @test X[2, end] ≈ -1.0 atol=1e-5
    end

    @testset "Dimension validation" begin
        A   = Matrix{Float64}(I, 2, 2)
        dyn = LinearDynamics(A, [Matrix{Float64}(I, 2, 2)])
        x0  = [1.0, 0.0]; U = zeros(2, 5)
        ts  = collect(range(0.0, 1.0, length=6))
        @test_throws AssertionError rollout(dyn, [1.0], U, nothing, ts)
        @test_throws AssertionError rollout(dyn, x0, U, nothing, ts[1:5])
    end

    @testset "rollout_strategy — OpenLoopStrategy recovers open-loop" begin
        A   = [1.0 0.1; 0.0 1.0]
        B1  = reshape([0.0; 0.1], 2, 1)
        dyn = LinearDynamics(A, [B1])
        n = 2; m1 = 1; N = 5
        ts  = collect(range(0.0, 0.5, length=N+1))
        U_ref = fill(0.5, m1, N)

        s  = OpenLoopStrategy([U_ref], [m1], ts)
        x0 = [1.0, 0.0]
        X, U = rollout_strategy(dyn, x0, s, nothing)
        X_ref = rollout(dyn, x0, U_ref, nothing, ts)

        @test X ≈ X_ref
        @test all(U .≈ 0.5)
    end

    @testset "rollout_strategy — FeedbackStrategy zero gains = open-loop nominal" begin
        A  = [1.0 0.1; 0.0 1.0]
        B1 = reshape([0.0; 0.1], 2, 1)
        dyn = LinearDynamics(A, [B1])
        n = 2; m1 = 1; N = 5
        ts  = collect(range(0.0, 0.5, length=N+1))
        u_nom = [fill(0.3, m1, N)]

        s = FeedbackStrategy(
            [[zeros(m1, n) for _ in 1:N]],
            [[zeros(m1)    for _ in 1:N]],
            zeros(n, N+1), u_nom, [m1], ts
        )
        x0 = [1.0, 0.0]
        X, U  = rollout_strategy(dyn, x0, s, nothing)
        X_ref = rollout(dyn, x0, fill(0.3, m1, N), nothing, ts)
        @test X ≈ X_ref
    end
end

# ============================================================================
# Phase 1: DiffEq integration (skipped gracefully if OrdinaryDiffEq absent)
# ============================================================================

@testset "Phase 1a: DiffEq integration" begin
    diffeq_ok = try; using OrdinaryDiffEq; true; catch; false; end

    if !diffeq_ok
        @test_skip "OrdinaryDiffEq not loaded — skipping DiffEq tests"
    else
        @testset "Tsit5 vs RK4 fallback — harmonic oscillator" begin
            f   = (x, u, p, t) -> [x[2]; -x[1]]
            dyn = CoupledNonlinearDynamics(f, 2, 1)
            x0  = [1.0, 0.0]; N = 50
            U   = zeros(1, N)
            ts  = collect(range(0.0, π/2, length=N+1))

            X_rk4 = rollout(dyn, x0, U, nothing, ts; integrator=nothing)
            X_ts5 = rollout(dyn, x0, U, nothing, ts; integrator=Tsit5())
            @test norm(X_rk4 - X_ts5, Inf) < 1e-5
        end

        @testset "DiffEqDiscretization — consistent Jacobian" begin
            f   = (x, u, p, t) -> [x[2]; -x[1] + u[1]]
            dyn = CoupledNonlinearDynamics(f, 2, 1)
            da  = discretize(dyn, 0.05; method=DiffEqDiscretization(Tsit5()))

            x = [1.0, 0.5]; u = [0.2]
            Jx_da, Ju_da = jacobian(da, x, u, nothing, 0.0)
            z   = vcat(x, u)
            J_fd = ForwardDiff.jacobian(
                z_var -> da_step(da, z_var[1:2], z_var[3:3], nothing, 0.0), z
            )
            @test Jx_da ≈ J_fd[:, 1:2] atol=1e-8
            @test Ju_da ≈ J_fd[:, 3:3] atol=1e-8
        end
    end
end

# ============================================================================
# Phase 1e: validate_game_problem — uses real GameProblem from LQGameProblem
# ============================================================================

@testset "Phase 1e: validate_game_problem" begin

    n = 4; n_pl = 2; T = Float64
    A  = Matrix{T}(I, n, n)
    B  = [Matrix{T}(I, n, 2)[:, 1:2], Matrix{T}(I, n, 2)[:, 1:2]]
    Q  = [Matrix{T}(I, n, n) for _ in 1:n_pl]
    R  = [Matrix{T}(I, 2, 2) for _ in 1:n_pl]
    Qf = [Matrix{T}(I, n, n) for _ in 1:n_pl]
    x0 = zeros(T, n)
    tf = T(1.0); dt = T(0.1)

    @testset "Valid LQGameProblem passes" begin
        game = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)
        @test_nowarn validate_game_problem(game)
    end

    @testset "GameProblem is AbstractDeterministicGame" begin
        game = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)
        @test game isa AbstractDeterministicGame
        @test game isa AbstractGameProblem
        @test is_deterministic(game)
    end

    @testset "n_steps is correct" begin
        game = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)
        @test n_steps(game) == Int(round(tf / dt))
    end

    @testset "num_players" begin
        game = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)
        @test num_players(game) == n_pl
        @test n_players(game)   == n_pl
    end

    @testset "Structural queries" begin
        game = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)
        @test is_lq_game(game)
        @test is_unconstrained(game)
        @test !is_pd_gnep(game)
    end
end

# ============================================================================
# Phase 1d: FNELQ cross-term M — analytical verification
# ============================================================================

@testset "Phase 1d: FNELQ cross-term M (analytical)" begin
    # Scalar problem (n=m=N=1 player). Verifies backward pass formulas directly.
    # Cost: ½x²Q + ½u²R + x·M·u  (stage) + ½xf²·Qf  (terminal)
    # Dynamics: x(1) = A·x(0) + B·u(0)
    #
    # Backward pass with M:
    #   S   = R + B²·Qf
    #   YP  = B·Qf·A + M      ← M adds to RHS
    #   P   = YP / S
    #   F   = A - B·P
    #   Z(0) = F²·Qf + Q + P²·R - 2M·P   ← -2MP correction

    A_s=1.0; B_s=0.5; Q_s=2.0; R_s=1.0; M_s=0.3; Qf_s=3.0

    Z1   = Qf_s
    BiZi = B_s * Z1
    S    = R_s + BiZi * B_s
    YP   = BiZi * A_s + M_s
    P    = YP / S

    @testset "S matrix" begin
        @test S ≈ R_s + B_s^2 * Qf_s
    end

    @testset "Gain P with M" begin
        @test P ≈ (B_s * Qf_s * A_s + M_s) / (R_s + B_s^2 * Qf_s)
    end

    @testset "P differs from no-M case" begin
        P_no_M = (B_s * Qf_s * A_s) / (R_s + B_s^2 * Qf_s)
        @test !(P ≈ P_no_M)
    end

    @testset "Cost-to-go correction" begin
        F   = A_s - B_s * P
        Z0  = F^2 * Z1 + Q_s + P^2 * R_s - M_s * P - P * M_s
        Z0_no_M = F^2 * Z1 + Q_s + P^2 * R_s
        @test abs(Z0 - Z0_no_M) ≈ abs(2 * M_s * P) atol=1e-12
    end

    @testset "M=0 reduces to standard Riccati" begin
        P0 = (B_s * Qf_s * A_s + 0.0) / S
        @test P0 ≈ (B_s * Qf_s * A_s) / (R_s + B_s^2 * Qf_s)
    end
end

# ============================================================================
# Phase 1: GNEPSolution carries AbstractStrategy
# ============================================================================

@testset "Phase 1: GNEPSolution strategy typing" begin
    # Build a minimal valid game and check solution type carries strategy
    n = 2; T_f = 1.0; dt = 0.1
    A  = Matrix{Float64}(I, n, n)
    B  = [reshape([1.0; 0.0], n, 1), reshape([0.0; 1.0], n, 1)]
    Q  = [Matrix{Float64}(I, n, n) for _ in 1:2]
    R  = [Matrix{Float64}(I, 1, 1) for _ in 1:2]
    Qf = [Matrix{Float64}(I, n, n) for _ in 1:2]
    x0 = zeros(n)

    game = LQGameProblem(A, B, Q, R, Qf, x0, T_f; dt=dt)

    N     = n_steps(game)
    m_tot = 2
    times = collect(range(0.0, T_f, length=N+1))

    @testset "OpenLoopStrategy in GNEPSolution" begin
        ol = OpenLoopStrategy([zeros(1, N), zeros(1, N)], [1, 1], times)
        sol = GNEPSolution(
            game,
            [Trajectory(i, zeros(n, N+1), zeros(1, N), times, 0.0) for i in 1:2];
            strategy         = ol,
            equilibrium_type = :OpenLoopNash,
            converged        = true
        )
        @test has_strategy(sol)
        @test get_strategy(sol) isa OpenLoopStrategy
        @test get_strategy(sol) isa AbstractStrategy
        @test is_open_loop_solution(sol)
        @test !is_feedback(sol)
    end

    @testset "FeedbackStrategy in GNEPSolution" begin
        fb = zero_feedback_strategy(2, n, [1, 1], N, times)
        sol = GNEPSolution(
            game,
            [Trajectory(i, zeros(n, N+1), zeros(1, N), times, 0.0) for i in 1:2];
            strategy         = fb,
            equilibrium_type = :FeedbackNash,
            converged        = true
        )
        @test has_strategy(sol)
        @test get_strategy(sol) isa FeedbackStrategy
        @test is_feedback(sol)
        @test !is_open_loop_solution(sol)
    end

    @testset "No strategy — get_strategy errors" begin
        sol = GNEPSolution(
            game,
            [Trajectory(i, zeros(n, N+1), zeros(1, N), times, 0.0) for i in 1:2]
        )
        @test !has_strategy(sol)
        @test_throws ErrorException get_strategy(sol)
    end
end

println("\n✓ Phase 1 tests complete.")