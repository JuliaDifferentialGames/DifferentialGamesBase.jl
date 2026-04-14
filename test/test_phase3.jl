using Test
using LinearAlgebra
using ForwardDiff
using DifferentialGamesBase

# ============================================================================
# Phase 3 test suite: trajectory expansion infrastructure
#
# Testing strategy:
#   All Jacobians and Hessians are verified against ForwardDiff-computed
#   ground truth. For LQ games the expansion must be exact (no regularisation
#   should trigger). For nonlinear games, the expansion is only a local
#   approximation — we test consistency, not absolute values.
#
# Ground truth availability:
#   LinearDynamics + LQStageCost → exact Jacobians and Hessians (analytical).
#   Nonlinear dynamics + costs   → ForwardDiff is ground truth.
# ============================================================================

# ============================================================================
# Shared fixtures
# ============================================================================

function make_double_integrator(; n=4, m1=2, m2=2, N=10, tf=1.0)
    dt  = tf / N
    A   = [1.0 0.0 0.0 0.0;
           0.0 1.0 0.0 0.0;
           dt  0.0 1.0 0.0;
           0.0 dt  0.0 1.0]
    B1  = [dt  0.0; 0.0 dt; 0.0 0.0; 0.0 0.0]
    B2  = [0.0 0.0; 0.0 0.0; dt  0.0; 0.0 dt]
    dyn = LinearDynamics(A, [B1, B2])

    Q  = [Matrix{Float64}(I, n, n) for _ in 1:2]
    R  = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    Qf = [2.0 * Matrix{Float64}(I, n, n) for _ in 1:2]
    x0 = [1.0, 0.0, 0.0, 0.5]

    game = LQGameProblem(A, [B1, B2], Q, R, Qf, x0, tf; dt=dt)
    da   = discretize(dyn, dt; method=ZOHDiscretization())
    times = collect(range(0.0, tf, length=N+1))

    U0 = zeros(m1+m2, N)
    X0 = rollout(dyn, x0, U0, nothing, times)

    return game, dyn, da, X0, U0, times
end

function make_nonlinear_2player(; N=8, tf=1.0)
    # 2 players, separable: double integrator kinematics in 1D each
    # Player i: state [xᵢ, vᵢ], control [aᵢ], dynamics [vᵢ; aᵢ]
    dt = tf / N
    f1 = (x, u, p, t) -> [x[2]; u[1]]
    f2 = (x, u, p, t) -> [x[2]; u[1]]

    dyn = SeparableDynamics([f1, f2], [2, 2], [1, 1])
    x0  = [1.0, 0.0, -1.0, 0.0]

    Q   = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    Qf  = [2.0 * Matrix{Float64}(I, 2, 2) for _ in 1:2]
    R   = [Matrix{Float64}(I, 1, 1) for _ in 1:2]

    o1 = PlayerObjective(1, LQStageCost(Q[1], R[1]), LQTerminalCost(Qf[1]))
    o2 = PlayerObjective(2, LQStageCost(Q[2], R[2]), LQTerminalCost(Qf[2]))
    p1 = Player{Float64}(1, 2, 1, x0[1:2], f1, o1, [])
    p2 = Player{Float64}(2, 2, 1, x0[3:4], f2, o2, [])

    game  = DifferentialGame([p1, p2], tf, dt)
    da    = discretize(dyn, dt; method=ZOHDiscretization())
    times = collect(range(0.0, tf, length=N+1))
    U0    = zeros(2, N)
    X0    = rollout(dyn, x0, U0, nothing, times)

    return game, dyn, da, X0, U0, times
end

# ============================================================================
# 3b: linearize_dynamics
# ============================================================================

@testset "3b: linearize_dynamics — LinearDynamics LTI" begin
    game, dyn, da, X0, U0, times = make_double_integrator()
    N = size(U0, 2)

    exp_dyn = linearize_dynamics(dyn, X0, U0, da, times)

    @test exp_dyn isa DynamicsExpansion{Float64}
    @test exp_dyn.N == N
    @test exp_dyn.n == 4
    @test exp_dyn.m == 4
    @test !exp_dyn.is_separable
    @test isnothing(exp_dyn.A_blocks)

    @testset "A_full[k] == A for LTI" begin
        A = get_A(dyn, 1)
        for k in 1:N
            @test exp_dyn.A_full[k] ≈ A
        end
    end

    @testset "B_full[k] == [B1 B2] for LTI" begin
        B_cat = get_B_concatenated(dyn, 1)
        for k in 1:N
            @test exp_dyn.B_full[k] ≈ B_cat
        end
    end

    @testset "Affine defect c is zero (feasible reference)" begin
        for k in 1:N
            @test norm(exp_dyn.c[k]) < 1e-12
        end
    end

    @testset "Jacobian consistent with jacobian(da,...)" begin
        k = 3
        Ak_da, Bk_da = jacobian(da, X0[:, k], U0[:, k], nothing, k)
        @test exp_dyn.A_full[k] ≈ Ak_da
        @test exp_dyn.B_full[k] ≈ Bk_da
    end
end

@testset "3b: linearize_dynamics — SeparableDynamics block structure" begin
    game, dyn, da, X0, U0, times = make_nonlinear_2player()
    N = size(U0, 2)

    exp_dyn = linearize_dynamics(dyn, X0, U0, da, times)

    @test exp_dyn.is_separable
    @test !isnothing(exp_dyn.A_blocks)
    @test !isnothing(exp_dyn.B_blocks)
    @test length(exp_dyn.A_blocks) == 2   # two players

    @testset "Block structure: A_full is block-diagonal" begin
        for k in 1:N
            A = exp_dyn.A_full[k]
            # Off-diagonal blocks should be zero for separable dynamics
            @test norm(A[1:2, 3:4]) < 1e-12
            @test norm(A[3:4, 1:2]) < 1e-12
        end
    end

    @testset "A_blocks[i][k] matches diagonal block of A_full[k]" begin
        for k in 1:N
            @test exp_dyn.A_blocks[1][k] ≈ exp_dyn.A_full[k][1:2, 1:2]
            @test exp_dyn.A_blocks[2][k] ≈ exp_dyn.A_full[k][3:4, 3:4]
        end
    end

    @testset "B_blocks[i][k] matches block of B_full[k]" begin
        for k in 1:N
            @test exp_dyn.B_blocks[1][k] ≈ exp_dyn.B_full[k][1:2, 1:1]
            @test exp_dyn.B_blocks[2][k] ≈ exp_dyn.B_full[k][3:4, 2:2]
        end
    end

    @testset "ForwardDiff ground truth for block Jacobians" begin
        k  = 2
        dt = times[k+1] - times[k]
        t  = times[k]
        f1 = dyn.player_dynamics[1]
        x1 = X0[1:2, k]; u1 = U0[1:1, k]
        # Ground truth: ForwardDiff through the discrete RK4 step for player 1
        J_fd = ForwardDiff.jacobian(
            z -> begin
                xx = z[1:2]; uu = z[3:3]
                k1 = f1(xx,             uu, nothing, t)
                k2 = f1(xx + dt/2 * k1, uu, nothing, t + dt/2)
                k3 = f1(xx + dt/2 * k2, uu, nothing, t + dt/2)
                k4 = f1(xx + dt    * k3, uu, nothing, t + dt)
                xx + (dt/6) * (k1 + 2k2 + 2k3 + k4)
            end,
            vcat(x1, u1)
        )
        @test exp_dyn.A_blocks[1][k] ≈ J_fd[:, 1:2] atol=1e-10
        @test exp_dyn.B_blocks[1][k] ≈ J_fd[:, 3:3] atol=1e-10
    end

    @testset "Affine defect zero for ZOH rollout" begin
        for k in 1:N
            @test norm(exp_dyn.c[k]) < 1e-6   # small but not machine zero for continuous dyn
        end
    end
end

@testset "3b: linearize_dynamics — CoupledNonlinearDynamics" begin
    f   = (x, u, p, t) -> [x[2]; -sin(x[1]) + u[1]; x[4]; -x[3] + u[2]]
    dyn = CoupledNonlinearDynamics(f, 4, 2)
    x0  = [0.1, 0.0, 0.1, 0.0]
    dt  = 0.05; N = 10; tf = dt * N

    Q  = [Matrix{Float64}(I, 4, 4) for _ in 1:2]
    R  = [Matrix{Float64}(I, 1, 1) for _ in 1:2]
    Qf = [Matrix{Float64}(I, 4, 4) for _ in 1:2]
    A_dummy = Matrix{Float64}(I, 4, 4)
    B_dummy = [reshape([0.0;0.0;1.0;0.0], 4, 1), reshape([0.0;0.0;0.0;1.0], 4, 1)]
    game = LQGameProblem(A_dummy, B_dummy, Q, R, Qf, x0, tf; dt=dt)
    da   = discretize(dyn, dt)
    times = collect(range(0.0, tf, length=N+1))
    U0   = zeros(2, N)
    X0   = rollout(dyn, x0, U0, nothing, times)

    exp_dyn = linearize_dynamics(dyn, X0, U0, da, times)

    @test !exp_dyn.is_separable
    @test isnothing(exp_dyn.A_blocks)

    @testset "A_full[k] matches ForwardDiff Jacobian of discrete map" begin
        k = 3
        Jx_fd, Ju_fd = jacobian(da, X0[:, k], U0[:, k], nothing, times[k])
        @test exp_dyn.A_full[k] ≈ Jx_fd atol=1e-8
        @test exp_dyn.B_full[k] ≈ Ju_fd atol=1e-8
    end
end

# ============================================================================
# 3c: quadraticize_costs
# ============================================================================

@testset "3c: quadraticize_costs — LQStageCost exact" begin
    game, dyn, da, X0, U0, times = make_double_integrator()
    N = size(U0, 2)

    cost_exp = quadraticize_costs(game, X0, U0, times; regularize=false)

    @test cost_exp isa CostExpansion{Float64}
    @test cost_exp.N == N
    @test cost_exp.n == 4
    @test cost_exp.n_players == 2

    @testset "Hxx[i][k] == Q for LTI LQStageCost" begin
        Q_ref = Matrix{Float64}(I, 4, 4)
        for i in 1:2, k in 1:N
            @test cost_exp.Hxx[i][k] ≈ Q_ref
        end
    end

    @testset "Huu[i][k] == R for LTI LQStageCost" begin
        R_ref = Matrix{Float64}(I, 2, 2)
        for i in 1:2, k in 1:N
            @test cost_exp.Huu[i][k] ≈ R_ref
        end
    end

    @testset "Hxu[i][k] == 0 (no cross term in default LQ)" begin
        for i in 1:2, k in 1:N
            @test norm(cost_exp.Hxu[i][k]) < 1e-14
        end
    end

    @testset "Terminal: Hxx_f[i] == Qf" begin
        Qf_ref = 2.0 * Matrix{Float64}(I, 4, 4)
        for i in 1:2
            @test cost_exp.Hxx_f[i] ≈ Qf_ref
        end
    end

    @testset "Gradient at x_ref: gx[i][k] correct at player i's slice" begin
        # For the double integrator (shared state), player 1's state offset is 0
        # and n_xi == n == 4, so gx[1][k] == Q*X0[:,k] == X0[:,k] (Q=I)
        for k in 1:N
            @test cost_exp.gx[1][k] ≈ X0[:, k]
        end
    end
end

@testset "3c: quadraticize_costs — ForwardDiff ground truth" begin
    game, dyn, da, X0, U0, times = make_nonlinear_2player()
    N = size(U0, 2)

    cost_exp = quadraticize_costs(game, X0, U0, times; regularize=false)

    @testset "Hxx[i][k] matches ForwardDiff Hessian (embedded in joint space)" begin
        # Player 1: private state is X0[1:2, k], control is U0[1:1, k]
        i = 1; k = 3
        obj = get_objective(game, i)
        xik = X0[1:2, k]; uik = U0[1:1, k]
        H_fd = ForwardDiff.hessian(
            z -> evaluate_stage_cost(obj.stage_cost, z[1:2], z[3:3], nothing, k),
            vcat(xik, uik)
        )
        # H_fd[1:2,1:2] should match the upper-left block of Hxx[i][k]
        @test cost_exp.Hxx[i][k][1:2, 1:2] ≈ H_fd[1:2, 1:2] atol=1e-8
        # Off-diagonal blocks should be zero (separable dynamics)
        @test norm(cost_exp.Hxx[i][k][1:2, 3:4]) < 1e-12
        @test norm(cost_exp.Hxx[i][k][3:4, :])   < 1e-12
    end

    @testset "gx[i][k] matches ForwardDiff gradient (embedded in joint space)" begin
        # Player 2: private state is X0[3:4, k], control is U0[2:2, k]
        i = 2; k = 5
        obj = get_objective(game, i)
        xik = X0[3:4, k]; uik = U0[2:2, k]
        g_fd = ForwardDiff.gradient(
            z -> evaluate_stage_cost(obj.stage_cost, z[1:2], z[3:3], nothing, k),
            vcat(xik, uik)
        )
        # gx[i][k] embeds ∇xi at player 2's offset (indices 3:4)
        @test cost_exp.gx[i][k][3:4] ≈ g_fd[1:2] atol=1e-8
        @test norm(cost_exp.gx[i][k][1:2]) < 1e-12   # player 1's block is zero
    end
end

@testset "3c: quadraticize_costs — PSD regularisation" begin
    # Build a game with a non-convex NonlinearStageCost: f(x,u) = -½x₁² + ½x₂² + ½u²
    # The Hessian w.r.t. x has eigenvalue -1 in the x₁ direction.
    # With regularize=true, Hxx must be PSD. With regularize=false, it is indefinite.
    n = 2; m = 1; N = 5; dt = 0.1
    f_cost = (x, u, p, t) -> -0.5 * x[1]^2 + 0.5 * x[2]^2 + 0.5 * u[1]^2
    sc  = NonlinearStageCost(f_cost)
    tc  = NonlinearTerminalCost((x, p) -> zero(eltype(x)))
    obj = PlayerObjective(1, sc, tc)

    A_id = Matrix{Float64}(I, n, n)
    B_id = [reshape([0.0; 1.0], n, m)]
    game_nl = LQGameProblem(A_id, B_id,
                            [Matrix{Float64}(I,n,n)],
                            [Matrix{Float64}(I,m,m)],
                            [Matrix{Float64}(I,n,n)],
                            zeros(n), Float64(N*dt); dt=dt)
    # Patch objective to use nonlinear cost
    da_nl = discretize(game_nl.dynamics, dt)
    times_nl = collect(range(0.0, N*dt, length=N+1))
    X_nl  = rollout(game_nl.dynamics, zeros(n), zeros(m, N), nothing, times_nl)

    # Build a minimal 1-player GameProblem with the nonlinear cost by
    # constructing it directly via DifferentialGame
    f_dyn = (x, u, p, t) -> [x[2]; u[1]]
    dyn_nl = CoupledNonlinearDynamics(f_dyn, n, m)
    da_nl2 = discretize(dyn_nl, dt)
    p1_nl  = Player{Float64}(1, n, m, zeros(n), f_dyn, obj, [])
    game_nl2 = DifferentialGame(game_nl.dynamics, [p1_nl], Float64(N*dt), dt)

    exp_reg   = expand(game_nl2, X_nl, zeros(m, N), da_nl; regularize=true)
    exp_noreg = expand(game_nl2, X_nl, zeros(m, N), da_nl; regularize=false)

    @testset "regularize=true → Hxx PSD at all steps" begin
        for k in 1:N
            λ_min = minimum(eigvals(Symmetric(exp_reg.costs.Hxx[1][k])))
            @test λ_min >= -1e-12
        end
    end

    @testset "regularize=false → Hxx may be indefinite" begin
        # At least one step should have a negative eigenvalue from -x₁² term
        any_indefinite = any(
            minimum(eigvals(Symmetric(exp_noreg.costs.Hxx[1][k]))) < -0.5
            for k in 1:N
        )
        @test any_indefinite
    end
end

# ============================================================================
# 3d: expand + assemble_lq_game
# ============================================================================

@testset "3d: expand — integration test" begin
    game, dyn, da, X0, U0, times = make_double_integrator()

    exp = expand(game, X0, U0, da)

    @test exp isa TrajectoryExpansion{Float64}
    @test size(exp.X) == size(X0)
    @test size(exp.U) == size(U0)
    @test exp.dynamics.N == size(U0, 2)
    @test exp.costs.N    == size(U0, 2)
end

@testset "3d: assemble_lq_game — structural validity" begin
    game, dyn, da, X0, U0, times = make_double_integrator()
    N = size(U0, 2)

    exp     = expand(game, X0, U0, da)
    lq_game = assemble_lq_game(exp, game)

    @test lq_game isa GameProblem{Float64}
    @test is_lq_game(lq_game)
    @test is_ltv(lq_game.dynamics)
    @test num_players(lq_game) == 2
    @test n_steps(lq_game) == N
    @test lq_game.initial_state ≈ game.initial_state
    @test_nowarn validate_game_problem(lq_game)
end

@testset "3d: assemble_lq_game — LQ expansion is exact for LQ game" begin
    game, dyn, da, X0, U0, times = make_double_integrator()

    exp     = expand(game, X0, U0, da; regularize=false)
    lq_game = assemble_lq_game(exp, game)

    # For a linear dynamics + LQ cost game, the assembled LQ game
    # should have exactly the same A,B matrices as the original.
    dyn_orig = game.dynamics
    dyn_lq   = lq_game.dynamics
    A_orig = get_A(dyn_orig, 1)

    for k in 1:n_steps(game)
        @test get_A(dyn_lq, k) ≈ A_orig
        @test get_B(dyn_lq, 1, k) ≈ get_B(dyn_orig, 1, k)
        @test get_B(dyn_lq, 2, k) ≈ get_B(dyn_orig, 2, k)
    end

    # Q matrices should match
    obj_orig = get_objective(game, 1)
    obj_lq   = get_objective(lq_game, 1)
    for k in 1:n_steps(game)
        @test get_Q(obj_lq.stage_cost, k) ≈ get_Q(obj_orig.stage_cost, k)
        @test get_R(obj_lq.stage_cost, k) ≈ get_R(obj_orig.stage_cost, k)
    end
end

# ============================================================================
# reference_trajectory
# ============================================================================

@testset "reference_trajectory" begin
    game, dyn, da, X0_ref, U0_ref, times_ref = make_double_integrator()
    N = n_steps(game)

    @testset "Nothing strategy → zero controls" begin
        X, U, times = reference_trajectory(game, nothing, da)
        @test size(X) == (4, N+1)
        @test size(U) == (4, N)
        @test all(iszero, U)
        @test X[:, 1] ≈ game.initial_state
    end

    @testset "OpenLoopStrategy → recovers X0" begin
        times = collect(range(0.0, game.time_horizon.tf, length=N+1))
        strat = OpenLoopStrategy([zeros(2, N), zeros(2, N)], [2, 2], times)
        X, U, _ = reference_trajectory(game, strat, da)
        @test X ≈ X0_ref
        @test all(iszero, U)
    end

    @testset "expand(game, X, U, da) does not throw" begin
        X, U, _ = reference_trajectory(game, nothing, da)
        @test_nowarn expand(game, X, U, da)
    end
end

# ============================================================================
# expand + assemble round-trip consistency
# ============================================================================

@testset "expand/assemble round-trip — LQ game is self-consistent" begin
    # For a linear dynamics + LQ cost game, expanding and reassembling must
    # produce a game with identical structure to the original. This is the
    # key correctness property: the iLQGames inner loop should converge in
    # one step for LQ games because the approximation is exact.
    game, dyn, da, X0, U0, times = make_double_integrator()
    N = n_steps(game)

    exp     = expand(game, X0, U0, da; regularize=false)
    lq_game = assemble_lq_game(exp, game)

    @testset "Assembled game has correct type structure" begin
        @test lq_game isa GameProblem{Float64}
        @test is_lq_game(lq_game)
        @test is_ltv(lq_game.dynamics)
        @test num_players(lq_game) == num_players(game)
        @test n_steps(lq_game) == N
        @test lq_game.initial_state ≈ game.initial_state
        @test_nowarn validate_game_problem(lq_game)
    end

    @testset "Re-expanding assembled game gives same Jacobians" begin
        # Expand the assembled LQ game around the same trajectory.
        # A_full must be identical to the first expansion.
        da_lq  = discretize(lq_game.dynamics, lq_game.time_horizon.dt)
        exp_lq = expand(lq_game, X0, U0, da_lq; regularize=false)
        for k in 1:N
            @test exp_lq.dynamics.A_full[k] ≈ exp.dynamics.A_full[k] atol=1e-10
            @test exp_lq.dynamics.B_full[k] ≈ exp.dynamics.B_full[k] atol=1e-10
        end
    end

    @testset "Re-expanding assembled game gives same cost Hessians" begin
        da_lq  = discretize(lq_game.dynamics, lq_game.time_horizon.dt)
        exp_lq = expand(lq_game, X0, U0, da_lq; regularize=false)
        for i in 1:num_players(game), k in 1:N
            @test exp_lq.costs.Hxx[i][k] ≈ exp.costs.Hxx[i][k] atol=1e-10
            @test exp_lq.costs.Huu[i][k] ≈ exp.costs.Huu[i][k] atol=1e-10
        end
    end

    @testset "Affine defect is zero for feasible reference" begin
        for k in 1:N
            @test norm(exp.dynamics.c[k]) < 1e-12
        end
    end
end

println("\n✓ Phase 3 tests complete.")