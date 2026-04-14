using Test
using LinearAlgebra
using ForwardDiff
using DifferentialGamesBase

# ============================================================================
# Phase 2 test suite
#
# Tests: Player alias, DifferentialGame smart constructor, remake, and all
# standard cost terms against analytical values.
#
# All concrete cost term types must be defined at top level so their
# evaluate_cost_term methods are visible to module-scoped dispatch.
# ============================================================================

# ============================================================================
# Phase 2a: Player{T} alias
# ============================================================================

@testset "Phase 2a: Player alias" begin

    @testset "Player === PlayerSpec" begin
        @test Player{Float64} === PlayerSpec{Float64}
    end

    @testset "Player construction — no constraints" begin
        Q  = Matrix{Float64}(I, 2, 2)
        Qf = Matrix{Float64}(I, 2, 2)
        R  = Matrix{Float64}(I, 1, 1)
        obj = PlayerObjective(1, LQStageCost(Q, R), LQTerminalCost(Qf))
        p   = Player{Float64}(1, 2, 1, zeros(2), (x,u,p,t)->x, obj, [])
        @test p isa PlayerSpec{Float64}
        @test p.id == 1
        @test p.n  == 2
        @test p.m  == 1
    end

    @testset "Player construction — with constraints" begin
        Q   = Matrix{Float64}(I, 2, 2)
        Qf  = Matrix{Float64}(I, 2, 2)
        R   = Matrix{Float64}(I, 1, 1)
        obj = PlayerObjective(2, LQStageCost(Q, R), LQTerminalCost(Qf))
        p   = Player{Float64}(2, 2, 1, ones(2), (x,u,p,t)->-x, obj, [])
        @test p.id == 2
        @test p.x0 == ones(2)
    end
end

# ============================================================================
# Phase 2b: DifferentialGame smart constructor
# ============================================================================

@testset "Phase 2b: DifferentialGame" begin

    function make_lq_player(id, n, m, x0)
        Q   = Matrix{Float64}(I, n, n)
        Qf  = 2.0 * Matrix{Float64}(I, n, n)
        R   = Matrix{Float64}(I, m, m)
        obj = PlayerObjective(id, LQStageCost(Q, R), LQTerminalCost(Qf))
        Player{Float64}(id, n, m, x0, (x,u,p,t)->x, obj, [])
    end

    @testset "Shared LinearDynamics → LQ game" begin
        n = 4; m1 = 2; m2 = 2
        A  = Matrix{Float64}(I, n, n)
        B  = [Matrix{Float64}(I, n, m1)[:, 1:m1],
              Matrix{Float64}(I, n, m2)[:, 1:m2]]
        dyn = LinearDynamics(A, B)

        p1 = make_lq_player(1, n, m1, zeros(n))
        p2 = make_lq_player(2, n, m2, zeros(n))

        game = DifferentialGame(dyn, [p1, p2], 1.0, 0.1)

        @test game isa GameProblem{Float64}
        @test num_players(game) == 2
        @test is_lq_game(game)
        @test is_unconstrained(game)
        @test n_steps(game) == 10
        @test_nowarn validate_game_problem(game)
    end

    @testset "Shared LinearDynamics — time_horizon override" begin
        n = 2
        A   = Matrix{Float64}(I, n, n)
        dyn = LinearDynamics(A, [Matrix{Float64}(I, n, 1)])
        p1  = make_lq_player(1, n, 1, zeros(n))

        th   = DiscreteTime(2.0, 0.1)
        game = DifferentialGame(dyn, [p1], 99.0, 99.0; time_horizon=th)

        @test n_steps(game) == 20   # from th, not tf/dt
    end

    @testset "PD-GNEP (separable dynamics)" begin
        f1 = (x, u, p, t) -> [x[2]; u[1]]
        f2 = (x, u, p, t) -> [x[2]; u[1]]

        Q  = Matrix{Float64}(I, 2, 2)
        Qf = Matrix{Float64}(I, 2, 2)
        R  = Matrix{Float64}(I, 1, 1)

        o1 = PlayerObjective(1, LQStageCost(Q, R), LQTerminalCost(Qf))
        o2 = PlayerObjective(2, LQStageCost(Q, R), LQTerminalCost(Qf))

        p1 = Player{Float64}(1, 2, 1, zeros(2), f1, o1, [])
        p2 = Player{Float64}(2, 2, 1, ones(2),  f2, o2, [])

        game = DifferentialGame([p1, p2], 1.0, 0.1)

        @test game isa GameProblem{Float64}
        @test is_pd_gnep(game)
        @test num_players(game) == 2
        @test state_dim(game)   == 4   # 2 + 2
        @test game.initial_state ≈ [0.0, 0.0, 1.0, 1.0]
        @test_nowarn validate_game_problem(game)
    end

    @testset "Player ID validation" begin
        n = 2
        A   = Matrix{Float64}(I, n, n)
        dyn = LinearDynamics(A, [Matrix{Float64}(I, n, 1)])
        p1  = make_lq_player(1, n, 1, zeros(n))
        p3  = make_lq_player(3, n, 1, zeros(n))   # gap in IDs

        @test_throws AssertionError DifferentialGame(dyn, [p1, p3], 1.0, 0.1)
    end

    @testset "Duplicate player IDs" begin
        n = 2
        A   = Matrix{Float64}(I, n, n)
        dyn = LinearDynamics(A, [Matrix{Float64}(I, n, 1),
                                  Matrix{Float64}(I, n, 1)])
        p1a = make_lq_player(1, n, 1, zeros(n))
        p1b = make_lq_player(1, n, 1, zeros(n))   # duplicate

        @test_throws AssertionError DifferentialGame(dyn, [p1a, p1b], 1.0, 0.1)
    end
end

# ============================================================================
# Phase 2c: remake
# ============================================================================

@testset "Phase 2c: remake" begin

    # Build a minimal game to use throughout
    n = 4; n_pl = 2; T = Float64
    A  = Matrix{T}(I, n, n)
    B  = [Matrix{T}(I, n, 2)[:, 1:2], Matrix{T}(I, n, 2)[:, 1:2]]
    Q  = [Matrix{T}(I, n, n) for _ in 1:n_pl]
    R  = [Matrix{T}(I, 2, 2) for _ in 1:n_pl]
    Qf = [Matrix{T}(I, n, n) for _ in 1:n_pl]
    x0 = zeros(T, n)

    game = LQGameProblem(A, B, Q, R, Qf, x0, T(1.0); dt=T(0.1))

    @testset "remake — initial_state only" begin
        x_new  = randn(n)
        game2  = remake(game; initial_state=x_new)

        @test game2.initial_state ≈ x_new
        @test game2.initial_state !== game.initial_state
        # All other fields shared by reference
        @test game2.dynamics === game.dynamics
        @test game2.objectives === game.objectives
        @test game2.metadata === game.metadata
        @test game2.time_horizon === game.time_horizon
        @test_nowarn validate_game_problem(game2)
    end

    @testset "remake — time_horizon" begin
        th_new = DiscreteTime(2.0, 0.1)
        game2  = remake(game; time_horizon=th_new)

        @test n_steps(game2) == 20
        @test n_steps(game)  == 10    # original unchanged
        @test game2.dynamics === game.dynamics
    end

    @testset "remake — original is unmodified (immutability)" begin
        x_orig = copy(game.initial_state)
        _      = remake(game; initial_state=randn(n))
        @test game.initial_state ≈ x_orig
    end

    @testset "remake — MPC loop pattern (10 steps)" begin
        x = copy(x0)
        g = game
        for _ in 1:10
            x = x .+ 0.01    # simulate trivial dynamics
            g = remake(g; initial_state=x)
        end
        @test g.initial_state ≈ x0 .+ 0.1
        @test g.dynamics === game.dynamics
        @test_nowarn validate_game_problem(g)
    end

    @testset "remake — n_players preserved" begin
        game2 = remake(game; initial_state=zeros(n))
        @test num_players(game2) == num_players(game)
    end
end

# ============================================================================
# Phase 2d: Standard cost terms — analytical values
# ============================================================================

@testset "Phase 2d: QuadraticStateCost" begin

    Q      = [4.0 0.0; 0.0 2.0]
    x_ref  = [1.0, 2.0]
    term   = QuadraticStateCost(Q, x_ref, 0, 2)
    x_full = [1.5, 3.0, 0.0, 0.0]   # player 1 slice is x_full[1:2]
    u_full = zeros(2)

    @testset "Evaluation — analytical" begin
        # ½ ([1.5,3.0] - [1.0,2.0])ᵀ [4 0; 0 2] ([1.5,3.0] - [1.0,2.0])
        # δ = [0.5, 1.0], cost = ½(4·0.25 + 2·1.0) = ½(1 + 2) = 1.5
        @test evaluate_cost_term(term, x_full, u_full, nothing, 0) ≈ 1.5
    end

    @testset "Gradient — analytical" begin
        ∇x, ∇u = cost_term_gradient(term, x_full, u_full, nothing, 0)
        # ∇xᵢ = Q·δ = [4·0.5, 2·1.0] = [2.0, 2.0], embedded in full ∇x
        @test ∇x[1:2] ≈ [2.0, 2.0]
        @test all(iszero, ∇x[3:end])
        @test all(iszero, ∇u)
    end

    @testset "Hessian — exact Q" begin
        Hxx, Huu, Hxu = cost_term_hessian(term, x_full, u_full, nothing, 0)
        @test Hxx[1:2, 1:2] ≈ Q
        @test all(iszero, Hxx[3:end, :])
        @test all(iszero, Huu)
    end

    @testset "Traits" begin
        @test is_quadratic(term)
        @test is_separable_term(term)
    end

    @testset "ForwardDiff compatible" begin
        @test_nowarn ForwardDiff.gradient(
            z -> evaluate_cost_term(term, z[1:4], z[5:6], nothing, 0),
            vcat(x_full, u_full)
        )
    end

    @testset "Offset — player 2 (offset=4)" begin
        term2  = QuadraticStateCost(Q, x_ref, 4, 2)
        x_full2 = [0.0, 0.0, 0.0, 0.0, 1.5, 3.0]
        @test evaluate_cost_term(term2, x_full2, u_full, nothing, 0) ≈ 1.5
        ∇x, _ = cost_term_gradient(term2, x_full2, u_full, nothing, 0)
        @test all(iszero, ∇x[1:4])
        @test ∇x[5:6] ≈ [2.0, 2.0]
    end

    @testset "track_goal constructor" begin
        t2 = track_goal(x_ref, Q; state_offset=0, state_dim=2)
        @test evaluate_cost_term(t2, x_full, u_full, nothing, 0) ≈ 1.5
    end
end

@testset "Phase 2d: QuadraticControlCost" begin

    R      = [3.0 0.0; 0.0 1.0]
    term   = QuadraticControlCost(R, 0, 2)
    x_full = zeros(4)
    u_full = [2.0, 1.0, 0.0, 0.0]

    @testset "Evaluation — analytical" begin
        # ½ [2,1]ᵀ [3 0; 0 1] [2,1] = ½(12 + 1) = 6.5
        @test evaluate_cost_term(term, x_full, u_full, nothing, 0) ≈ 6.5
    end

    @testset "Gradient — analytical" begin
        _, ∇u = cost_term_gradient(term, x_full, u_full, nothing, 0)
        # ∇u₁ = R·u₁ = [6.0, 1.0]
        @test ∇u[1:2] ≈ [6.0, 1.0]
        @test all(iszero, ∇u[3:end])
    end

    @testset "Traits" begin
        @test is_quadratic(term)
        @test is_separable_term(term)
    end

    @testset "Control offset — player 2 (offset=2)" begin
        term2 = QuadraticControlCost(R, 2, 2)
        u2    = [0.0, 0.0, 2.0, 1.0]
        @test evaluate_cost_term(term2, x_full, u2, nothing, 0) ≈ 6.5
        _, ∇u2 = cost_term_gradient(term2, x_full, u2, nothing, 0)
        @test all(iszero, ∇u2[1:2])
        @test ∇u2[3:4] ≈ [6.0, 1.0]
    end

    @testset "regularize_input constructor" begin
        t2 = regularize_input(R; control_offset=0, control_dim=2)
        @test evaluate_cost_term(t2, x_full, u_full, nothing, 0) ≈ 6.5
    end
end

@testset "Phase 2d: ProximityCost" begin

    term = ProximityCost(0, 2, 2, 1.0, 1.0; α=20.0)

    @testset "Far separation → near zero" begin
        # d = 5.0 >> d_min = 1.0 → cost ≈ 0
        x = [0.0, 0.0, 5.0, 0.0]
        @test evaluate_cost_term(term, x, zeros(2), nothing, 0) < 1e-6
    end

    @testset "Co-located → large cost" begin
        x = [0.0, 0.0, 0.0, 0.0]
        cost = evaluate_cost_term(term, x, zeros(2), nothing, 0)
        @test cost > 0.1
    end

    @testset "Monotone in distance" begin
        costs = [evaluate_cost_term(term, [0.0, 0.0, d, 0.0], zeros(2), nothing, 0)
                 for d in [0.0, 0.3, 0.6, 0.9, 1.1, 2.0]]
        @test issorted(costs; rev=true)
    end

    @testset "ForwardDiff compatible" begin
        x = [0.3, 0.0, 0.0, 0.0]
        @test_nowarn ForwardDiff.gradient(
            z -> evaluate_cost_term(term, z, zeros(2), nothing, 0), x
        )
    end

    @testset "Traits" begin
        @test !is_quadratic(term)
        @test !is_separable_term(term)
    end

    @testset "avoid_proximity constructor" begin
        t2 = avoid_proximity(i_offset=0, j_offset=2, pos_dim=2,
                              d_min=1.0, weight=1.0, α=20.0)
        x  = [0.3, 0.0, 0.0, 0.0]
        @test evaluate_cost_term(t2, x, zeros(2), nothing, 0) ≈
              evaluate_cost_term(term, x, zeros(2), nothing, 0)
    end
end

@testset "Phase 2d: CommunicationCost" begin

    term = CommunicationCost(0, 2, 2, 3.0, 1.0; α=20.0)

    @testset "Close together → near zero" begin
        x = [0.0, 0.0, 0.5, 0.0]   # d ≈ 0.5 < d_max=3
        @test evaluate_cost_term(term, x, zeros(2), nothing, 0) < 1e-6
    end

    @testset "Far apart → large cost" begin
        x = [0.0, 0.0, 10.0, 0.0]
        @test evaluate_cost_term(term, x, zeros(2), nothing, 0) > 0.1
    end

    @testset "Monotone increasing in distance beyond d_max" begin
        costs = [evaluate_cost_term(term, [0.0, 0.0, d, 0.0], zeros(2), nothing, 0)
                 for d in [1.0, 2.0, 3.1, 4.0, 6.0, 10.0]]
        @test issorted(costs)
    end

    @testset "ForwardDiff compatible" begin
        @test_nowarn ForwardDiff.gradient(
            z -> evaluate_cost_term(term, z, zeros(2), nothing, 0),
            [0.0, 0.0, 5.0, 0.0]
        )
    end
end

@testset "Phase 2d: ControlBarrierCost" begin

    # Box constraint: x[1] ≤ 2.0 → h(x) = x[1] - 2.0
    term = ControlBarrierCost(x -> x[1] - 2.0; weight=1.0, α=2.0)

    @testset "Deep interior → near zero" begin
        x = [-5.0, 0.0]
        @test evaluate_cost_term(term, x, zeros(1), nothing, 0) < 1e-3
    end

    @testset "At boundary → weight·exp(0) = weight" begin
        x = [2.0, 0.0]
        @test evaluate_cost_term(term, x, zeros(1), nothing, 0) ≈ 1.0 * exp(0.0)
    end

    @testset "Outside boundary → grows" begin
        x_in  = [1.0, 0.0]; x_bd = [2.0, 0.0]; x_out = [3.0, 0.0]
        c_in  = evaluate_cost_term(term, x_in,  zeros(1), nothing, 0)
        c_bd  = evaluate_cost_term(term, x_bd,  zeros(1), nothing, 0)
        c_out = evaluate_cost_term(term, x_out, zeros(1), nothing, 0)
        @test c_in < c_bd < c_out
    end

    @testset "ForwardDiff compatible" begin
        @test_nowarn ForwardDiff.gradient(
            z -> evaluate_cost_term(term, z, zeros(1), nothing, 0),
            [1.5, 0.0]
        )
    end
end

@testset "Phase 2d: QuadraticTerminalCost" begin

    Qf    = [2.0 0.0; 0.0 3.0]
    x_ref = [1.0, 0.0]
    term  = QuadraticTerminalCost(Qf, x_ref, 0, 2)
    x_f   = [2.0, 1.0, 0.0, 0.0]   # terminal joint state

    @testset "Evaluation — analytical" begin
        # δ = [1.0, 1.0], cost = ½(2·1 + 3·1) = 2.5
        @test evaluate_cost_term(term, x_f, nothing) ≈ 2.5
    end

    @testset "Gradient — analytical" begin
        ∇x = cost_term_gradient(term, x_f, nothing)
        @test ∇x[1:2] ≈ Qf * [1.0, 1.0]
        @test all(iszero, ∇x[3:end])
    end

    @testset "Hessian — exact" begin
        Hx = cost_term_hessian(term, x_f, nothing)
        @test Hx[1:2, 1:2] ≈ Qf
        @test all(iszero, Hx[3:end, :])
    end

    @testset "terminal_goal constructor" begin
        t2 = terminal_goal(x_ref, Qf; state_offset=0, state_dim=2)
        @test evaluate_cost_term(t2, x_f, nothing) ≈ 2.5
    end
end

@testset "Phase 2d: Cost term composition (DSL)" begin

    Q  = Matrix{Float64}(I, 2, 2)
    Qf = 2.0 * Matrix{Float64}(I, 2, 2)
    R  = Matrix{Float64}(I, 1, 1)

    stage = track_goal(zeros(2), Q; state_offset=0, state_dim=2) +
            regularize_input(R; control_offset=0, control_dim=1)
    term  = terminal_goal(zeros(2), Qf; state_offset=0, state_dim=2)

    @testset "CompositeCostTerm formed" begin
        @test stage isa CompositeCostTerm
    end

    @testset "Composite evaluation = sum of parts" begin
        x = [1.0, 2.0]; u = [0.5]
        c1 = evaluate_cost_term(track_goal(zeros(2), Q), x, u, nothing, 0)
        c2 = evaluate_cost_term(regularize_input(R), x, u, nothing, 0)
        @test evaluate_cost_term(stage, x, u, nothing, 0) ≈ c1 + c2
    end

    @testset "is_quadratic propagates" begin
        @test is_quadratic(stage)
    end

    @testset "minimize() produces PlayerObjective" begin
        obj = minimize(stage; terminal=term, player_id=1)
        @test obj isa PlayerObjective
        @test obj.player_id == 1
    end
end

@testset "Phase 2d: ProximityTerminalCost" begin

    term = ProximityTerminalCost(0, 2, 2, 1.0, 1.0; α=20.0)

    @testset "Far → near zero" begin
        x = [0.0, 0.0, 5.0, 0.0]
        @test evaluate_cost_term(term, x, nothing) < 1e-6
    end

    @testset "Co-located → large" begin
        x = [0.0, 0.0, 0.0, 0.0]
        @test evaluate_cost_term(term, x, nothing) > 0.1
    end

    @testset "ForwardDiff compatible" begin
        @test_nowarn ForwardDiff.gradient(
            z -> evaluate_cost_term(term, z, nothing),
            [0.3, 0.0, 0.0, 0.0]
        )
    end
end

println("\n✓ Phase 2 tests complete.")