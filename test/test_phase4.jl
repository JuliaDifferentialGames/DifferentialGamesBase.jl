using Test
using LinearAlgebra
using ForwardDiff
using DifferentialGamesBase

# ============================================================================
# Phase 4: Constraint system tests
#
# Tests: abstract hierarchy, private constraints, shared constraints,
# standard constructors, Jacobian consistency, ForwardDiff compatibility,
# and integration with GameProblem via PDGNEProblem.
# ============================================================================

# ============================================================================
# Abstract hierarchy
# ============================================================================

@testset "Phase 4: Abstract constraint hierarchy" begin

    @testset "Type hierarchy" begin
        @test AbstractPrivateInequality  <: AbstractPrivateConstraint
        @test AbstractPrivateEquality    <: AbstractPrivateConstraint
        @test AbstractSharedInequality   <: AbstractSharedConstraint
        @test AbstractSharedEquality     <: AbstractSharedConstraint
        @test AbstractPrivateConstraint  <: AbstractConstraint
        @test AbstractSharedConstraint   <: AbstractConstraint

        # Branches are disjoint
        @test !(AbstractPrivateConstraint <: AbstractSharedConstraint)
        @test !(AbstractSharedConstraint  <: AbstractPrivateConstraint)
    end

    @testset "Trait queries — is_private / is_shared" begin
        cb  = control_bounds(1; control_offset=0, control_dim=1,
                             lower=[-1.0], upper=[1.0])
        prx = collision_avoidance([1,2]; i_offset=0, j_offset=2, pos_dim=2, d_min=0.5)

        @test  is_private(cb)
        @test !is_shared(cb)
        @test  is_shared(prx)
        @test !is_private(prx)
    end

    @testset "Trait queries — is_inequality / is_equality" begin
        cb  = control_bounds(1; control_offset=0, control_dim=1,
                             lower=[-1.0], upper=[1.0])
        ceq = PrivateEquality(1; func=(x,u,p,t)->[x[1]-1.0], dim=1)
        prx = collision_avoidance([1,2]; i_offset=0, j_offset=2, pos_dim=2, d_min=0.5)
        seq = SharedEquality([1,2]; func=(x,u,p,t)->[sum(x)-1.0], dim=1)

        @test  is_inequality(cb)
        @test !is_equality(cb)
        @test  is_equality(ceq)
        @test !is_inequality(ceq)
        @test  is_inequality(prx)
        @test  is_equality(seq)
    end

    @testset "get_player / get_players" begin
        cb  = control_bounds(2; control_offset=1, control_dim=1,
                             lower=[-1.0], upper=[1.0])
        prx = collision_avoidance([3,1]; i_offset=0, j_offset=4, pos_dim=2, d_min=0.5)

        @test get_player(cb)   == 2
        @test get_players(prx) == [1, 3]   # sorted
    end
end

# ============================================================================
# ControlBounds
# ============================================================================

@testset "Phase 4: ControlBounds" begin

    c  = control_bounds(1; control_offset=0, control_dim=2,
                        lower=[-5.0, -3.0], upper=[5.0, 3.0])
    x  = zeros(4)

    @testset "Satisfied — interior" begin
        u = [2.0, 1.0, 0.0, 0.0]
        v = evaluate_constraint(c, x, u, nothing, 0)
        @test all(v .<= 0)
    end

    @testset "Violated — upper bound" begin
        u = [6.0, 0.0, 0.0, 0.0]
        v = evaluate_constraint(c, x, u, nothing, 0)
        @test any(v .> 0)
    end

    @testset "Boundary — exact bound" begin
        u = [5.0, 3.0, 0.0, 0.0]
        v = evaluate_constraint(c, x, u, nothing, 0)
        @test all(v .<= 0)
        @test any(abs.(v) .< 1e-12)   # exactly at bound
    end

    @testset "Jacobian — analytical exact" begin
        u = [2.0, 1.0, 0.0, 0.0]
        Jx, Ju = constraint_jacobian(c, x, u, nothing, 0)
        @test all(iszero, Jx)
        # Ju[:, 1:2] should be [-I; I]
        @test Ju[1:2, 1:2] ≈ -Matrix{Float64}(I, 2, 2)
        @test Ju[3:4, 1:2] ≈  Matrix{Float64}(I, 2, 2)
        @test all(iszero, Ju[:, 3:end])
    end

    @testset "Jacobian matches ForwardDiff" begin
        u = [1.0, -1.0, 0.5, 0.5]
        Jx_a, Ju_a = constraint_jacobian(c, x, u, nothing, 0)
        z   = vcat(x, u)
        J_fd = ForwardDiff.jacobian(
            z_var -> evaluate_constraint(c, z_var[1:4], z_var[5:end], nothing, 0), z
        )
        @test Jx_a ≈ J_fd[:, 1:4] atol=1e-10
        @test Ju_a ≈ J_fd[:, 5:end] atol=1e-10
    end

    @testset "ForwardDiff compatible" begin
        u = [2.0, 1.0, 0.0, 0.0]
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_constraint(c, z[1:4], z[5:end], nothing, 0),
            vcat(x, u)
        )
    end

    @testset "standard constructor: control_bounds" begin
        c2 = control_bounds(1; control_offset=0, control_dim=2,
                            lower=[-5.0, -3.0], upper=[5.0, 3.0])
        @test c2 isa ControlBounds{Float64}
        @test get_player(c2) == 1
    end
end

# ============================================================================
# StateBounds
# ============================================================================

@testset "Phase 4: StateBounds" begin

    c = state_bounds(2; state_offset=2, state_dim=2,
                     lower=[-10.0, -10.0], upper=[10.0, 10.0])
    u = zeros(4)

    @testset "Satisfied" begin
        x = [0.0, 0.0, 5.0, -3.0]
        @test all(evaluate_constraint(c, x, u, nothing, 0) .<= 0)
    end

    @testset "Violated" begin
        x = [0.0, 0.0, 15.0, 0.0]
        @test any(evaluate_constraint(c, x, u, nothing, 0) .> 0)
    end

    @testset "Jacobian — analytical" begin
        x = [0.0, 0.0, 1.0, 2.0]
        Jx, Ju = constraint_jacobian(c, x, u, nothing, 0)
        @test all(iszero, Ju)
        @test Jx[1:2, 3:4] ≈ -Matrix{Float64}(I, 2, 2)
        @test Jx[3:4, 3:4] ≈  Matrix{Float64}(I, 2, 2)
        @test all(iszero, Jx[:, 1:2])
    end

    @testset "ForwardDiff compatible" begin
        x = [0.0, 0.0, 1.0, 2.0]
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_constraint(c, z[1:4], z[5:end], nothing, 0),
            vcat(x, u)
        )
    end
end

# ============================================================================
# PrivateInequality / PrivateEquality
# ============================================================================

@testset "Phase 4: PrivateNonlinear" begin

    # Player 1 must stay inside unit circle (first 2 state components)
    c_ineq = PrivateInequality(1;
        func = (x, u, p, t) -> [dot(x[1:2], x[1:2]) - 1.0],
        dim  = 1
    )

    @testset "Inequality — inside" begin
        x = [0.3, 0.4, 0.0, 0.0]; u = zeros(2)
        @test evaluate_constraint(c_ineq, x, u, nothing, 0)[1] < 0
    end

    @testset "Inequality — outside" begin
        x = [1.5, 0.0, 0.0, 0.0]; u = zeros(2)
        @test evaluate_constraint(c_ineq, x, u, nothing, 0)[1] > 0
    end

    @testset "ForwardDiff compatible" begin
        x = [0.5, 0.5, 0.0, 0.0]; u = zeros(2)
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_constraint(c_ineq, z[1:4], z[5:end], nothing, 0),
            vcat(x, u)
        )
    end

    @testset "Jacobian matches ForwardDiff" begin
        x = [0.5, 0.5, 0.0, 0.0]; u = zeros(2)
        Jx, Ju = constraint_jacobian(c_ineq, x, u, nothing, 0)
        z = vcat(x, u)
        J_fd = ForwardDiff.jacobian(
            z_var -> evaluate_constraint(c_ineq, z_var[1:4], z_var[5:end], nothing, 0), z
        )
        @test Jx ≈ J_fd[:, 1:4] atol=1e-8
        @test Ju ≈ J_fd[:, 5:end] atol=1e-8
    end

    @testset "Analytical Jacobian override" begin
        c_anal = PrivateInequality(1;
            func     = (x, u, p, t) -> [dot(x[1:2], x[1:2]) - 1.0],
            dim      = 1,
            jacobian = (x, u, p, t) -> (
                reshape([2x[1], 2x[2], 0.0, 0.0], 1, 4),
                zeros(1, 2)
            )
        )
        x = [0.5, 0.5, 0.0, 0.0]; u = zeros(2)
        Jx, Ju = constraint_jacobian(c_anal, x, u, nothing, 0)
        @test Jx ≈ reshape([1.0, 1.0, 0.0, 0.0], 1, 4)
    end

    @testset "Equality — get_player" begin
        c_eq = PrivateEquality(3; func=(x,u,p,t)->[x[1]-x[2]], dim=1)
        @test get_player(c_eq) == 3
        @test is_equality(c_eq)
        @test !is_inequality(c_eq)
    end
end

# ============================================================================
# ProximityConstraint
# ============================================================================

@testset "Phase 4: ProximityConstraint" begin

    c = collision_avoidance([1,2]; i_offset=0, j_offset=2, pos_dim=2, d_min=1.0)

    @testset "Satisfied — separated" begin
        x = [0.0, 0.0, 5.0, 0.0]; u = zeros(2)
        @test evaluate_constraint(c, x, u, nothing, 0)[1] < 0
    end

    @testset "Violated — too close" begin
        x = [0.0, 0.0, 0.3, 0.0]; u = zeros(2)
        @test evaluate_constraint(c, x, u, nothing, 0)[1] > 0
    end

    @testset "Players are sorted" begin
        c2 = collision_avoidance([3,1]; i_offset=0, j_offset=4, pos_dim=2, d_min=0.5)
        @test get_players(c2) == [1, 3]
    end

    @testset "Jacobian — analytical vs ForwardDiff" begin
        x = [0.0, 0.0, 2.0, 0.0]; u = zeros(2)
        Jx_a, Ju_a = constraint_jacobian(c, x, u, nothing, 0)
        z = vcat(x, u)
        J_fd = ForwardDiff.jacobian(
            z_var -> evaluate_constraint(c, z_var[1:4], z_var[5:end], nothing, 0), z
        )
        @test Jx_a ≈ J_fd[:, 1:4] atol=1e-6
        @test Ju_a ≈ J_fd[:, 5:end] atol=1e-6
    end

    @testset "Jacobian near co-location — no NaN (ε regularisation)" begin
        x = [0.0, 0.0, 1e-8, 0.0]; u = zeros(2)
        Jx, Ju = constraint_jacobian(c, x, u, nothing, 0)
        @test !any(isnan, Jx)
        @test !any(isinf, Jx)
    end

    @testset "ForwardDiff compatible" begin
        x = [0.5, 0.0, 1.5, 0.0]; u = zeros(2)
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_constraint(c, z[1:4], z[5:end], nothing, 0),
            vcat(x, u)
        )
    end

    @testset "constraint_violation" begin
        x_ok  = [0.0, 0.0, 2.0, 0.0]; u = zeros(2)
        x_bad = [0.0, 0.0, 0.3, 0.0]
        @test constraint_violation(c, x_ok,  u, nothing, 0) ≈ 0.0 atol=1e-10
        @test constraint_violation(c, x_bad, u, nothing, 0)  > 0.0
    end
end

# ============================================================================
# CommunicationConstraint
# ============================================================================

@testset "Phase 4: CommunicationConstraint" begin

    c = keep_in_range([1,2]; i_offset=0, j_offset=2, pos_dim=2, d_max=5.0)

    @testset "Satisfied — close enough" begin
        x = [0.0, 0.0, 3.0, 0.0]; u = zeros(2)
        @test evaluate_constraint(c, x, u, nothing, 0)[1] < 0
    end

    @testset "Violated — too far" begin
        x = [0.0, 0.0, 10.0, 0.0]; u = zeros(2)
        @test evaluate_constraint(c, x, u, nothing, 0)[1] > 0
    end

    @testset "Jacobian — analytical vs ForwardDiff" begin
        x = [0.0, 0.0, 3.0, 0.0]; u = zeros(2)
        Jx_a, Ju_a = constraint_jacobian(c, x, u, nothing, 0)
        z = vcat(x, u)
        J_fd = ForwardDiff.jacobian(
            z_var -> evaluate_constraint(c, z_var[1:4], z_var[5:end], nothing, 0), z
        )
        @test Jx_a ≈ J_fd[:, 1:4] atol=1e-6
        @test Ju_a ≈ J_fd[:, 5:end] atol=1e-6
    end
end

# ============================================================================
# SharedNonlinear
# ============================================================================

@testset "Phase 4: SharedNonlinear" begin

    # Players 1 and 2 must maintain a relative velocity constraint
    c = SharedInequality([1,2];
        func = (x, u, p, t) -> [(x[2] - x[4])^2 - 4.0],
        dim  = 1
    )

    @testset "Evaluation" begin
        x_ok  = [0.0, 1.0, 0.0, 0.0]; u = zeros(2)  # Δv = 1 < 2
        x_bad = [0.0, 5.0, 0.0, 0.0]                 # Δv = 5 > 2
        @test evaluate_constraint(c, x_ok,  u, nothing, 0)[1] < 0
        @test evaluate_constraint(c, x_bad, u, nothing, 0)[1] > 0
    end

    @testset "Players sorted" begin
        c2 = SharedInequality([3,1]; func=(x,u,p,t)->[0.0], dim=1)
        @test get_players(c2) == [1, 3]
    end

    @testset "Jacobian matches ForwardDiff" begin
        x = [0.0, 1.5, 0.0, 0.0]; u = zeros(2)
        Jx, Ju = constraint_jacobian(c, x, u, nothing, 0)
        z = vcat(x, u)
        J_fd = ForwardDiff.jacobian(
            z_var -> evaluate_constraint(c, z_var[1:4], z_var[5:end], nothing, 0), z
        )
        @test Jx ≈ J_fd[:, 1:4] atol=1e-8
        @test Ju ≈ J_fd[:, 5:end] atol=1e-8
    end

    @testset "SharedEquality" begin
        c_eq = SharedEquality([1,2]; func=(x,u,p,t)->[x[1]-x[3]], dim=1)
        @test is_equality(c_eq)
        @test is_shared(c_eq)
    end
end

# ============================================================================
# LinearCoupling
# ============================================================================

@testset "Phase 4: LinearCoupling" begin

    # u[1] + u[3] ≤ 10 (shared resource constraint on two players)
    A = reshape([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1, 8)
    c = linear_coupling([1,2], A, [10.0])

    @testset "Satisfied" begin
        x = zeros(4); u = [3.0, 0.0, 4.0, 0.0]
        @test evaluate_constraint(c, x, u, nothing, 0)[1] ≈ -3.0
    end

    @testset "Violated" begin
        x = zeros(4); u = [6.0, 0.0, 6.0, 0.0]
        @test evaluate_constraint(c, x, u, nothing, 0)[1] > 0
    end

    @testset "Analytical Jacobian exact" begin
        x = zeros(4); u = [3.0, 0.0, 4.0, 0.0]
        Jx, Ju = constraint_jacobian(c, x, u, nothing, 0)
        @test Jx ≈ A[:, 1:4]
        @test Ju ≈ A[:, 5:end]
    end
end

# ============================================================================
# Integration: PDGNEProblem with new constraint types
# ============================================================================

@testset "Phase 4: Integration with PDGNEProblem" begin

    f1 = (x, u, p, t) -> [x[2]; u[1]]
    f2 = (x, u, p, t) -> [x[2]; u[1]]

    Q  = Matrix{Float64}(I, 2, 2)
    Qf = Matrix{Float64}(I, 2, 2)
    R  = Matrix{Float64}(I, 1, 1)
    o1 = PlayerObjective(1, LQStageCost(Q, R), LQTerminalCost(Qf))
    o2 = PlayerObjective(2, LQStageCost(Q, R), LQTerminalCost(Qf))

    # Private control bounds on each player
    cb1 = control_bounds(1; control_offset=0, control_dim=1,
                         lower=[-2.0], upper=[2.0])
    cb2 = control_bounds(2; control_offset=1, control_dim=1,
                         lower=[-2.0], upper=[2.0])

    # Shared collision avoidance
    prx = collision_avoidance([1,2]; i_offset=0, j_offset=2, pos_dim=1, d_min=0.3)

    p1 = Player{Float64}(1, 2, 1, [1.0, 0.0], f1, o1, [cb1])
    p2 = Player{Float64}(2, 2, 1, [-1.0, 0.0], f2, o2, [cb2])

    game = DifferentialGame([p1, p2], 1.0, 0.1; shared_constraints=[prx])

    @testset "Game constructs without error" begin
        @test game isa GameProblem{Float64}
        @test num_players(game) == 2
        @test is_pd_gnep(game)
        @test_nowarn validate_game_problem(game)
    end

    @testset "Private constraints accessible" begin
        @test length(game.private_constraints) == 2
        pc1 = game.private_constraints[1]
        @test pc1 isa AbstractPrivateConstraint
        @test get_player(pc1) == 1
    end

    @testset "Shared constraints accessible" begin
        @test length(game.shared_constraints) == 1
        sc = game.shared_constraints[1]
        @test sc isa AbstractSharedConstraint
        @test get_players(sc) == [1, 2]
    end

    @testset "Constraint evaluation at initial state" begin
        x0  = game.initial_state
        u0  = zeros(2)
        # Collision: agents start at distance 2 >> d_min=0.3 → satisfied
        @test evaluate_constraint(prx, x0, u0, nothing, 0)[1] < 0
        # Control bounds: zero control → satisfied
        @test all(evaluate_constraint(cb1, x0, u0, nothing, 0) .<= 0)
    end
end

println("\n✓ Phase 4 tests complete.")