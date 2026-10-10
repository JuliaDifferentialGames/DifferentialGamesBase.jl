using Test
using LinearAlgebra
using SparseArrays
using DifferentialGamesBase

# ============================================================================
# Phase 5 test suite
#
# Tests: information-structure wiring (A(a)), InverseGameProblem subtyping
# fix (A(e)), concrete Stochastic/Robust/PO wrappers (A(b)), canonical
# capability vocabulary + generalized dispatch (A(c)), communication
# topology (A(d)).
# ============================================================================

# Minimal stub so InverseGameProblem construction tests don't need a real
# forward solver implementation.
struct ForwardSolverWrapperTestStub <: DifferentialGamesBase.ForwardSolverWrapper end

function _make_lq_player(id, n, m, x0)
    Q   = Matrix{Float64}(I, n, n)
    Qf  = 2.0 * Matrix{Float64}(I, n, n)
    R   = Matrix{Float64}(I, m, m)
    obj = PlayerObjective(id, LQStageCost(Q, R), LQTerminalCost(Qf))
    Player{Float64}(id, n, m, x0, (x, u, p, t) -> x, obj, [])
end

@testset "Phase 5a: Information structure wiring" begin
    @testset "PlayerSpec defaults to PerfectStateInformation" begin
        p = _make_lq_player(1, 2, 1, zeros(2))
        @test p.info isa PerfectStateInformation
    end

    @testset "PlayerSpec accepts explicit info" begin
        Q   = Matrix{Float64}(I, 2, 2)
        R   = Matrix{Float64}(I, 1, 1)
        obj = PlayerObjective(1, LQStageCost(Q, R), LQTerminalCost(Q))
        p = Player{Float64}(1, 2, 1, zeros(2), (x, u, p, t) -> x, obj, [], OpenLoopInformation())
        @test p.info isa OpenLoopInformation
    end

    @testset "DifferentialGame succeeds for deterministic info" begin
        p1 = _make_lq_player(1, 2, 1, zeros(2))
        p2 = _make_lq_player(2, 2, 1, ones(2))
        g = DifferentialGame([p1, p2], 1.0, 0.1)
        @test g isa GameProblem{Float64}
    end

    @testset "DifferentialGame errors on partially-observable info without force_type" begin
        Q   = Matrix{Float64}(I, 2, 2)
        R   = Matrix{Float64}(I, 1, 1)
        obj1 = PlayerObjective(1, LQStageCost(Q, R), LQTerminalCost(Q))
        obj2 = PlayerObjective(2, LQStageCost(Q, R), LQTerminalCost(Q))
        h(x, v, p, t) = x
        po_info = PrivateObservation(h, 2, Matrix{Float64}(I, 2, 2))
        p1 = Player{Float64}(1, 2, 1, zeros(2), (x, u, p, t) -> x, obj1, [], po_info)
        p2 = Player{Float64}(2, 2, 1, ones(2), (x, u, p, t) -> x, obj2, [])
        @test_throws ErrorException DifferentialGame([p1, p2], 1.0, 0.1)
        # force_type bypasses the check
        g = DifferentialGame([p1, p2], 1.0, 0.1; force_type = :deterministic)
        @test g isa GameProblem{Float64}
    end
end

@testset "Phase 5e: InverseGameProblem subtyping" begin
    @test InverseGameProblem <: AbstractInverseGameProblem
    @test AbstractInverseGameProblem{Float64} <: AbstractGameProblem{Float64}

    players = [
        PlayerSpec(1, 2, 1, zeros(2), (x, u, p, t) -> x,
                   PlayerObjective(1, LQStageCost(Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 1, 1)),
                                   LQTerminalCost(Matrix{Float64}(I, 2, 2)))),
        PlayerSpec(2, 2, 1, ones(2), (x, u, p, t) -> x,
                   PlayerObjective(2, LQStageCost(Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 1, 1)),
                                   LQTerminalCost(Matrix{Float64}(I, 2, 2)))),
    ]
    prob = InversePDGNEProblem(
        players,
        [KnownObjective(players[1].objective), UnknownObjective()],
        FullStateObservation(4),
        ForwardSolverWrapperTestStub(),
        1.0, 0.1,
    )
    @test prob isa AbstractGameProblem{Float64}
    @test n_players(prob) == 2
    @test time_horizon(prob) isa DiscreteTime
    @test required_capabilities(prob) == Symbol[]  # generic fallback, not MethodError
end

@testset "Phase 5b: Stochastic / Robust / PO game wrappers" begin
    p1 = _make_lq_player(1, 2, 1, zeros(2))
    p2 = _make_lq_player(2, 2, 1, ones(2))
    g = DifferentialGame([p1, p2], 1.0, 0.1)

    @testset "StochasticGameProblem" begin
        Σ = Matrix{Float64}(I, 4, 4)
        sg = StochasticGameProblem(g, ProcessNoiseModel(Σ))
        @test sg isa AbstractStochasticGame{Float64}
        @test sg isa AbstractGameProblem{Float64}
        @test n_players(sg) == 2
        @test time_horizon(sg) == g.time_horizon
    end

    @testset "RobustGameProblem" begin
        d = DisturbanceSet(-ones(4), ones(4))
        rg = RobustGameProblem(g, d)
        @test rg isa AbstractStochasticGame{Float64}
        @test n_players(rg) == 2
        @test length(d) == 4
        @test_throws AssertionError DisturbanceSet([1.0, 0.0], [0.0, 1.0])
    end

    @testset "POGameProblem" begin
        h(x, v, p, t) = x
        obs = [PrivateObservation(h, 2, Matrix{Float64}(I, 2, 2)) for _ in 1:2]
        pog = POGameProblem(g, obs)
        @test pog isa AbstractPartiallyObservableGame{Float64}
        @test n_players(pog) == 2
        @test_throws AssertionError POGameProblem(g, [obs[1]])  # wrong length
    end
end

@testset "Phase 5c: Canonical capability vocabulary + generalized dispatch" begin
    @testset "validate_capabilities" begin
        @test validate_capabilities([:LQGame, :DiscreteTime]) === nothing
        @test_throws ErrorException validate_capabilities([:NotARealCapability])
    end

    p1 = _make_lq_player(1, 2, 1, zeros(2))
    p2 = _make_lq_player(2, 2, 1, ones(2))
    g = DifferentialGame([p1, p2], 1.0, 0.1)

    @testset "required_capabilities no longer MethodErrors on wrapped types" begin
        ccp = Function[(a, b) -> 0.0 (a, b) -> 0.0; (a, b) -> 0.0 (a, b) -> 0.0]
        lex = LexicographicGameProblem(
            [p1, p2],
            ccp,
            Function[z -> 0.0, z -> 0.0],
            1.0, 0.1,
        )
        @test required_capabilities(lex) == Symbol[]

        cvx = ConvexGameProblem(g)
        @test :ConvexGame in required_capabilities(cvx)

        rhn = RecedingHorizonNashProblem(g, zeros(4), 5)
        @test required_capabilities(rhn) == Symbol[]
    end

    @testset "wrapper types add their own capability tag" begin
        sg = StochasticGameProblem(g, ProcessNoiseModel(Matrix{Float64}(I, 4, 4)))
        @test :Stochastic in required_capabilities(sg)

        rg = RobustGameProblem(g, DisturbanceSet(-ones(4), ones(4)))
        @test :RobustDisturbance in required_capabilities(rg)

        h(x, v, p, t) = x
        pog = POGameProblem(g, [PrivateObservation(h, 2, Matrix{Float64}(I, 2, 2)) for _ in 1:2])
        @test :PartiallyObservable in required_capabilities(pog)
    end

    @testset "all canonical capabilities are self-consistent" begin
        @test validate_capabilities(collect(CANONICAL_CAPABILITIES)) === nothing
    end
end

@testset "Phase 5d: Communication topology" begin
    @test FullCommunication() isa AbstractCommunicationTopology
    @test NoCommunication() isa AbstractCommunicationTopology
    gc = GraphCommunication(sparse([true false; false true]))
    @test gc isa AbstractCommunicationTopology
    dc = DelayedCommunication(FullCommunication(), 3)
    @test dc.delay_steps == 3
    bc = BandwidthLimitedCommunication(NoCommunication(), 128)
    @test bc.max_bits_per_step == 128
    @test_throws AssertionError DelayedCommunication(FullCommunication(), -1)

    @testset "GameMetadata communication field defaults to nothing, non-breaking" begin
        p1 = _make_lq_player(1, 2, 1, zeros(2))
        p2 = _make_lq_player(2, 2, 1, ones(2))
        g = DifferentialGame([p1, p2], 1.0, 0.1)
        @test g.metadata.communication === nothing
        gm = GameMetadata(g.metadata.state_dims, g.metadata.control_dims,
                           g.metadata.state_offsets, g.metadata.control_offsets,
                           g.metadata.coupling_graph, g.metadata.is_potential,
                           g.metadata.potential_function, FullCommunication())
        @test gm.communication isa FullCommunication
    end
end

@testset "Phase 5f: NonlinearStageCost automatic-differentiation fallback" begin
    # Regression test: stage_cost_gradient/hessian previously called
    # automatic_differentiation_gradient/hessian, neither of which was
    # defined anywhere in the package (only the constraint-Jacobian analogue
    # existed) — any NonlinearStageCost without explicit gradient/hessian
    # kwargs would UndefVarError the first time a solver needed derivatives.
    f(x, u, p, t) = 0.5 * (x[1]^2 + 2 * x[1] * x[2] + 3 * u[1]^2)
    cost = NonlinearStageCost(f)
    x = [1.0, 2.0]
    u = [3.0]

    gx, gu = stage_cost_gradient(cost, x, u, nothing, 1)
    @test gx ≈ [x[1] + x[2], x[1]]
    @test gu ≈ [3 * u[1]]

    Hxx, Huu, Hxu = stage_cost_hessian(cost, x, u, nothing, 1)
    @test Hxx ≈ [1.0 1.0; 1.0 0.0]
    @test Huu ≈ [3.0;;]
    @test all(iszero, Hxu)
end
