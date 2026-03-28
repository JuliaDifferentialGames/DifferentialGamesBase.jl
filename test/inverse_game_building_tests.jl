using Test
using LinearAlgebra
using SparseArrays
using DifferentialGamesBase

# ============================================================================
# Test Utilities and Stubs
# ============================================================================

# Minimal concrete ForwardSolverWrapper for testing the interface.
# Returns a trivially predictable next state (zero-dynamics) so tests
# are deterministic without requiring a real Nash solver.
struct ZeroDynamicsWrapper <: ForwardSolverWrapper
    dt::Float64
end

# Predict x_{t+1} = x_t (stationary, trivially deterministic)
function DifferentialGamesBase.predict_next_state(
    wrapper::ZeroDynamicsWrapper,
    prob::GameProblem{T},
    x0::AbstractVector{T}
) where {T}
    return copy(x0)
end

# solve_forward not needed for problem construction tests; stub it to error
# clearly if accidentally called.
function DifferentialGamesBase.solve_forward(
    ::ZeroDynamicsWrapper, ::GameProblem, ::AbstractVector
)
    error("ZeroDynamicsWrapper.solve_forward should not be called in construction tests")
end

# Helper: build a minimal valid PlayerSpec for player i with state dim n,
# control dim m, and LQ costs.
function make_player(i::Int, n::Int, m::Int, x0::Vector{Float64})
    dynamics = (xi, ui, p, t) -> [xi[min(n÷2+1, n):n]; ui[1:min(m, n-n÷2)];
                                   zeros(max(0, n - n÷2 - m))]
    stage    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
    terminal = DiagonalLQTerminalCost(10.0 * ones(n))
    obj      = PlayerObjective(i, stage, terminal)
    return PlayerSpec(i, n, m, x0, dynamics, obj)
end

# Helper: a simple nonlinear range+LOS measurement h: R^n -> R^4
# (mimics the CWH observation model from Balo et al.)
function make_cwh_obs(n_total::Int)
    function h(x)
        δx, δy, δz = x[1], x[2], x[3]
        R = sqrt(δx^2 + δy^2 + δz^2) + 1e-8   # avoid divide-by-zero
        return [R; δx/R; δy/R; δz/R]
    end
    R_meas = Diagonal(1e-4 * ones(4))
    return NoisyObservation(h, Matrix(R_meas), 4)
end

# ============================================================================
# Tests
# ============================================================================

@testset "Inverse Game Problem" begin

    # ------------------------------------------------------------------
    @testset "PlayerKnowledge types" begin
        stage    = DiagonalLQStageCost([1.0, 0.1], [0.1])
        terminal = DiagonalLQTerminalCost([10.0, 1.0])
        obj      = PlayerObjective(1, stage, terminal)

        known   = KnownObjective(obj)
        unknown = UnknownObjective()

        @test known   isa PlayerKnowledge
        @test unknown isa PlayerKnowledge
        @test known.objective === obj
        @test known.objective.player_id == 1

        # Type checks used by accessors
        @test known   isa KnownObjective
        @test unknown isa UnknownObjective
        @test !(known   isa UnknownObjective)
        @test !(unknown isa KnownObjective)
    end

    # ------------------------------------------------------------------
    @testset "ObservationModel — FullStateObservation" begin
        obs = FullStateObservation(6)

        @test observation_dim(obs) == 6

        x = randn(6)
        y = observe(obs, x)
        @test y == x
        @test y !== x     # copy, not alias
    end

    # ------------------------------------------------------------------
    @testset "ObservationModel — NoisyObservation" begin
        n_total = 12
        h       = x -> x[1:4]          # trivial projection
        R       = Diagonal(1e-6 * ones(4))
        obs     = NoisyObservation(h, Matrix(R), 4)

        @test observation_dim(obs) == 4

        x  = randn(n_total)
        ys = [observe(obs, x) for _ in 1:500]
        ȳ  = sum(ys) / length(ys)

        # Mean should be close to h(x) under low noise
        @test norm(ȳ - h(x)) < 0.01

        # NoisyObservation must validate R at construction
        R_bad = -1.0 * Matrix(I, 4, 4)   # not positive definite
        @test_throws AssertionError NoisyObservation(h, R_bad, 4)

        # Dimension mismatch
        R_wrong = Diagonal(ones(3))
        @test_throws AssertionError NoisyObservation(h, Matrix(R_wrong), 4)
    end

    # ------------------------------------------------------------------
    @testset "ObservationData — construction and mutation" begin
        data = ObservationData{Float64}()
        @test isempty(data)
        @test length(data) == 0

        x1 = [1.0, 2.0, 3.0]
        push_observation!(data, x1, 0.0)
        @test length(data) == 1
        @test data.states[1] == x1
        @test data.states[1] !== x1    # push_observation! must copy

        # Mutating original should not affect stored observation
        x1 .= 0.0
        @test data.states[1] == [1.0, 2.0, 3.0]

        push_observation!(data, [4.0, 5.0, 6.0], 1.0)
        @test length(data) == 2
        @test data.times == [0.0, 1.0]

        # Pre-loaded constructor
        states = [[1.0, 2.0], [3.0, 4.0]]
        times  = [0.0, 1.0]
        data2  = ObservationData{Float64}(states, times)
        @test length(data2) == 2

        # Length mismatch must error
        @test_throws AssertionError ObservationData{Float64}(states, [0.0])
    end

    # ------------------------------------------------------------------
    @testset "InverseSolverState — construction" begin
        state = InverseSolverState{Float64}()
        @test isempty(state.observations)
        @test state.t_current == 0.0

        push_observation!(state.observations, [1.0, 2.0], 0.5)
        @test length(state.observations) == 1
        state.t_current = 0.5
        @test state.t_current == 0.5
    end

    # ------------------------------------------------------------------
    @testset "InversePDGNEProblem — two-player, one unknown" begin
        # Chief (ego, known objective) vs. deputy (unknown intent)
        n, m = 6, 3
        chief_spec  = make_player(1, n, m, zeros(n))
        deputy_spec = make_player(2, n, m, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        stage    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal = DiagonalLQTerminalCost(10.0 * ones(n))
        chief_obj = PlayerObjective(1, stage, terminal)

        knowledge = [KnownObjective(chief_obj), UnknownObjective()]
        obs_model = FullStateObservation(2n)
        wrapper   = ZeroDynamicsWrapper(1.0)

        prob = InversePDGNEProblem(
            [chief_spec, deputy_spec],
            knowledge,
            [],
            obs_model,
            wrapper,
            400.0, 1.0
        )

        @test prob isa InverseGameProblem{Float64}
        @test prob.n_players == 2
        @test observation_dim(prob.observation_model) == 2n

        # Accessor correctness
        @test unknown_players(prob) == [2]
        @test known_players(prob)   == [1]
        @test n_unknown(prob)       == 1

        @test known_objective(prob, 1).player_id == 1
        @test_throws ErrorException known_objective(prob, 2)   # player 2 is unknown

        # Metadata built at construction and consistent with PlayerSpecs
        @test prob.metadata.state_dims   == [n, n]
        @test prob.metadata.control_dims == [m, m]
        @test prob.metadata.state_offsets   == [0, n]
        @test prob.metadata.control_offsets == [0, m]
    end

    # ------------------------------------------------------------------
    @testset "InversePDGNEProblem — three-player, two unknown" begin
        specs = [make_player(i, 4, 2, zeros(4)) for i in 1:3]

        stage_1   = DiagonalLQStageCost(ones(4), 0.1 * ones(2))
        terminal_1 = DiagonalLQTerminalCost(10.0 * ones(4))
        obj_1     = PlayerObjective(1, stage_1, terminal_1)

        knowledge = [KnownObjective(obj_1), UnknownObjective(), UnknownObjective()]
        obs_model = FullStateObservation(12)
        wrapper   = ZeroDynamicsWrapper(0.1)

        prob = InversePDGNEProblem(
            specs, knowledge, [],
            obs_model, wrapper,
            10.0, 0.1
        )

        @test prob.n_players  == 3
        @test n_unknown(prob) == 2
        @test unknown_players(prob) == [2, 3]
        @test known_players(prob)   == [1]

        @test prob.metadata.state_offsets   == [0, 4, 8]
        @test prob.metadata.control_offsets == [0, 2, 4]
    end

    # ------------------------------------------------------------------
    @testset "InversePDGNEProblem — with shared constraints" begin
        n, m = 6, 3
        spec1 = make_player(1, n, m, zeros(n))
        spec2 = make_player(2, n, m, ones(n))

        collision = NonlinearConstraint(
            (x, u, p, t) -> [5.0^2 - sum((x[1:3] - x[7:9]).^2)],
            1,
            constraint_type = :inequality
        )
        shared = SharedConstraint(collision, [1, 2])

        stage_1    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal_1 = DiagonalLQTerminalCost(10.0 * ones(n))
        obj_1      = PlayerObjective(1, stage_1, terminal_1)
        knowledge  = [KnownObjective(obj_1), UnknownObjective()]
        wrapper    = ZeroDynamicsWrapper(1.0)

        prob = InversePDGNEProblem(
            [spec1, spec2], knowledge, [shared],
            FullStateObservation(2n), wrapper,
            20.0, 1.0
        )

        @test length(prob.shared_constraints) == 1
        @test prob.shared_constraints[1].players == [1, 2]
    end

    # ------------------------------------------------------------------
    @testset "InversePDGNEProblem — assertion guards" begin
        n, m  = 4, 2
        spec  = make_player(1, n, m, zeros(n))
        obs   = FullStateObservation(n)
        wrap  = ZeroDynamicsWrapper(0.1)
    
        stage_1    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal_1 = DiagonalLQTerminalCost(10.0 * ones(n))
        obj_1      = PlayerObjective(1, stage_1, terminal_1)
    
        # All players known — must error (at least one must be unknown).
        # Vector{KnownObjective} now dispatches correctly after the widening fix.
        @test_throws AssertionError InversePDGNEProblem(
            [spec],
            [KnownObjective(obj_1)],       # Vector{KnownObjective} <: AbstractVector{<:PlayerKnowledge} ✓
            [], obs, wrap, 5.0, 0.1
        )
    
        # knowledge length mismatch
        @test_throws AssertionError InversePDGNEProblem(
            [spec],
            PlayerKnowledge[KnownObjective(obj_1), UnknownObjective()],  # 2 tags, 1 player
            [], obs, wrap, 5.0, 0.1
        )
    
        # KnownObjective player_id mismatch.
        # Must be tested separately from the all-known case: use a two-player
        # problem so the "at least one unknown" assertion passes, isolating
        # the id-mismatch assertion.
        spec2      = make_player(2, n, m, ones(n))
        obj_wrong_id = PlayerObjective(99, stage_1, terminal_1)  # id 99 ≠ spec id 1
        @test_throws AssertionError InversePDGNEProblem(
            [spec, spec2],
            PlayerKnowledge[KnownObjective(obj_wrong_id), UnknownObjective()],
            [], obs, wrap, 5.0, 0.1
        )
    
        # Duplicate player IDs
        spec_dup = make_player(1, n, m, zeros(n))   # same id as spec
        @test_throws AssertionError InversePDGNEProblem(
            [spec, spec_dup],
            PlayerKnowledge[KnownObjective(obj_1), UnknownObjective()],
            [], obs, wrap, 5.0, 0.1
        )
    end

    # ------------------------------------------------------------------
    @testset "as_forward_problem — structure and metadata reuse" begin
        n, m = 6, 3
        chief_spec  = make_player(1, n, m, zeros(n))
        deputy_spec = make_player(2, n, m, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        stage_c    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal_c = DiagonalLQTerminalCost(10.0 * ones(n))
        chief_obj  = PlayerObjective(1, stage_c, terminal_c)

        knowledge = [KnownObjective(chief_obj), UnknownObjective()]
        prob = InversePDGNEProblem(
            [chief_spec, deputy_spec], knowledge, [],
            FullStateObservation(2n), ZeroDynamicsWrapper(1.0),
            400.0, 1.0
        )

        # Hypothesized objective for the unknown deputy
        stage_h    = DiagonalLQStageCost(2.0 * ones(n), 0.5 * ones(m))
        terminal_h = DiagonalLQTerminalCost(5.0 * ones(n))
        hyp_obj    = PlayerObjective(2, stage_h, terminal_h)
        hypothesized = Dict{Int, PlayerObjective}(2 => hyp_obj)

        fwd = as_forward_problem(prob, hypothesized)

        @test fwd isa GameProblem{Float64}
        @test fwd.n_players == 2

        # Known player objective is unchanged
        @test get_objective(fwd, 1).player_id == 1
        @test get_objective(fwd, 1).stage_cost === chief_obj.stage_cost

        # Unknown player objective is the hypothesized one
        @test get_objective(fwd, 2).player_id == 2
        @test get_objective(fwd, 2).stage_cost === hyp_obj.stage_cost

        # Metadata object is reused directly (identity, not copy)
        @test fwd.metadata === prob.metadata

        # Wrong keys must error
        @test_throws AssertionError as_forward_problem(prob, Dict{Int, PlayerObjective}())
        @test_throws AssertionError as_forward_problem(
            prob,
            Dict{Int, PlayerObjective}(1 => hyp_obj)  # key 1 is known, not unknown
        )
    end

    # ------------------------------------------------------------------
    @testset "as_forward_problem — metadata identity under repeated calls" begin
        # Verify that N_ensemble calls to as_forward_problem do not
        # allocate new metadata objects (regression test for the EnKF hot path)
        n, m = 4, 2
        spec1 = make_player(1, n, m, zeros(n))
        spec2 = make_player(2, n, m, ones(n))

        stage_1    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal_1 = DiagonalLQTerminalCost(10.0 * ones(n))
        obj_1      = PlayerObjective(1, stage_1, terminal_1)
        knowledge  = [KnownObjective(obj_1), UnknownObjective()]

        prob = InversePDGNEProblem(
            [spec1, spec2], knowledge, [],
            FullStateObservation(2n), ZeroDynamicsWrapper(0.1),
            5.0, 0.1
        )

        stage_h    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal_h = DiagonalLQTerminalCost(10.0 * ones(n))
        hyp_obj    = PlayerObjective(2, stage_h, terminal_h)
        hyp        = Dict{Int, PlayerObjective}(2 => hyp_obj)

        fwd_problems = [as_forward_problem(prob, hyp) for _ in 1:20]

        # All calls must return the same metadata object
        for fp in fwd_problems
            @test fp.metadata === prob.metadata
        end
    end

    # ------------------------------------------------------------------
    @testset "NoisyObservation — CWH range/LOS model" begin
        # Validates the observation model used in the Balo et al. application
        obs   = make_cwh_obs(12)
        x_rel = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0,   # deputy relative state (6-DOF)
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]    # chief state (padding)

        @test observation_dim(obs) == 4

        # Under tiny noise, observed range ≈ true range = 5.0
        ys    = [observe(obs, x_rel) for _ in 1:1000]
        ȳ     = sum(ys) / length(ys)
        R_true = sqrt(3.0^2 + 4.0^2)    # = 5.0
        @test abs(ȳ[1] - R_true) < 0.01

        # LOS unit vector should be approximately [3/5, 4/5, 0]
        @test abs(ȳ[2] - 3.0/5.0) < 0.01
        @test abs(ȳ[3] - 4.0/5.0) < 0.01
        @test abs(ȳ[4])           < 0.01
    end

    # ------------------------------------------------------------------
    @testset "Convenience constructor — no shared constraints" begin
        n, m  = 4, 2
        spec1 = make_player(1, n, m, zeros(n))
        spec2 = make_player(2, n, m, ones(n))

        stage_1    = DiagonalLQStageCost(ones(n), 0.1 * ones(m))
        terminal_1 = DiagonalLQTerminalCost(10.0 * ones(n))
        obj_1      = PlayerObjective(1, stage_1, terminal_1)
        knowledge  = [KnownObjective(obj_1), UnknownObjective()]

        prob = InversePDGNEProblem(
            [spec1, spec2], knowledge,
            FullStateObservation(2n), ZeroDynamicsWrapper(0.1),
            5.0, 0.1
        )

        @test prob isa InverseGameProblem{Float64}
        @test isempty(prob.shared_constraints)
        @test n_unknown(prob) == 1
    end

end