using Test
using LinearAlgebra
using ForwardDiff
using DifferentialGamesBase

# Explicitly import every package function we extend with test-local methods.
# Julia requires this — a function defined in another module cannot be extended
# in Main without explicit import.
import DifferentialGamesBase:
    n_players,
    evaluate_cost_term,
    is_quadratic,
    is_separable_term

# ============================================================================
# Phase 0 test suite
#
# Runs against the installed DifferentialGamesBase package.
# No stubs or source includes — the package is loaded by runtests.jl.
#
# Concrete test subtypes (TestDeterministicGame, TrackStateTerm, etc.) are
# defined at TOP LEVEL so their method overloads are visible to module-scoped
# dispatch from cost_terms.jl. This is the same rule as Phase 0 originally,
# but now they subtype the package abstract types rather than stub types.
# ============================================================================

# Concrete test subtypes — minimal AbstractGameProblem implementations
# ============================================================================

struct TestDeterministicGame{T} <: AbstractDeterministicGame{T}
    np::Int
end
n_players(g::TestDeterministicGame) = g.np

struct TestStochasticGame{T}    <: AbstractStochasticGame{T} end
struct TestPOGame{T}            <: AbstractPartiallyObservableGame{T} end
struct TestInverseGame{T}       <: AbstractInverseGameProblem{T} end

# ============================================================================
# Phase 0a: AbstractGameProblem hierarchy
# ============================================================================

@testset "Phase 0a: AbstractGameProblem hierarchy" begin

    @testset "Type hierarchy" begin
        @test TestDeterministicGame{Float64} <: AbstractDeterministicGame{Float64}
        @test TestDeterministicGame{Float64} <: AbstractGameProblem{Float64}
        @test TestStochasticGame{Float64}    <: AbstractStochasticGame{Float64}
        @test TestPOGame{Float64}            <: AbstractPartiallyObservableGame{Float64}
        @test TestInverseGame{Float64}       <: AbstractInverseGameProblem{Float64}

        # Branches are mutually exclusive
        @test !(AbstractDeterministicGame{Float64} <: AbstractStochasticGame{Float64})
        @test !(AbstractStochasticGame{Float64}    <: AbstractDeterministicGame{Float64})
        @test !(AbstractInverseGameProblem{Float64} <: AbstractDeterministicGame{Float64})
    end

    @testset "Trait queries: is_deterministic" begin
        @test  is_deterministic(TestDeterministicGame{Float64}(2))
        @test !is_deterministic(TestStochasticGame{Float64}())
        @test !is_deterministic(TestPOGame{Float64}())
        @test !is_deterministic(TestInverseGame{Float64}())
    end

    @testset "Trait queries: is_stochastic" begin
        @test  is_stochastic(TestStochasticGame{Float64}())
        @test !is_stochastic(TestDeterministicGame{Float64}(2))
    end

    @testset "Trait queries: is_partially_observable" begin
        @test  is_partially_observable(TestPOGame{Float64}())
        @test !is_partially_observable(TestDeterministicGame{Float64}(2))
    end

    @testset "Trait queries: is_inverse" begin
        @test  is_inverse(TestInverseGame{Float64}())
        @test !is_inverse(TestDeterministicGame{Float64}(2))
    end

    @testset "n_players interface" begin
        g = TestDeterministicGame{Float64}(3)
        @test n_players(g) == 3
    end

    @testset "Deterministic type parameter" begin
        # T propagates correctly
        @test TestDeterministicGame{Float32} <: AbstractGameProblem{Float32}
        @test !(TestDeterministicGame{Float32} <: AbstractGameProblem{Float64})
    end
end

# ============================================================================
# Phase 0b: AbstractInformationStructure
# ============================================================================

@testset "Phase 0b: InformationStructure" begin

    @testset "PerfectStateInformation" begin
        info = PerfectStateInformation()
        @test info isa AbstractInformationStructure
        @test !requires_belief_state(info)
        @test  is_feedback_compatible(info)
        @test !is_open_loop(info)
    end

    @testset "OpenLoopInformation" begin
        info = OpenLoopInformation()
        @test !requires_belief_state(info)
        @test !is_feedback_compatible(info)
        @test  is_open_loop(info)
    end

    @testset "PrivateObservation — valid construction" begin
        h     = (x, v, p, t) -> x[1:2] + v
        R_obs = Matrix{Float64}(I, 2, 2)
        info  = PrivateObservation(h, 2, R_obs)

        @test info isa PrivateObservation
        @test info.obs_dim == 2
        @test  requires_belief_state(info)
        @test  is_feedback_compatible(info)
        @test !is_open_loop(info)
    end

    @testset "PrivateObservation — invalid construction" begin
        h = (x, v, p, t) -> x[1:2] + v
        # Not positive definite
        @test_throws AssertionError PrivateObservation(h, 2, -Matrix{Float64}(I, 2, 2))
        # Dimension mismatch
        @test_throws AssertionError PrivateObservation(h, 2, Matrix{Float64}(I, 3, 3))
        # Zero obs_dim
        @test_throws AssertionError PrivateObservation(h, 0, Matrix{Float64}(I, 1, 1))
    end

    @testset "SharedObservation" begin
        h     = (x, v, p, t) -> x[1:3] + v
        R_obs = Diagonal(fill(0.1, 3)) |> Matrix
        info  = SharedObservation(h, 3, R_obs)

        @test info isa SharedObservation
        @test info.obs_dim == 3
        @test !requires_belief_state(info)
        @test  is_feedback_compatible(info)
    end

    @testset "AsymmetricInformation" begin
        info = AsymmetricInformation(1)
        @test info.leader_id == 1
        @test !is_feedback_compatible(info)
        @test !requires_belief_state(info)
        @test_throws AssertionError AsymmetricInformation(0)
        @test_throws AssertionError AsymmetricInformation(-3)
    end

    @testset "_infer_game_class" begin
        R1 = Matrix{Float64}(I, 1, 1)
        h  = (x, v, p, t) -> x[1:1] + v

        all_perfect = [PerfectStateInformation(), PerfectStateInformation()]
        mixed_open  = [PerfectStateInformation(), OpenLoopInformation()]
        has_private = [PerfectStateInformation(), PrivateObservation(h, 1, R1)]
        has_shared  = [PerfectStateInformation(), SharedObservation(h, 1, R1)]
        has_asymm   = [PerfectStateInformation(), AsymmetricInformation(1)]
        # Stackelberg + PO — Stackelberg should take priority
        asymm_po    = [AsymmetricInformation(1), PrivateObservation(h, 1, R1)]

        @test _infer_game_class(all_perfect) == :deterministic
        @test _infer_game_class(mixed_open)  == :deterministic
        @test _infer_game_class(has_private) == :partially_observable
        @test _infer_game_class(has_shared)  == :partially_observable
        @test _infer_game_class(has_asymm)   == :stackelberg
        @test _infer_game_class(asymm_po)    == :stackelberg
    end
end

# ============================================================================
# Phase 0c: AbstractPlayerDynamics
# ============================================================================

@testset "Phase 0c: AbstractPlayerDynamics" begin

    # ── ContinuousPlayerDynamics ────────────────────────────────────────────

    @testset "ContinuousPlayerDynamics — construction" begin
        f   = (x, u, p, t) -> [x[2]; u[1]]
        dyn = ContinuousPlayerDynamics(f, 2, 1)

        @test dyn isa AbstractPlayerDynamics
        @test  is_continuous(dyn)
        @test !is_discrete(dyn)
        @test !is_linear(dyn)
        @test  is_separable_dynamics(dyn)
        @test dyn.state_dim   == 2
        @test dyn.control_dim == 1

        @test_throws AssertionError ContinuousPlayerDynamics(f, 0, 1)
        @test_throws AssertionError ContinuousPlayerDynamics(f, 2, 0)
    end

    @testset "ContinuousPlayerDynamics — evaluation" begin
        f   = (x, u, p, t) -> [x[2]; u[1]]
        dyn = ContinuousPlayerDynamics(f, 2, 1)
        x   = [1.0, 0.5]
        u   = [2.0]

        ẋ = evaluate_player_dynamics(dyn, x, u, nothing, 0.0)
        @test ẋ ≈ [0.5, 2.0]
    end

    @testset "ContinuousPlayerDynamics — Jacobian via AD" begin
        f   = (x, u, p, t) -> [x[2]; u[1]]
        dyn = ContinuousPlayerDynamics(f, 2, 1)
        x   = [1.0, 0.5]
        u   = [2.0]

        Jx, Ju = player_dynamics_jacobian(dyn, x, u, nothing, 0.0)
        @test Jx ≈ [0.0 1.0; 0.0 0.0]
        @test Ju ≈ reshape([0.0; 1.0], 2, 1)
    end

    @testset "ContinuousPlayerDynamics — analytical Jacobian override" begin
        f   = (x, u, p, t) -> [x[2]; u[1]]
        jac = (x, u, p, t) -> (
            [0.0 1.0; 0.0 0.0],
            reshape([0.0; 1.0], 2, 1)
        )
        dyn = ContinuousPlayerDynamics(f, 2, 1; jacobian = jac)
        x   = [1.0, 0.5]
        u   = [2.0]

        Jx, Ju = player_dynamics_jacobian(dyn, x, u, nothing, 0.0)
        @test Jx ≈ [0.0 1.0; 0.0 0.0]
        @test Ju ≈ reshape([0.0; 1.0], 2, 1)
    end

    @testset "ContinuousPlayerDynamics — ForwardDiff compatibility" begin
        f   = (x, u, p, t) -> [x[2]; u[1]]
        dyn = ContinuousPlayerDynamics(f, 2, 1)
        x   = [1.0, 0.5]
        u   = [2.0]
        # Must not error when called with dual numbers
        @test_nowarn ForwardDiff.jacobian(
            z -> evaluate_player_dynamics(dyn, z[1:2], z[3:3], nothing, 0.0),
            vcat(x, u)
        )
    end

    # ── DiscretePlayerDynamics ──────────────────────────────────────────────

    @testset "DiscretePlayerDynamics — evaluation" begin
        dt  = 0.1
        f   = (x, u, p, k) -> [x[1] + dt*x[2]; x[2] + dt*u[1]]
        dyn = DiscretePlayerDynamics(f, 2, 1)

        @test  is_discrete(dyn)
        @test !is_continuous(dyn)

        x      = [1.0, 0.5]
        u      = [2.0]
        x_next = evaluate_player_dynamics(dyn, x, u, nothing, 1)
        @test x_next ≈ [1.0 + dt*0.5, 0.5 + dt*2.0]
    end

    # ── CoupledPlayerDynamics ────────────────────────────────────────────────

    @testset "CoupledPlayerDynamics — construction" begin
        # Player 1 attracted toward player 2's position
        f1(xi, ui, xo, p, t) = [xi[2] + 0.1*(xo[1] - xi[1]); ui[1]]
        dyn = CoupledPlayerDynamics(f1, 2, 1)

        @test dyn isa AbstractPlayerDynamics
        @test  is_continuous(dyn)
        @test !is_discrete(dyn)
        @test !is_separable_dynamics(dyn)
        @test !is_linear(dyn)
    end

    @testset "CoupledPlayerDynamics — evaluation" begin
        f1  = (xi, ui, xo, p, t) -> [xi[2] + 0.1*(xo[1] - xi[1]); ui[1]]
        dyn = CoupledPlayerDynamics(f1, 2, 1)

        xi      = [0.0, 1.0]
        ui      = [0.5]
        x_other = [3.0, 0.0]    # player 2 at x=3

        ẋᵢ = evaluate_player_dynamics(dyn, xi, ui, x_other, nothing, 0.0)
        # ẋᵢ[1] = xi[2] + 0.1*(x_other[1] - xi[1]) = 1.0 + 0.1*3.0 = 1.3
        # ẋᵢ[2] = ui[1] = 0.5
        @test ẋᵢ ≈ [1.3, 0.5]
    end

    @testset "CoupledPlayerDynamics — Jacobian via AD" begin
        f1  = (xi, ui, xo, p, t) -> [xi[2] + 0.1*(xo[1] - xi[1]); ui[1]]
        dyn = CoupledPlayerDynamics(f1, 2, 1)

        xi      = [0.0, 1.0]
        ui      = [0.5]
        x_other = [3.0, 0.0]

        Jx, Ju, Jxo = player_dynamics_jacobian(dyn, xi, ui, x_other, nothing, 0.0)

        # ∂ẋᵢ/∂xᵢ: row 1 = [-0.1, 1.0], row 2 = [0.0, 0.0]
        @test Jx  ≈ [-0.1 1.0; 0.0 0.0]
        # ∂ẋᵢ/∂uᵢ: row 1 = [0.0], row 2 = [1.0]
        @test Ju  ≈ reshape([0.0; 1.0], 2, 1)
        # ∂ẋᵢ/∂xo: row 1 = [0.1, 0.0], row 2 = [0.0, 0.0]
        @test Jxo ≈ [0.1 0.0; 0.0 0.0]
    end

    @testset "CoupledPlayerDynamics — ForwardDiff through x_others" begin
        f1  = (xi, ui, xo, p, t) -> xi + xo + ui[1]*ones(length(xi))
        dyn = CoupledPlayerDynamics(f1, 2, 1)
        xi  = [1.0, 2.0]
        ui  = [0.5]
        xo  = [0.1, 0.2]

        @test_nowarn ForwardDiff.jacobian(
            xo_var -> evaluate_player_dynamics(dyn, xi, ui, xo_var, nothing, 0.0),
            xo
        )
    end

    # ── LinearPlayerDynamics ────────────────────────────────────────────────

    @testset "LinearPlayerDynamics — LTI evaluation" begin
        A   = [1.0 0.1; 0.0 1.0]
        B   = reshape([0.0; 0.1], 2, 1)
        dyn = LinearPlayerDynamics(A, B)

        @test  is_linear(dyn)
        @test  is_discrete(dyn)
        @test !is_continuous(dyn)
        @test !is_ltv(dyn)
        @test !is_separable_dynamics(dyn)
        @test dyn.state_dim   == 2
        @test dyn.control_dim == 1

        x      = [1.0, 0.5]
        u      = [2.0]
        x_next = evaluate_player_dynamics(dyn, x, u, nothing, 1)
        @test x_next ≈ A * x + B * u
    end

    @testset "LinearPlayerDynamics — LTI Jacobian (exact)" begin
        A   = [1.0 0.1; 0.0 1.0]
        B   = reshape([0.0; 0.1], 2, 1)
        dyn = LinearPlayerDynamics(A, B)
        x   = [1.0, 0.5]
        u   = [2.0]

        Jx, Ju = player_dynamics_jacobian(dyn, x, u, nothing, 1)
        @test Jx ≈ A
        @test Ju ≈ B
        # k is ignored for LTI
        Jx99, _ = player_dynamics_jacobian(dyn, x, u, nothing, 99)
        @test Jx99 ≈ A
    end

    @testset "LinearPlayerDynamics — LTV" begin
        N     = 4
        A_seq = [Matrix{Float64}(I, 2, 2) * Float64(k) for k in 1:N]
        B_seq = [reshape([0.1*k; 0.0], 2, 1) for k in 1:N]
        dyn   = LinearPlayerDynamics(A_seq, B_seq)

        @test is_ltv(dyn)
        @test get_A(dyn, 3) === A_seq[3]
        @test get_B(dyn, 3) === B_seq[3]

        x      = [1.0, 0.0]
        u      = [1.0]
        x_next = evaluate_player_dynamics(dyn, x, u, nothing, 2)
        @test x_next ≈ A_seq[2] * x + B_seq[2] * u
    end

    @testset "LinearPlayerDynamics — dimension validation" begin
        A = Matrix{Float64}(I, 2, 2)
        # B has wrong number of rows
        B_bad = ones(3, 1)
        @test_throws AssertionError LinearPlayerDynamics(A, B_bad)
    end

    @testset "LinearPlayerDynamics — LTV length mismatch" begin
        A_seq = [Matrix{Float64}(I, 2, 2) for _ in 1:3]
        B_seq = [ones(2, 1) for _ in 1:5]
        @test_throws AssertionError LinearPlayerDynamics(A_seq, B_seq)
    end

    @testset "is_continuous / is_discrete consistency" begin
        fa  = (x, u, p, t) -> x + u
        fd  = (x, u, p, k) -> x + u
        fc  = (xi, ui, xo, p, t) -> xi + xo
        A   = Matrix{Float64}(I, 2, 2)
        B   = Matrix{Float64}(I, 2, 2)

        dyn_cont   = ContinuousPlayerDynamics(fa, 2, 2)
        dyn_disc   = DiscretePlayerDynamics(fd, 2, 2)
        dyn_coupled = CoupledPlayerDynamics(fc, 2, 2)
        dyn_lin    = LinearPlayerDynamics(A, B)

        @test  is_continuous(dyn_cont)    && !is_discrete(dyn_cont)
        @test !is_continuous(dyn_disc)    &&  is_discrete(dyn_disc)
        @test  is_continuous(dyn_coupled) && !is_discrete(dyn_coupled)
        @test !is_continuous(dyn_lin)     &&  is_discrete(dyn_lin)
    end
end

# ============================================================================
# Phase 0d: concrete test cost term types — MUST be top-level
#
# Method definitions inside @testset blocks are invisible to module-scoped
# generic function dispatch. All struct definitions and their evaluate_cost_term,
# is_quadratic, and is_separable_term methods must be at module scope so that
# the composite implementations in cost_terms.jl can dispatch to them.
# ============================================================================

struct TrackStateTerm <: AbstractCostTerm
    Q::Matrix{Float64}
    offset::Int    # 0-based start index into joint state vector
end
function evaluate_cost_term(t::TrackStateTerm, x, u, p, ts)
    xi = player_slice(x, t.offset, size(t.Q, 1))
    return 0.5 * dot(xi, t.Q * xi)
end
is_quadratic(::TrackStateTerm)      = true
is_separable_term(::TrackStateTerm) = true

struct RegInputTerm <: AbstractCostTerm
    R::Matrix{Float64}
    offset::Int    # 0-based start index into joint control vector
end
function evaluate_cost_term(t::RegInputTerm, x, u, p, ts)
    ui = player_slice(u, t.offset, size(t.R, 1))
    return 0.5 * dot(ui, t.R * ui)
end
is_quadratic(::RegInputTerm)      = true
is_separable_term(::RegInputTerm) = true

struct ProximityTerm <: AbstractCostTerm
    i_offset::Int; i_dim::Int
    j_offset::Int; j_dim::Int
    d_min::Float64
    weight::Float64
end
function evaluate_cost_term(t::ProximityTerm, x, u, p, ts)
    xi = player_slice(x, t.i_offset, t.i_dim)
    xj = player_slice(x, t.j_offset, t.j_dim)
    Δ  = xi - xj
    d  = sqrt(dot(Δ, Δ) + 1e-8)
    return t.weight * max(t.d_min - d, zero(eltype(x)))^2
end
is_quadratic(::ProximityTerm)      = false
is_separable_term(::ProximityTerm) = false

struct QuadTerminal <: AbstractTerminalCostTerm
    Qf::Matrix{Float64}
    offset::Int
end
function evaluate_cost_term(t::QuadTerminal, x, p)
    xi = player_slice(x, t.offset, size(t.Qf, 1))
    return 0.5 * dot(xi, t.Qf * xi)
end
is_quadratic(::QuadTerminal) = true

struct NonQuadTerm <: AbstractCostTerm end
evaluate_cost_term(::NonQuadTerm, x, u, p, t) = norm(x)
is_quadratic(::NonQuadTerm)      = false
is_separable_term(::NonQuadTerm) = false

struct LinTerminalTerm <: AbstractTerminalCostTerm
    c::Vector{Float64}
end
evaluate_cost_term(t::LinTerminalTerm, x, p) = dot(t.c, x)
is_quadratic(::LinTerminalTerm) = false

# ============================================================================
# Phase 0d: AbstractCostTerm + composition
# ============================================================================

@testset "Phase 0d: AbstractCostTerm composition" begin

    Q  = Matrix{Float64}(I, 2, 2)
    R  = Matrix{Float64}(I, 1, 1)
    Qf = 2.0 * Matrix{Float64}(I, 2, 2)

    # Two players: player 1 at x[1:2], player 2 at x[3:4]
    # Controls: player 1 at u[1:1], player 2 at u[2:2]
    qt  = TrackStateTerm(Q, 0)
    rt  = RegInputTerm(R, 0)
    qft = QuadTerminal(Qf, 0)
    prx = ProximityTerm(0, 2, 2, 2, 1.0, 100.0)

    x_joint = [1.0, 2.0, 4.0, 5.0]     # [x1; x2]
    u_joint = [0.5, 0.3]               # [u1; u2]

    # ── Individual evaluation ───────────────────────────────────────────────

    @testset "player_slice" begin
        v = [10.0, 20.0, 30.0, 40.0]
        s = player_slice(v, 0, 2)
        @test s == [10.0, 20.0]
        s2 = player_slice(v, 2, 2)
        @test s2 == [30.0, 40.0]
        # Returns a view (no allocation)
        @test typeof(s) <: SubArray
    end

    @testset "Individual term evaluation" begin
        x1 = x_joint[1:2]
        u1 = u_joint[1:1]

        @test evaluate_cost_term(qt, x_joint, u_joint, nothing, 1) ≈
              0.5 * dot(x1, Q * x1)
        @test evaluate_cost_term(rt, x_joint, u_joint, nothing, 1) ≈
              0.5 * dot(u1, R * u1)
        @test evaluate_cost_term(qft, x_joint, nothing)            ≈
              0.5 * dot(x1, Qf * x1)
    end

    @testset "Cross-agent proximity coupling" begin
        x1 = x_joint[1:2]   # [1, 2]
        x2 = x_joint[3:4]   # [4, 5]
        Δ  = x1 - x2         # [-3, -3]
        d  = sqrt(dot(Δ, Δ) + 1e-8)
        expected = 100.0 * max(1.0 - d, 0.0)^2   # d > 1, so penalty = 0

        @test evaluate_cost_term(prx, x_joint, u_joint, nothing, 1) ≈ expected atol=1e-6

        # Close agents — proximity penalty should be nonzero
        x_close = [0.0, 0.0, 0.3, 0.0]
        val_close = evaluate_cost_term(prx, x_close, u_joint, nothing, 1)
        @test val_close > 0.0
    end

    # ── Composition ─────────────────────────────────────────────────────────

    @testset "Stage composition via +" begin
        comp2 = qt + rt
        @test comp2 isa CompositeCostTerm
        @test length(comp2.terms) == 2

        comp3 = qt + rt + prx
        @test comp3 isa CompositeCostTerm
        @test length(comp3.terms) == 3   # flat, not nested

        comp4 = (qt + rt) + (prx + prx)
        @test comp4 isa CompositeCostTerm
        @test length(comp4.terms) == 4   # fully flattened
    end

    @testset "Stage composite evaluation" begin
        comp = qt + rt
        expected = evaluate_cost_term(qt, x_joint, u_joint, nothing, 1) +
                   evaluate_cost_term(rt, x_joint, u_joint, nothing, 1)
        @test evaluate_cost_term(comp, x_joint, u_joint, nothing, 1) ≈ expected
    end

    @testset "Composite with coupling term" begin
        comp = qt + prx
        v1   = evaluate_cost_term(qt,  x_joint, u_joint, nothing, 1)
        v2   = evaluate_cost_term(prx, x_joint, u_joint, nothing, 1)
        @test evaluate_cost_term(comp, x_joint, u_joint, nothing, 1) ≈ v1 + v2
    end

    @testset "Terminal composition via +" begin
        qt2  = QuadTerminal(Qf, 2)   # player 2 terminal (offset 2)
        comp = qft + qt2
        @test comp isa CompositeTerminalCostTerm
        @test length(comp.terms) == 2
        expected = evaluate_cost_term(qft, x_joint, nothing) +
                   evaluate_cost_term(qt2, x_joint, nothing)
        @test evaluate_cost_term(comp, x_joint, nothing) ≈ expected
    end

    # ── Trait propagation ───────────────────────────────────────────────────

    @testset "is_quadratic propagation" begin
        @test  is_quadratic(qt)
        @test  is_quadratic(rt)
        @test !is_quadratic(prx)

        @test  is_quadratic(qt + rt)       # both quadratic
        @test !is_quadratic(qt + prx)      # proximity is not quadratic
        @test  is_quadratic(qft)
    end

    @testset "is_separable_term propagation" begin
        @test  is_separable_term(qt)
        @test  is_separable_term(rt)
        @test !is_separable_term(prx)

        @test  is_separable_term(qt + rt)   # both separable
        @test !is_separable_term(qt + prx)  # prx is not separable
    end

    # ── Gradient and Hessian (ForwardDiff fallback) ──────────────────────────

    @testset "Gradient via ForwardDiff" begin
        comp = qt + rt
        gx, gu = cost_term_gradient(comp, x_joint, u_joint, nothing, 1)

        # ∇ₓ should be Q * x1 in the first 2 elements, zero elsewhere
        @test gx[1:2] ≈ Q * x_joint[1:2]
        @test gx[3:4] ≈ zeros(2)
        # ∇ᵤ should be R * u1 in first element, zero in second
        @test gu[1:1] ≈ R * u_joint[1:1]
        @test gu[2]   ≈ 0.0
    end

    @testset "Hessian via ForwardDiff" begin
        comp = qt + rt
        Hxx, Huu, Hxu = cost_term_hessian(comp, x_joint, u_joint, nothing, 1)

        # Top-left 2×2 block of Hxx should be Q
        @test Hxx[1:2, 1:2] ≈ Q
        @test norm(Hxx[3:4, :]) < 1e-12  # Player 2 state untouched
        # Huu should be R in top-left, zeros elsewhere
        @test Huu[1, 1] ≈ R[1, 1]
        @test Huu[2, 2] ≈ 0.0
        # No cross terms for sum of separable quadratic terms
        @test norm(Hxu) < 1e-12
    end

    @testset "Proximity Hessian is nonzero (cross-agent coupling)" begin
        x_close = [0.0, 0.0, 0.3, 0.0]
        Hxx, Huu, Hxu = cost_term_hessian(prx, x_close, u_joint, nothing, 1)
        # The cross-player Hessian block (player i vs player j position) is nonzero
        @test norm(Hxx[1:2, 3:4]) > 0.0
    end

    @testset "Terminal gradient and hessian" begin
        gx   = cost_term_gradient(qft, x_joint, nothing)
        Hxx  = cost_term_hessian(qft, x_joint, nothing)
        x1   = x_joint[1:2]

        @test gx[1:2] ≈ Qf * x1
        @test gx[3:4] ≈ zeros(2)
        @test Hxx[1:2, 1:2] ≈ Qf
    end

    # ── minimize() ──────────────────────────────────────────────────────────

    @testset "minimize() basic" begin
        obj = minimize(qt + rt; terminal = qft, player_id = 1)
        @test obj isa PlayerObjective
        @test obj.player_id == 1
        @test obj.scaling   ≈ 1.0

        val_stage = obj.stage_cost.func(x_joint, u_joint, nothing, 1)
        @test val_stage ≈ evaluate_cost_term(qt + rt, x_joint, u_joint, nothing, 1)

        val_term = obj.terminal_cost.func(x_joint, nothing)
        @test val_term ≈ evaluate_cost_term(qft, x_joint, nothing)
    end

    @testset "minimize() — zero terminal when terminal=nothing" begin
        obj = minimize(qt; player_id = 2)
        @test obj.terminal_cost.func([1.0, 2.0, 0.0, 0.0], nothing) ≈ 0.0
    end

    @testset "minimize() — coupling cost in objective" begin
        # Objective with cross-agent coupling term
        obj = minimize(qt + prx; terminal = qft, player_id = 1)
        x_close = [0.0, 0.0, 0.3, 0.0]
        val = obj.stage_cost.func(x_close, u_joint, nothing, 1)
        expected = evaluate_cost_term(qt + prx, x_close, u_joint, nothing, 1)
        @test val ≈ expected
    end

    @testset "minimize() — invalid arguments" begin
        @test_throws AssertionError minimize(qt; player_id = 0)
        @test_throws AssertionError minimize(qt; player_id = 1, scaling = -1.0)
        @test_throws AssertionError minimize(qt; player_id = 1, scaling = 0.0)
    end

    @testset "minimize() — scaling" begin
        obj = minimize(qt; player_id = 1, scaling = 2.5)
        @test obj.scaling ≈ 2.5
    end
end

println("\n✓ All Phase 0 tests passed.")