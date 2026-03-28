# ============================================================================
# Knowledge Specification
# ============================================================================

"""
    PlayerKnowledge

Abstract type representing what is known about a player's objective.
"""
abstract type PlayerKnowledge end

"""
    KnownObjective <: PlayerKnowledge

The player's objective is fully specified and held fixed during inference.
"""
struct KnownObjective <: PlayerKnowledge
    objective::PlayerObjective
end

"""
    UnknownObjective <: PlayerKnowledge

The player's objective is unknown and must be inferred from observations.
Parameterization (e.g., basis weights) is the solver's responsibility.
"""
struct UnknownObjective <: PlayerKnowledge end

# ============================================================================
# Observation Model
# ============================================================================

"""
    ObservationModel

Abstract type mapping joint game states to observable quantities.

# Required Interface

    observe(model, x) -> AbstractVector
    observation_dim(model) -> Int
"""
abstract type ObservationModel end

function observe end
function observation_dim end

"""
    FullStateObservation <: ObservationModel

Trivial noiseless full-state observation. Default for offline/batch settings.
"""
struct FullStateObservation <: ObservationModel
    n_total::Int
end

observe(model::FullStateObservation, x::AbstractVector) = copy(x)
observation_dim(model::FullStateObservation) = model.n_total

"""
    NoisyObservation <: ObservationModel

Additive Gaussian noise: ỹ = h(x) + w, w ~ N(0, R).

# Fields
- `h::Function` : Measurement function h: Rⁿ → Rᵖ
- `R::AbstractMatrix` : Measurement noise covariance (p×p, positive definite)
- `obs_dim::Int` : Output dimension of h
"""
struct NoisyObservation <: ObservationModel
    h::Function
    R::AbstractMatrix
    obs_dim::Int

    function NoisyObservation(h::Function, R::AbstractMatrix, obs_dim::Int)
        @assert size(R, 1) == size(R, 2) == obs_dim "R must be obs_dim × obs_dim"
        @assert isposdef(R) "Measurement noise covariance R must be positive definite"
        new(h, R, obs_dim)
    end
end

observe(model::NoisyObservation, x::AbstractVector) =
    model.h(x) .+ cholesky(model.R).L * randn(model.obs_dim)
observation_dim(model::NoisyObservation) = model.obs_dim

# ============================================================================
# Forward Solver Wrapper
# ============================================================================

"""
    ForwardSolverWrapper

Abstract type adapting external forward Nash solvers to the inverse game interface.

The inverse solver calls `predict_next_state` during the EnKF measurement
prediction step (N_ensemble times per timestep), so implementations should
exploit warm-starting and caching aggressively.

# Required Interface

    solve_forward(wrapper, prob::GameProblem{T}, x0) -> ForwardSolution{T}
    predict_next_state(wrapper, prob::GameProblem{T}, x0) -> Vector{T}

# Notes
Wrapper state (cached solutions, warm-start trajectories) belongs in the
wrapper struct itself, which is held by the mutable `InverseSolverState`.
The `ForwardSolverWrapper` stored in `InverseGameProblem` is immutable
configuration only (options, parameters). Mutable cache lives in the solver state.
"""
abstract type ForwardSolverWrapper end

function solve_forward end

"""
    predict_next_state(wrapper, prob, x0) -> Vector

Default: call solve_forward, return state at t=1.
Subtypes should override for warm-starting.
"""
function predict_next_state(
    wrapper::ForwardSolverWrapper,
    prob::GameProblem{T},
    x0::AbstractVector{T}
) where {T}
    sol = solve_forward(wrapper, prob, x0)
    return first_step_state(sol)
end

# ============================================================================
# Inverse Game Problem — Pure Specification (Immutable)
# ============================================================================

"""
    InverseGameProblem{T}

Immutable specification of an inverse game problem.

Stores only what defines the problem structure — dynamics, constraints,
observation model, knowledge tags, and forward solver configuration.
Contains *no* mutable solver state; all state (observations, ensemble,
cached solutions) belongs in `InverseSolverState`, owned by the solver.

This separation mirrors the DifferentialEquations.jl pattern: `ODEProblem`
is a pure specification, the integrator owns all mutable state.

# Fields
- `n_players::Int`
- `player_specs::Vector{PlayerSpec{T}}` : Dynamics, dims, constraints per player
- `knowledge::Vector{PlayerKnowledge}` : Per-player knowledge tags
- `shared_constraints::AbstractVector`
- `observation_model::ObservationModel`
- `forward_solver::ForwardSolverWrapper` : Solver configuration (immutable options only)
- `time_horizon::TimeHorizon{T}`
- `metadata::GameMetadata` : Cached structural info; reused by `as_forward_problem`

# Design Note: why no observations field?
Observations are solver state, not problem specification. The same
`InverseGameProblem` can be solved offline (batch), online (MONGOOSE), or
re-solved from a different initial belief — all without mutating the problem.
"""
struct InverseGameProblem{T}
    n_players::Int
    player_specs::Vector{PlayerSpec{T}}
    knowledge::Vector{PlayerKnowledge}            
    shared_constraints::AbstractVector
    observation_model::ObservationModel
    forward_solver::ForwardSolverWrapper
    time_horizon::TimeHorizon{T}
    metadata::GameMetadata
 
    function InverseGameProblem{T}(
        n_players::Int,
        player_specs::Vector{PlayerSpec{T}},
        knowledge::AbstractVector{<:PlayerKnowledge},  
        shared_constraints::AbstractVector,
        observation_model::ObservationModel,
        forward_solver::ForwardSolverWrapper,
        time_horizon::TimeHorizon{T},
        metadata::GameMetadata
    ) where {T}
        @assert n_players > 0 "Must have at least one player"
        @assert length(player_specs) == n_players "Must have a PlayerSpec per player"
        @assert length(knowledge) == n_players "Must have a knowledge tag per player"
        @assert allunique(s.id for s in player_specs) "Player IDs must be unique"
        @assert any(k isa UnknownObjective for k in knowledge) "At least one player must have an unknown objective"
 
        for (i, (spec, know)) in enumerate(zip(player_specs, knowledge))
            if know isa KnownObjective
                @assert know.objective.player_id == spec.id "KnownObjective player_id must match PlayerSpec id for player $i"
            end
        end
 
        new{T}(
            n_players, player_specs,
            Vector{PlayerKnowledge}(knowledge),    # ← concretize on store
            shared_constraints, observation_model, forward_solver,
            time_horizon, metadata
        )
    end
end

# ============================================================================
# Accessors
# ============================================================================

unknown_players(prob::InverseGameProblem) =
    [i for (i, k) in enumerate(prob.knowledge) if k isa UnknownObjective]

known_players(prob::InverseGameProblem) =
    [i for (i, k) in enumerate(prob.knowledge) if k isa KnownObjective]

n_unknown(prob::InverseGameProblem) =
    count(k isa UnknownObjective for k in prob.knowledge)

function known_objective(prob::InverseGameProblem, i::Int)
    k = prob.knowledge[i]
    k isa KnownObjective || error("Player $i has an unknown objective")
    return k.objective
end

"""
    as_forward_problem(prob, hypothesized) -> GameProblem{T}

Reconstruct a forward `GameProblem{T}` by substituting hypothesized objectives
for unknown players. Reuses `prob.metadata` directly — O(n_players) cost only.

Called N_ensemble times per EnKF timestep; must be allocation-efficient.
"""
function as_forward_problem(
    prob::InverseGameProblem{T},
    hypothesized::Dict{Int, <:PlayerObjective}
) where {T}
    @assert Set(keys(hypothesized)) == Set(unknown_players(prob)) "Must supply objectives for all unknown players"

    objectives = map(1:prob.n_players) do i
        prob.knowledge[i] isa KnownObjective ? prob.knowledge[i].objective : hypothesized[i]
    end

    player_dynamics = [s.dynamics for s in prob.player_specs]
    state_dims   = prob.metadata.state_dims
    control_dims = prob.metadata.control_dims
    dynamics = SeparableDynamics(player_dynamics, state_dims, control_dims)
    initial_state = vcat([s.x0 for s in prob.player_specs]...)
    private_constraints = Vector{Any}(vcat([s.constraints for s in prob.player_specs]...))

    return GameProblem{T}(
        prob.n_players,
        objectives,
        dynamics,
        initial_state,
        private_constraints,
        prob.shared_constraints,
        prob.time_horizon,
        prob.metadata          # ← reused directly; zero metadata recomputation
    )
end

# ============================================================================
# Mutable Solver State (owned by solver, not by problem)
# ============================================================================

"""
    ObservationData{T}

Trajectory observation log. Owned by `InverseSolverState`, not the problem.

# Fields
- `states::Vector{Vector{T}}` : Observed joint states {x̃_0, x̃_1, ...}
- `times::Vector{T}` : Corresponding timestamps
"""
mutable struct ObservationData{T}
    states::Vector{Vector{T}}
    times::Vector{T}

    ObservationData{T}() where {T} = new{T}(Vector{Vector{T}}(), Vector{T}())

    function ObservationData{T}(
        states::Vector{Vector{T}},
        times::Vector{T}
    ) where {T}
        @assert length(states) == length(times) "States and times must have equal length"
        new{T}(states, times)
    end
end

function push_observation!(data::ObservationData{T}, x::Vector{T}, t::T) where {T}
    push!(data.states, copy(x))   # copy: avoid aliasing with external buffers
    push!(data.times, t)
    return data
end

Base.length(data::ObservationData) = length(data.states)
Base.isempty(data::ObservationData) = isempty(data.states)

"""
    InverseSolverState{T}

Mutable solver state for inverse game inference. Owned exclusively by the
running solver; never stored in `InverseGameProblem`.

Concrete solver implementations (MONGOOSE, batch IRL, etc.) should subtype
this to add solver-specific fields (ensemble, belief weights, STLS buffer, etc.).

# Fields (base)
- `observations::ObservationData{T}` : Accumulated trajectory data
- `t_current::T` : Current simulation time
"""
mutable struct InverseSolverState{T}
    observations::ObservationData{T}
    t_current::T

    InverseSolverState{T}() where {T} = new{T}(ObservationData{T}(), zero(T))
end

# ============================================================================
# Specialized Constructor: Inverse PD-GNEP
# ============================================================================

"""
    InversePDGNEProblem(players, knowledge, shared_constraints,
                        observation_model, forward_solver, tf, dt)

Construct an `InverseGameProblem` for a Partially-Decoupled GNEP.
Metadata is computed once here and reused by `as_forward_problem`.

# Example
```julia
prob = InversePDGNEProblem(
    [chief_spec, deputy_spec],
    [KnownObjective(chief_obj), UnknownObjective()],
    [collision_constraint],
    NoisyObservation(h_cwh, R_meas, 4),
    iLQGamesWrapper(opts),
    400.0, 1.0
)
```
"""
function InversePDGNEProblem(
    players::Vector{PlayerSpec{T}},
    knowledge::AbstractVector{<:PlayerKnowledge},   # ← was Vector{PlayerKnowledge}
    shared_constraints::AbstractVector,
    observation_model::ObservationModel,
    forward_solver::ForwardSolverWrapper,
    tf::T,
    dt::T
) where {T}
    n_players = length(players)
    @assert length(knowledge) == n_players "knowledge must have one entry per player"
 
    # Concretize to Vector{PlayerKnowledge} so dispatch is unambiguous downstream
    knowledge_vec = Vector{PlayerKnowledge}(knowledge)
 
    state_dims      = [p.n for p in players]
    control_dims    = [p.m for p in players]
    state_offsets   = [0; cumsum(state_dims)[1:end-1]]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
 
    constraint_coupling = Vector{Int}[c.players for c in shared_constraints]
    cost_coupling = sparse(trues(n_players, n_players))
    coupling_graph = CouplingGraph(cost_coupling, constraint_coupling, nothing)
 
    metadata = GameMetadata(
        state_dims, control_dims,
        state_offsets, control_offsets,
        coupling_graph,
        false, nothing
    )
 
    return InverseGameProblem{T}(
        n_players, players, knowledge_vec,
        shared_constraints, observation_model, forward_solver,
        DiscreteTime(tf, dt), metadata
    )
end
 
# Convenience: no shared constraints
InversePDGNEProblem(
    players::Vector{PlayerSpec{T}},
    knowledge::AbstractVector{<:PlayerKnowledge},   # ← same widening
    observation_model::ObservationModel,
    forward_solver::ForwardSolverWrapper,
    tf::T,
    dt::T
) where {T} = InversePDGNEProblem(
    players, knowledge, [],
    observation_model, forward_solver,
    tf, dt
)

