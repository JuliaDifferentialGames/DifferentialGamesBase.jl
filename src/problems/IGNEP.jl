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

"""
    observe(model::ObservationModel, x::AbstractVector) -> AbstractVector

Apply the observation model to joint state `x`, returning the observable vector.
"""
function observe end

"""
    observation_dim(model::ObservationModel) -> Int

Return the dimension of the observation vector produced by `model`.
"""
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

"""
    solve_forward(wrapper::ForwardSolverWrapper, prob::GameProblem, x0) -> GNEPSolution

Solve the forward game problem from initial state `x0` and return the full solution.
Must be implemented by each `ForwardSolverWrapper` subtype.
"""
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

"""
    unknown_players(prob::InverseGameProblem) -> Vector{Int}

Return the player indices whose objectives are unknown (to be inferred).
"""
unknown_players(prob::InverseGameProblem) =
    [i for (i, k) in enumerate(prob.knowledge) if k isa UnknownObjective]

"""
    known_players(prob::InverseGameProblem) -> Vector{Int}

Return the player indices whose objectives are known.
"""
known_players(prob::InverseGameProblem) =
    [i for (i, k) in enumerate(prob.knowledge) if k isa KnownObjective]

"""
    n_unknown(prob::InverseGameProblem) -> Int

Return the number of players with unknown objectives.
"""
n_unknown(prob::InverseGameProblem) =
    count(k isa UnknownObjective for k in prob.knowledge)

"""
    known_objective(prob::InverseGameProblem, i::Int) -> PlayerObjective

Return the known objective for player `i`. Throws if player `i` is not a known player.
"""
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

"""
    push_observation!(data::ObservationData{T}, x::Vector{T}, t::T) -> ObservationData

Append a new joint state observation `x` at time `t` to `data`.
"""
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

# ============================================================================
# InverseLQGame — Algebraic Inverse LQ Differential Game
# ============================================================================

"""
    InverseLQGame{T} <: AbstractInverseGameProblem{T}

Specification of an inverse infinite-horizon linear-quadratic (LQ) game.

Given system dynamics
    ẋ(t) = Ax(t) + Σᵢ Bᵢuᵢ(t)
and a Nash equilibrium characterised by linear feedback laws uᵢ(t) = −Kᵢx(t),
find **all** cost-function parameter vectors
    θᵢ = [vec(Qᵢ)ᵀ  vec(Rᵢ₁)ᵀ  ⋯  vec(RᵢN)ᵀ]ᵀ
consistent with that equilibrium (Theorem 2, Inga et al. 2019).

# Fields
- `n_players` : N — number of players
- `A`          : joint state matrix (n × n)
- `B`          : per-player input matrices; `B[i]` is n × pᵢ
- `K_star`     : Nash feedback matrices (pᵢ × n each), or `nothing` when
                 trajectories must be used for estimation
- `state_trajectories`   : observed joint-state trajectory (n × K_samples),
                           or `nothing` when `K_star` is given directly
- `control_trajectories` : observed control trajectories; `control_trajectories[i]`
                           is pᵢ × K_samples, or `nothing`
- `sample_times`         : time stamps for trajectory samples, or `nothing`
- `state_dim`    : n (cached for convenience)
- `control_dims` : [p₁, …, pN] (cached for convenience)

# Constructors
    InverseLQGame(A, B, K_star)                              # exact K* given
    InverseLQGame(A, B, X_traj, U_traj; times=nothing)       # trajectory mode

# References
Inga, J., Bischoff, E., Molloy, T.L., Flad, M., Hohmann, S. (2019).
Solution sets for inverse non-cooperative linear-quadratic differential games.
*IEEE Control Systems Letters*, 3(4), 871–876. DOI: 10.1109/LCSYS.2019.2919271
"""
struct InverseLQGame{T} <: AbstractInverseGameProblem{T}
    n_players           ::Int
    A                   ::Matrix{T}
    B                   ::Vector{Matrix{T}}
    K_star              ::Union{Nothing, Vector{Matrix{T}}}
    state_trajectories  ::Union{Nothing, Matrix{T}}
    control_trajectories::Union{Nothing, Vector{Matrix{T}}}
    sample_times        ::Union{Nothing, Vector{T}}
    state_dim           ::Int
    control_dims        ::Vector{Int}

    function InverseLQGame{T}(
        n_players           ::Int,
        A                   ::Matrix{T},
        B                   ::Vector{Matrix{T}},
        K_star              ::Union{Nothing, Vector{Matrix{T}}},
        state_trajectories  ::Union{Nothing, Matrix{T}},
        control_trajectories::Union{Nothing, Vector{Matrix{T}}},
        sample_times        ::Union{Nothing, Vector{T}}
    ) where {T}
        n = size(A, 1)
        @assert size(A, 2) == n                 "A must be square (n×n)"
        @assert n_players == length(B)          "Must have one B matrix per player"
        @assert all(size(Bi, 1) == n for Bi in B) "Each Bᵢ must have n rows"

        control_dims = [size(Bi, 2) for Bi in B]

        if K_star !== nothing
            @assert length(K_star) == n_players "K_star must have one matrix per player"
            for i in 1:n_players
                @assert(size(K_star[i]) == (control_dims[i], n),
                    "K_star[$i] must be $(control_dims[i])×$n")
            end
        end

        if state_trajectories !== nothing
            @assert size(state_trajectories, 1) == n "state_trajectories must have n rows"
        end
        if control_trajectories !== nothing
            @assert(length(control_trajectories) == n_players,
                "Must have one control trajectory per player")
        end

        new{T}(n_players, A, B, K_star,
               state_trajectories, control_trajectories, sample_times,
               n, control_dims)
    end
end

# ─── Constructors ─────────────────────────────────────────────────────────────

"""
    InverseLQGame(A, B, K_star) -> InverseLQGame{T}

Construct an inverse LQ game with **exactly known** Nash feedback matrices K*.

# Arguments
- `A`      : n×n state matrix
- `B`      : `Vector{Matrix{T}}` — `B[i]` is n×pᵢ
- `K_star` : `Vector{Matrix{T}}` — `K_star[i]` is pᵢ×n (feedback gain for player i)
"""
function InverseLQGame(
    A      ::Matrix{T},
    B      ::Vector{Matrix{T}},
    K_star ::Vector{Matrix{T}}
) where {T}
    InverseLQGame{T}(length(B), A, B, K_star, nothing, nothing, nothing)
end

"""
    InverseLQGame(A, B, X_traj, U_traj; times=nothing) -> InverseLQGame{T}

Construct an inverse LQ game from **observed trajectories**.
The Nash feedback matrices K* will be estimated inside the solver via
least-squares (eq. 22 of Inga et al. 2019).

# Arguments
- `A`       : n×n state matrix
- `B`       : `Vector{Matrix{T}}` — `B[i]` is n×pᵢ
- `X_traj`  : n × K_samples matrix of observed joint states
- `U_traj`  : `Vector{Matrix{T}}` — `U_traj[i]` is pᵢ × K_samples observed controls
- `times`   : optional K_samples-vector of time stamps
"""
function InverseLQGame(
    A      ::Matrix{T},
    B      ::Vector{Matrix{T}},
    X_traj ::Matrix{T},
    U_traj ::Vector{Matrix{T}};
    times  ::Union{Nothing, Vector{T}} = nothing
) where {T}
    InverseLQGame{T}(length(B), A, B, nothing, X_traj, U_traj, times)
end

# ─── Accessors / display ──────────────────────────────────────────────────────

n_players(g::InverseLQGame) = g.n_players

"""
    has_exact_K(prob::InverseLQGame) -> Bool

`true` when Nash feedback matrices are given directly; `false` in trajectory mode.
"""
has_exact_K(prob::InverseLQGame) = prob.K_star !== nothing

"""
    has_trajectory_data(prob::InverseLQGame) -> Bool

`true` when observed trajectory data is available for K* estimation.
"""
has_trajectory_data(prob::InverseLQGame) =
    prob.state_trajectories !== nothing && prob.control_trajectories !== nothing

function Base.show(io::IO, g::InverseLQGame{T}) where {T}
    mode = has_exact_K(g) ? "exact K*" : "trajectory mode"
    print(io, "InverseLQGame{$T} with $(g.n_players) players [$mode, n=$(g.state_dim)]")
end

function Base.show(io::IO, ::MIME"text/plain", g::InverseLQGame{T}) where {T}
    println(io, "InverseLQGame{$T}")
    println(io, "  Players      : ", g.n_players)
    println(io, "  State dim n  : ", g.state_dim)
    println(io, "  Control dims : ", g.control_dims)
    println(io, "  Mode         : ", has_exact_K(g) ? "exact K*" : "trajectory (K* estimated)")
    println(io, "  Parameter L  : ", g.state_dim^2 + sum(p^2 for p in g.control_dims))
end

