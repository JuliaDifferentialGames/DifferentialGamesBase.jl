# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractGame{T}

Base type for all game-theoretic problems.

# Type Parameters
- `T` : Numeric type (Float64, Float32, etc.)

# Notes
All games involve:
- Multiple decision-makers (players)
- Strategic interaction (Nash equilibrium concept)
- Coupled or decoupled objectives
- Possible constraints on strategy spaces
"""
abstract type AbstractGame{T} end

# ============================================================================
# Dynamics Specifications
# ============================================================================

"""
    DynamicsSpec{T}

Abstract base type for dynamics specifications.

Encapsulates how system state evolves over time given control inputs.
"""
abstract type DynamicsSpec{T} end

"""
    SeparableDynamics{T, F} <: DynamicsSpec{T}

Separable dynamics where each player's state evolution depends only on their own state and control.

# Mathematical Form
ẋᵢ(t) = fᵢ(xᵢ(t), uᵢ(t), p, t) for each player i

# Fields
- `player_dynamics::Vector{F}` : Per-player dynamics functions [f₁, f₂, ..., fₙ]
- `state_dims::Vector{Int}` : State dimensions [n₁, n₂, ..., nₙ]
- `control_dims::Vector{Int}` : Control dimensions [m₁, m₂, ..., mₙ]

# Properties
- Enables parallel state propagation
- Block-diagonal dynamics Jacobian structure
- Characteristic of PD-GNEPs (Partially-Decoupled GNEPs)

# Notes
For stacked state x = [x₁; x₂; ...; xₙ], the full dynamics Jacobian ∂f/∂x
is block-diagonal, with each block corresponding to ∂fᵢ/∂xᵢ.
"""
struct SeparableDynamics{T, F} <: DynamicsSpec{T}
    player_dynamics::Vector{F}
    state_dims::Vector{Int}
    control_dims::Vector{Int}
    
    function SeparableDynamics(
        player_dynamics::Vector{F},
        state_dims::Vector{Int},
        control_dims::Vector{Int}
    ) where {F}
        n_players = length(player_dynamics)
        @assert length(state_dims) == n_players "Must have state_dims for each player"
        @assert length(control_dims) == n_players "Must have control_dims for each player"
        @assert all(state_dims .> 0) "State dimensions must be positive"
        @assert all(control_dims .> 0) "Control dimensions must be positive"
        
        T = Float64  # Default numeric type
        new{T, F}(player_dynamics, state_dims, control_dims)
    end
end

"""
    LinearDynamics{T} <: DynamicsSpec{T}

Linear dynamics with shared state space and per-player control matrices.

# Mathematical Form
ẋ(t) = A x(t) + Σᵢ Bᵢ uᵢ(t)

# Fields
- `A::Matrix{T}` : System dynamics matrix (n × n)
- `B::Vector{Matrix{T}}` : Control matrices [B₁, B₂, ..., Bₙₚ], each (n × mᵢ)
- `state_dim::Int` : Shared state dimension n
- `control_dims::Vector{Int}` : Control dimensions [m₁, m₂, ..., mₙₚ]

# Properties
- All players operate in same state space
- Control inputs enter linearly
- Standard formulation for LQ games (Başar & Olsder, 1998)

# Notes
This is the coupled dynamics formulation. For separable linear dynamics,
use SeparableDynamics with linear functions.
"""
struct LinearDynamics{T} <: DynamicsSpec{T}
    A::Matrix{T}
    B::Vector{Matrix{T}}
    state_dim::Int
    control_dims::Vector{Int}
    
    function LinearDynamics(A::Matrix{T}, B::Vector{Matrix{T}}) where {T}
        n = size(A, 1)
        @assert size(A) == (n, n) "A must be square"
        
        n_players = length(B)
        control_dims = [size(Bi, 2) for Bi in B]
        
        for (i, Bi) in enumerate(B)
            @assert size(Bi, 1) == n "B[$i] must have $n rows to match state dimension"
        end
        
        new{T}(A, B, n, control_dims)
    end
end

"""
    CoupledNonlinearDynamics{T, F} <: DynamicsSpec{T}

General nonlinear coupled dynamics where state evolution can depend on all states and controls.

# Mathematical Form
ẋ(t) = f(x(t), u(t), p, t)

# Fields
- `func::F` : Dynamics function f(x, u, p, t) -> ẋ
- `state_dim::Int` : Total state dimension
- `control_dim::Int` : Total control dimension
- `jacobian::Union{Nothing, Function}` : Optional analytical Jacobian (∂f/∂x, ∂f/∂u)

# Notes
Most general dynamics formulation. Use when:
- Dynamics cannot be separated by player
- Nonlinear coupling exists between state components
- No special structure to exploit
"""
struct CoupledNonlinearDynamics{T, F} <: DynamicsSpec{T}
    func::F
    state_dim::Int
    control_dim::Int
    jacobian::Union{Nothing, Function}
    
    function CoupledNonlinearDynamics(
        func::F,
        state_dim::Int,
        control_dim::Int;
        jacobian::Union{Nothing, Function} = nothing
    ) where {F}
        @assert state_dim > 0 "State dimension must be positive"
        @assert control_dim > 0 "Control dimension must be positive"
        
        T = Float64
        new{T, F}(func, state_dim, control_dim, jacobian)
    end
end

# ============================================================================
# Time Horizon Specifications
# ============================================================================

"""
    TimeHorizon{T}

Abstract base type for time horizon specifications.
"""
abstract type TimeHorizon{T} end

"""
    ContinuousTime{T} <: TimeHorizon{T}

Continuous-time horizon for differential games.

# Fields
- `tf::T` : Final time
- `integrator_type::Symbol` : ODE integration scheme (:rk4, :tsit5, :euler, etc.)

# Notes
Solver performs ODE integration to propagate dynamics.
Choice of integrator affects accuracy and computational cost.
"""
struct ContinuousTime{T} <: TimeHorizon{T}
    tf::T
    integrator_type::Symbol
    
    function ContinuousTime(tf::T; integrator_type::Symbol = :rk4) where {T}
        @assert tf > 0 "Time horizon must be positive"
        @assert integrator_type in (:euler, :rk4, :tsit5, :radau) "Unknown integrator type"
        new{T}(tf, integrator_type)
    end
end

"""
    DiscreteTime{T} <: TimeHorizon{T}

Discrete-time horizon with fixed time step.

# Fields
- `tf::T` : Final time
- `dt::T` : Time step
- `N::Int` : Number of time steps (computed as ceil(tf/dt))

# Notes
Standard for direct transcription methods.
State and control discretized at times [0, dt, 2dt, ..., N*dt].
"""
struct DiscreteTime{T} <: TimeHorizon{T}
    tf::T
    dt::T
    N::Int
    
    function DiscreteTime(tf::T, dt::T) where {T}
        @assert tf > 0 "Time horizon must be positive"
        @assert dt > 0 "Time step must be positive"
        @assert dt < tf "Time step must be less than time horizon"
        
        N = Int(ceil(tf / dt))
        new{T}(tf, dt, N)
    end
end

# ============================================================================
# Game Metadata
# ============================================================================

"""
    CouplingGraph

Encodes coupling structure between players for solver exploitation.

# Fields
- `cost_coupling::SparseMatrixCSC{Bool}` : cost_coupling[i,j] = true if Jᵢ depends on uⱼ
- `constraint_coupling::Vector{Vector{Int}}` : Players involved in each shared constraint
- `dynamics_coupling::Union{Nothing, SparseMatrixCSC{Bool}}` : For coupled dynamics

# Notes
Sparse coupling enables:
- Parallel computation of independent subproblems
- Reduced communication in distributed algorithms
- Improved convergence rates for iterative methods
"""
struct CouplingGraph
    cost_coupling::SparseMatrixCSC{Bool}
    constraint_coupling::Vector{Vector{Int}}
    dynamics_coupling::Union{Nothing, SparseMatrixCSC{Bool}}
end

"""
    GameMetadata

Cached information about game structure for solver efficiency.

# Fields
- `state_dims::Vector{Int}` : State dimensions per player
- `control_dims::Vector{Int}` : Control dimensions per player
- `state_offsets::Vector{Int}` : Starting indices in stacked state vector
- `control_offsets::Vector{Int}` : Starting indices in stacked control vector
- `coupling_graph::CouplingGraph` : Coupling structure between players
- `is_potential::Bool` : Whether game has potential function structure
- `potential_function::Union{Nothing, Function}` : Potential function if exists

# Notes
Computed once at problem construction, reused by solvers.
"""
struct GameMetadata
    state_dims::Vector{Int}
    control_dims::Vector{Int}
    state_offsets::Vector{Int}
    control_offsets::Vector{Int}
    coupling_graph::CouplingGraph
    is_potential::Bool
    potential_function::Union{Nothing, Function}
end

# ============================================================================
# Universal Game Problem Container
# ============================================================================

"""
    GameProblem{T}

Universal game problem representation supporting arbitrary structure.

# Type Parameters
- `T` : Numeric type

# Fields
- `n_players::Int` : Number of players
- `objectives::Vector{PlayerObjective}` : Per-player objective functions
- `dynamics::DynamicsSpec{T}` : System dynamics specification
- `initial_state::Vector{T}` : Initial state x(0)
- `private_constraints::Vector{PrivateConstraint}` : Player-specific constraints
- `shared_constraints::Vector{SharedConstraint}` : Constraints coupling multiple players
- `time_horizon::TimeHorizon{T}` : Time specification
- `metadata::GameMetadata` : Cached structural information

# Mathematical Formulation
For each player i ∈ {1, ..., N}, the game seeks a Nash equilibrium where:

    min   Jᵢ(x(·), uᵢ(·))
    uᵢ(·)

subject to:
    ẋ(t) = f(x(t), u(t), p, t),  x(0) = x₀
    gᵢ(xᵢ, uᵢ, p, t) ≤ 0          [private constraints]
    h(x, u, p, t) ≤ 0              [shared constraints]

A strategy profile u* = (u₁*, ..., uₙ*) is a Nash equilibrium if for all i:
    Jᵢ(x(u*), uᵢ*) ≤ Jᵢ(x(uᵢ, u₋ᵢ*), uᵢ)  ∀ feasible uᵢ

# Property Queries
Use query functions to determine game structure:
- `has_separable_dynamics(game)` : Check if dynamics are decoupled
- `is_lq_game(game)` : Check if costs are quadratic and dynamics linear
- `is_potential_game(game)` : Check if potential function exists
- `has_shared_constraints(game)` : Check for constraint coupling

# Example
```julia
# Build game from components
game = GameProblem{Float64}(
    n_players,
    objectives,
    dynamics,
    x0,
    private_constraints,
    shared_constraints,
    time_horizon,
    metadata
)

# Solve with automatic method selection
solution = solve(game)
```
"""
struct GameProblem{T}
    n_players::Int
    objectives::Vector{<:PlayerObjective}
    dynamics::DynamicsSpec{T}
    initial_state::Vector{T}
    private_constraints::AbstractVector
    shared_constraints::AbstractVector
    time_horizon::TimeHorizon{T}
    metadata::GameMetadata
    
    function GameProblem{T}(
        n_players::Int,
        objectives::Vector{<:PlayerObjective},
        dynamics::DynamicsSpec{T},
        initial_state::Vector{T},
        private_constraints::AbstractVector,
        shared_constraints::AbstractVector,
        time_horizon::TimeHorizon{T},
        metadata::GameMetadata
    ) where {T}
        @assert n_players > 0 "Must have at least one player"
        @assert length(objectives) == n_players "Must have objective for each player"
        @assert all(obj.player_id > 0 && obj.player_id <= n_players for obj in objectives) "Invalid player IDs in objectives"
        @assert allunique(obj.player_id for obj in objectives) "Duplicate player IDs in objectives"
        
        # Validate constraint player references
        all_player_ids = Set(1:n_players)
        for c in private_constraints
            @assert c.player in all_player_ids "Private constraint references invalid player"
        end
        for c in shared_constraints
            @assert all(p in all_player_ids for p in c.players) "Shared constraint references invalid player"
        end
        
        new{T}(n_players, objectives, dynamics, initial_state,
               private_constraints, shared_constraints, time_horizon, metadata)
    end
end

# ============================================================================
# Property Query Functions
# ============================================================================

"""
    has_separable_dynamics(game::GameProblem)

Check if game has separable (decoupled) dynamics structure.

Returns true for PD-GNEPs where ẋᵢ = fᵢ(xᵢ, uᵢ).
"""
has_separable_dynamics(game::GameProblem) = game.dynamics isa SeparableDynamics

"""
    is_lq_game(game::GameProblem)

Check if all objectives are linear-quadratic and dynamics are linear.

Returns true if this is a classical LQ game solvable via Riccati equations.
"""
function is_lq_game(game::GameProblem)
    # Check dynamics are linear
    if !(game.dynamics isa LinearDynamics)
        return false
    end
    
    # Check all objectives have LQ stage and terminal costs
    for obj in game.objectives
        if !(obj.stage_cost isa Union{LQStageCost, DiagonalLQStageCost})
            return false
        end
        if !(obj.terminal_cost isa Union{LQTerminalCost, DiagonalLQTerminalCost})
            return false
        end
    end
    
    return true
end

"""
    is_potential_game(game::GameProblem)

Check if game has potential function structure.

Returns true if stored in metadata (must be computed/verified at construction).
"""
is_potential_game(game::GameProblem) = game.metadata.is_potential

"""
    has_shared_constraints(game::GameProblem)

Check if game has constraints coupling multiple players.
"""
has_shared_constraints(game::GameProblem) = !isempty(game.shared_constraints)

"""
    is_unconstrained(game::GameProblem)

Check if game has no constraints of any kind.
"""
is_unconstrained(game::GameProblem) = 
    isempty(game.private_constraints) && isempty(game.shared_constraints)

"""
    is_pd_gnep(game::GameProblem)

Check if game is a Partially-Decoupled GNEP (separable dynamics).
"""
is_pd_gnep(game::GameProblem) = has_separable_dynamics(game)

"""
    is_lq_pd_gnep(game::GameProblem)

Check if game is an LQ game with separable dynamics.
"""
is_lq_pd_gnep(game::GameProblem) = is_lq_game(game) && is_pd_gnep(game)

# ============================================================================
# Helper Functions
# ============================================================================

"""
    num_players(game::GameProblem)

Get number of players in game.
"""
num_players(game::GameProblem) = game.n_players

"""
    control_dim(game::GameProblem)

Get total control dimension (sum across all players).
"""
control_dim(game::GameProblem) = sum(game.metadata.control_dims)

"""
    state_dim(game::GameProblem)

Get total state dimension (sum across all players for separable, or single dimension for shared).
"""
function state_dim(game::GameProblem)
    if has_separable_dynamics(game)
        return sum(game.metadata.state_dims)
    else
        # Shared state space
        return game.metadata.state_dims[1]
    end
end

"""
    state_dim(game::GameProblem, player::Int)

Get state dimension for specific player.
"""
state_dim(game::GameProblem, player::Int) = game.metadata.state_dims[player]

"""
    control_dim(game::GameProblem, player::Int)

Get control dimension for specific player.
"""
control_dim(game::GameProblem, player::Int) = game.metadata.control_dims[player]

"""
    get_objective(game::GameProblem, player::Int)

Get objective function for specific player.
"""
function get_objective(game::GameProblem, player::Int)
    idx = findfirst(obj -> obj.player_id == player, game.objectives)
    isnothing(idx) && error("No objective found for player $player")
    return game.objectives[idx]
end

# ============================================================================
# Specialized Constructors
# ============================================================================

"""
    PlayerSpec{T}

Specification for a single player in a PD-GNEP.

# Fields
- `id::Int` : Player identifier
- `n::Int` : State dimension
- `m::Int` : Control dimension
- `x0::Vector{T}` : Initial state
- `dynamics::Function` : Dynamics function fᵢ(xᵢ, uᵢ, p, t)
- `objective::PlayerObjective` : Cost functional
- `constraints::Vector{PrivateConstraint}` : Player-specific constraints

# Notes
Used to build PD-GNEPs where each player has separable dynamics.
"""
struct PlayerSpec{T}
    id::Int
    n::Int
    m::Int
    x0::Vector{T}
    dynamics::Function
    objective::PlayerObjective
    constraints::AbstractVector
    
    # Inner constructor
    function PlayerSpec{T}(
        id::Int,
        n::Int,
        m::Int,
        x0::Vector{T},
        dynamics::Function,
        objective::PlayerObjective,
        constraints::AbstractVector 
    ) where {T}
        @assert id > 0 "Player ID must be positive"
        @assert n > 0 "State dimension must be positive"
        @assert m > 0 "Control dimension must be positive"
        @assert length(x0) == n "Initial state must match state dimension"
        @assert objective.player_id == id "Objective player_id must match PlayerSpec id"
        
        new{T}(id, n, m, x0, dynamics, objective, Vector{Any}(constraints))
    end
end

PlayerSpec(
    id::Int,
    n::Int,
    m::Int,
    x0::Vector{T},
    dynamics::Function,
    objective::PlayerObjective
) where {T} = PlayerSpec{T}(id, n, m, x0, dynamics, objective, [])

# 7-argument version (with constraints)
PlayerSpec(
    id::Int,
    n::Int,
    m::Int,
    x0::Vector{T},
    dynamics::Function,
    objective::PlayerObjective,
    constraints::AbstractVector
) where {T} = PlayerSpec{T}(id, n, m, x0, dynamics, objective, constraints)

"""
    PDGNEProblem(players, shared_constraints, tf, dt)

Construct a Partially-Decoupled GNEP from player specifications.

# Arguments
- `players::Vector{PlayerSpec{T}}` : Player specifications with separable dynamics
- `shared_constraints::Vector{SharedConstraint}` : Constraints coupling multiple players
- `tf::T` : Final time
- `dt::T` : Time step for discretization

# Returns
`GameProblem{T}` with separable dynamics structure.

# Example
```julia
player1 = PlayerSpec(1, 6, 3, x0_1, dynamics_1, objective_1)
player2 = PlayerSpec(2, 6, 3, x0_2, dynamics_2, objective_2)

collision = SharedConstraint(collision_constraint, [1, 2])

game = PDGNEProblem([player1, player2], [collision], 10.0, 0.1)
```
"""
function PDGNEProblem(
    players::Vector{PlayerSpec{T}},
    shared_constraints::AbstractVector,
    tf::T,
    dt::T
) where {T}
    n_players = length(players)
    @assert n_players > 0 "Must have at least one player"
    @assert allunique(p.id for p in players) "Player IDs must be unique"
    
    # Extract components
    objectives = [p.objective for p in players]
    state_dims = [p.n for p in players]
    control_dims = [p.m for p in players]
    
    # Build separable dynamics
    player_dynamics = [p.dynamics for p in players]
    dynamics = SeparableDynamics(player_dynamics, state_dims, control_dims)
    
    # Stack initial states
    initial_state = vcat([p.x0 for p in players]...)

    # Convert constraint vectors to Vector{Any}
    shared_constraints_any = Vector{Any}(shared_constraints)
    
    # Collect private constraints - convert to Vector{Any}
    private_constraints = Vector{Any}(vcat([p.constraints for p in players]...))
    
    # Time horizon
    time_horizon = DiscreteTime(tf, dt)
    
    # Build metadata
    state_offsets = [0; cumsum(state_dims)[1:end-1]]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    
    # Build coupling graph
    cost_coupling = sparse(trues(n_players, n_players))
    for (i, obj) in enumerate(objectives)
        if is_separable(obj.stage_cost)
            cost_coupling[i, i] = true
        else
            # Conservative: assume full coupling
            cost_coupling[i, :] .= true
        end
    end
    
    # Extract constraint coupling - be explicit about types
    constraint_coupling = Vector{Int}[c.players for c in shared_constraints]
    coupling_graph = CouplingGraph(cost_coupling, constraint_coupling, nothing)
    
    metadata = GameMetadata(
        state_dims,
        control_dims,
        state_offsets,
        control_offsets,
        coupling_graph,
        false,  # is_potential
        nothing  # potential_function
    )
    
    return GameProblem{T}(
        n_players,
        objectives,
        dynamics,
        initial_state,
        private_constraints,
        shared_constraints,
        time_horizon,
        metadata
    )
end

# Convenience constructor without shared constraints
PDGNEProblem(players::Vector{PlayerSpec{T}}, tf::T, dt::T) where {T} =
    PDGNEProblem(players, SharedConstraint[], tf, dt)

"""
    LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=0.01, M=nothing, q=nothing, r=nothing)

Construct a linear-quadratic differential game from matrix data.

# Arguments
- `A::Matrix{T}` : System dynamics matrix (n × n)
- `B::Vector{Matrix{T}}` : Control matrices [B₁, ..., Bₙₚ]
- `Q::Vector{Matrix{T}}` : State cost matrices [Q₁, ..., Qₙₚ]
- `R::Vector{Matrix{T}}` : Control cost matrices [R₁, ..., Rₙₚ]
- `Qf::Vector{Matrix{T}}` : Terminal cost matrices [Qf₁, ..., Qfₙₚ]
- `x0::Vector{T}` : Initial state
- `tf::T` : Final time
- `dt::T` : Time step (default: 0.01)
- `M::Union{Vector{Matrix{T}}, Nothing}` : Cross-term matrices (optional)
- `q::Union{Vector{Vector{T}}, Nothing}` : Linear state costs (optional)
- `r::Union{Vector{Vector{T}}, Nothing}` : Linear control costs (optional)

# Returns
`GameProblem{T}` representing classical LQ game.

# Mathematical Form
Dynamics: ẋ = Ax + Σᵢ Bᵢuᵢ
Cost: Jᵢ = ∫[xᵀQᵢx + uᵢᵀRᵢuᵢ + 2xᵀMᵢuᵢ + qᵢᵀx + rᵢᵀuᵢ]dt + x(tf)ᵀQfᵢx(tf)

# Example
```julia
n, n_players = 4, 2
A = randn(n, n)
B = [randn(n, 2) for _ in 1:n_players]
Q = [diagm(ones(n)) for _ in 1:n_players]
R = [diagm(0.1 * ones(2)) for _ in 1:n_players]
Qf = [diagm(10.0 * ones(n)) for _ in 1:n_players]

game = LQGameProblem(A, B, Q, R, Qf, zeros(n), 10.0)
```
"""
function LQGameProblem(
    A::Matrix{T},
    B::Vector{Matrix{T}},
    Q::Vector{Matrix{T}},
    R::Vector{Matrix{T}},
    Qf::Vector{Matrix{T}},
    x0::Vector{T},
    tf::T;
    dt::T = T(0.01),
    M::Union{Vector{Matrix{T}}, Nothing} = nothing,
    q::Union{Vector{Vector{T}}, Nothing} = nothing,
    r::Union{Vector{Vector{T}}, Nothing} = nothing
) where {T}
    n = size(A, 1)
    n_players = length(B)
    
    @assert length(Q) == n_players "Must have Q for each player"
    @assert length(R) == n_players "Must have R for each player"
    @assert length(Qf) == n_players "Must have Qf for each player"
    @assert length(x0) == n "Initial state must match state dimension"
    
    # Validate LQ structure
    @assert size(A) == (n, n) "A must be n × n"
    for i in 1:n_players
        @assert size(Q[i]) == (n, n) "Q[$i] must be n × n"
        @assert issymmetric(Q[i]) "Q[$i] must be symmetric"
        @assert size(R[i], 1) == size(R[i], 2) "R[$i] must be square"
        @assert issymmetric(R[i]) "R[$i] must be symmetric"
        @assert isposdef(R[i]) "R[$i] must be positive definite"
        @assert size(Qf[i]) == (n, n) "Qf[$i] must be n × n"
        @assert issymmetric(Qf[i]) "Qf[$i] must be symmetric"
    end
    
    # Build dynamics
    control_dims = [size(Bi, 2) for Bi in B]
    dynamics = LinearDynamics(A, B)
    
    # Build objectives
    objectives = PlayerObjective[]
    for i in 1:n_players
        mi = control_dims[i]
        
        # Extract optional components
        Mi = isnothing(M) ? zeros(T, n, mi) : M[i]
        qi = isnothing(q) ? zeros(T, n) : q[i]
        ri = isnothing(r) ? zeros(T, mi) : r[i]
        
        stage_cost = LQStageCost(Q[i], R[i], Mi, qi, ri, zero(T))
        terminal_cost = LQTerminalCost(Qf[i])
        
        push!(objectives, PlayerObjective(i, stage_cost, terminal_cost))
    end
    
    # Time horizon
    time_horizon = DiscreteTime(tf, dt)
    
    # Build metadata
    state_dims = [n]  # Correct: single shared state space
    state_offsets = [0]  # Only one offset needed
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    
    # Coupling graph (LQ games typically have full cost coupling via shared state)
    cost_coupling = sparse(trues(n_players, n_players))  # Dense
    coupling_graph = CouplingGraph(cost_coupling, Vector{Int}[], nothing)
    
    metadata = GameMetadata(
        state_dims,
        control_dims,
        state_offsets,
        control_offsets,
        coupling_graph,
        false,
        nothing
    )
    
    return GameProblem{T}(
        n_players,
        objectives,
        dynamics,
        x0,
        PrivateConstraint[],
        SharedConstraint[],
        time_horizon,
        metadata
    )
end

"""
    UnconstrainedLQGame(A, B, Q, R, Qf, x0, tf; dt=0.01)

Convenience constructor for unconstrained LQ games.

Equivalent to `LQGameProblem` but emphasizes unconstrained nature.
Useful for classical LQ game benchmarks and theoretical analysis.

# Example
```julia
# Two-player pursuit-evasion
n = 4  # [x, y, vx, vy]
A = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
B = [zeros(2,2); I(2)], [zeros(2,2); -I(2)]  # Opposing controls

Q1 = diagm([1.0, 1.0, 0.0, 0.0])  # Pursuer wants small position error
Q2 = diagm([1.0, 1.0, 0.0, 0.0])  # Evader wants large position error (minimizes -Q2)

R1 = 0.1 * I(2)
R2 = 0.1 * I(2)

game = UnconstrainedLQGame(A, [B...], [Q1, Q2], [R1, R2], [Q1, Q2], zeros(4), 5.0)
```
"""
UnconstrainedLQGame(
    A::Matrix{T},
    B::Vector{Matrix{T}},
    Q::Vector{Matrix{T}},
    R::Vector{Matrix{T}},
    Qf::Vector{Matrix{T}},
    x0::Vector{T},
    tf::T;
    dt::T = T(0.01)
) where {T} = LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=dt)

# ============================================================================
# Display Methods
# ============================================================================

function Base.show(io::IO, game::GameProblem{T}) where {T}
    print(io, "GameProblem{$T} with $(game.n_players) players")
    
    # Add structure tags
    tags = String[]
    is_lq_game(game) && push!(tags, "LQ")
    is_pd_gnep(game) && push!(tags, "PD-GNEP")
    is_potential_game(game) && push!(tags, "Potential")
    is_unconstrained(game) && push!(tags, "Unconstrained")
    
    if !isempty(tags)
        print(io, " [", join(tags, ", "), "]")
    end
end

function Base.show(io::IO, ::MIME"text/plain", game::GameProblem{T}) where {T}
    println(io, "GameProblem{$T}")
    println(io, "  Players: ", game.n_players)
    println(io, "  State dimension: ", state_dim(game))
    println(io, "  Control dimension: ", control_dim(game))
    println(io, "  Dynamics: ", typeof(game.dynamics))
    println(io, "  Time horizon: ", game.time_horizon)
    println(io, "  Private constraints: ", length(game.private_constraints))
    println(io, "  Shared constraints: ", length(game.shared_constraints))
    
    println(io, "  Properties:")
    println(io, "    - LQ game: ", is_lq_game(game))
    println(io, "    - PD-GNEP: ", is_pd_gnep(game))
    println(io, "    - Potential game: ", is_potential_game(game))
    println(io, "    - Unconstrained: ", is_unconstrained(game))
end