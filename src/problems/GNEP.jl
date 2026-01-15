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
# Specialized Constructors
# ============================================================================

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