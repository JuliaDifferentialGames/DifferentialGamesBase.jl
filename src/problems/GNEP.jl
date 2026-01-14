"""
    AbstractGNEP{T} <: DifferentialGame

Abstract base type for a Generalized Nash Equilibrium Problem.

# Type parameters
- `T` : numeric type (e.g., `Float64`, `Float32`)

# Notes
All GNEPs involve:
- Shared dynamics coupling all players
- Per-player cost functionals
- Strategy spaces for each player
- Possible coupled constraints
"""
abstract type AbstractGNEP{T} <: DifferentialGame end

"""
    AbstractPDGNEP{T} <: AbstractGNEP{T}

Abstract base type for a Partially Decoupled Generalized Nash Equilibrium Problem.

# Type parameters
- `T` : numeric type (e.g., `Float64`, `Float32`)

# Notes
All PDGNEPs involve:
- Shared dynamics coupling all players
- Per-player cost functionals
- Strategy spaces for each player
- Possible coupled constraints
"""
abstract type AbstractPDGNEP{T} <: AbstractGNEP{T} end


"""
    AbstractPotentialGame{T} <: AbstractGNEP{T}

Abstract base type for potential games (special subclass of GNEP).

# Type parameters
- `T` : numeric type

# Notes
Potential games have the special property that all players' gradients
derive from a single potential function, enabling gradient-based solution methods.
"""
abstract type AbstractPotentialGame{T} <: AbstractGNEP{T} end

"""
    AbstractLexicographicGame{T} <: AbstractPotentialGame{T}

Abstract base type for lexicographic games.

# Type parameters
- `T` : numeric type

# Notes
Lexicographic games can be reformulated as potential games (Zargham et al.).
Players optimize costs in a hierarchical priority ordering rather than
simultaneously optimizing individual costs.
"""
abstract type AbstractLexicographicGame{T} <: AbstractPotentialGame{T} end

"""
    AbstractLQGame{T} <: AbstractGNEP{T}

Abstract base type for linear-quadratic differential games.

# Type parameters
- `T` : numeric type

# Notes
LQ games have:
- Linear dynamics: ẋ = A(t)x + Σᵢ Bᵢ(t)uᵢ
- Quadratic costs: Jᵢ = ∫(xᵀQᵢx + uᵢᵀRᵢuᵢ)dt + xᵀ(tf)Qfᵢx(tf)
- Analytical solution via coupled Riccati equations
"""
abstract type AbstractLQGame{T} <: AbstractGNEP{T} end

"""
    GNEProblem{T, N, M, NP}

Concrete implementation of a general nonlinear finite-horizon GNEP.

# Type parameters
- `T` : numeric type
- `N` : state dimension
- `M` : total control dimension
- `NP` : number of players

# Fields
- `dynamics::Function` : dynamics function f(t, x, u) -> ẋ
- `costs::Vector{Function}` : cost functionals [J₁, J₂, ..., Jₙₚ]
- `running_costs::Vector{Function}` : running cost functions [L₁(t,x,u₁), ...]
- `terminal_costs::Vector{Function}` : terminal cost functions [Φ₁(x(tf)), ...]
- `constraints::Vector{Function}` : constraint functions [g₁(t,x,u), ...]
- `x0::Vector{T}` : initial state
- `tf::T` : final time
- `control_dims::Vector{Int}` : control dimensions per player
- `constraint_dims::Vector{Int}` : constraint dimensions per player

# Notes
Represents the general GNEP:
- Dynamics: ẋ(t) = f(t, x(t), u(t)), x(0) = x0
- Cost for player i: Jᵢ = ∫₀ᵗᶠ Lᵢ(t,x,uᵢ)dt + Φᵢ(x(tf))
- Constraints: gᵢ(t, x, u) ≤ 0
"""
struct GNEProblem{T, N, M, NP} <: AbstractGNEP{T}
    dynamics::Function               # f(t, x, u) -> ẋ
    running_costs::Vector{Function}  # [L₁(t,x,u₁), L₂(t,x,u₂), ...]
    terminal_costs::Vector{Function} # [Φ₁(x), Φ₂(x), ...]
    constraints::Vector{Function}    # [g₁(t,x,u), g₂(t,x,u), ...]
    x0::Vector{T}
    tf::T
    control_dims::Vector{Int}
    constraint_dims::Vector{Int}
    
    function GNEProblem{T, N, M, NP}(
        dynamics::Function,
        running_costs::Vector{Function},
        terminal_costs::Vector{Function},
        constraints::Vector{Function},
        x0::Vector{T},
        tf::T,
        control_dims::Vector{Int},
        constraint_dims::Vector{Int}
    ) where {T, N, M, NP}
        @assert length(running_costs) == NP "Must have NP running cost functions"
        @assert length(terminal_costs) == NP "Must have NP terminal cost functions"
        @assert length(x0) == N "x0 must have length N"
        @assert length(control_dims) == NP "Must specify control_dims for each player"
        @assert sum(control_dims) == M "Control dimensions must sum to M"
        @assert tf > 0 "Time horizon must be positive"
        
        new{T, N, M, NP}(dynamics, running_costs, terminal_costs, constraints, 
                         x0, tf, control_dims, constraint_dims)
    end
end

# Convenience constructor
function GNEProblem(
    dynamics::Function,
    running_costs::Vector{Function},
    terminal_costs::Vector{Function},
    constraints::Vector{Function},
    x0::Vector{T},
    tf::T,
    control_dims::Vector{Int},
    constraint_dims::Vector{Int} = Int[]
) where T
    N = length(x0)
    M = sum(control_dims)
    NP = length(running_costs)
    return GNEProblem{T, N, M, NP}(dynamics, running_costs, terminal_costs, 
                                    constraints, x0, tf, control_dims, constraint_dims)
end



"""
    PDGNEProblem{T, NP}

Partially-Decoupled Generalized Nash Equilibrium Problem with heterogeneous players.

# Type Parameters
- `T` : Numeric type (Float64, Float32, etc.)
- `NP` : Number of players (encoded at type level for dispatch)

# Fields
- `players::Vector{Player{T}}` : Vector of N players (length NP)
- `shared_constraints::Vector{ConstraintSpec}` : Constraints coupling multiple players
- `tf::T` : Final time
- `dt::T` : Time step for discretization
- `state_dims::Vector{Int}` : State dimensions [n¹, n², ..., nᴺ]
- `control_dims::Vector{Int}` : Control dimensions [m¹, m², ..., mᴺ]
- `total_n::Int` : Sum of all state dimensions
- `total_m::Int` : Sum of all control dimensions
- `state_offsets::Vector{Int}` : Starting indices for each player's state in concatenated vector
- `control_offsets::Vector{Int}` : Starting indices for each player's control in concatenated vector
- `time_steps::Int` : Number of time steps (computed from tf and dt)

# Notes
Represents the PD-GNEP:

For each player i ∈ {1,...,N}, solve:
    min    Jⁱ(X, uⁱ) = ∫₀ᵗᶠ Lⁱ(X, uⁱ, t)dt + Φⁱ(X(tf))
    uⁱ(·)

subject to:
    ẋⁱ(t) = fⁱ(xⁱ(t), uⁱ(t), pⁱ, t)    [separable dynamics]
    xⁱ(0) = x⁰ⁱ
    Cⁱₚᵣᵢᵥ(xⁱ, uⁱ) ≤ 0                  [private constraints]
    Cˢʰᵃʳᵉᵈ(X, U) ≤ 0                   [shared constraints]

where X = [x¹, ..., xᴺ], U = [u¹, ..., uᴺ] are all states and controls.

# Nash Equilibrium Definition
A strategy profile U* = (u¹*, ..., uᴺ*) is a Nash equilibrium if for all i and all 
feasible uⁱ:

    Jⁱ(X(U*), uⁱ*) ≤ Jⁱ(X(uⁱ, U⁻ⁱ*), uⁱ)

where X(U) = {xⁱ(·; uⁱ)}ᵢ₌₁ᴺ are the trajectories generated by controls U under 
separable dynamics, and U⁻ⁱ* = (u¹*, ..., uⁱ⁻¹*, uⁱ⁺¹*, ..., uᴺ*).

# Example
```julia
# Define two heterogeneous players
player1 = Player(
    id=1, n=4, m=2, 
    x0=[1.0, 0.0, 0.0, 0.0], 
    dynamics=(xⁱ, uⁱ, p, t) -> [xⁱ[3], xⁱ[4], uⁱ[1], uⁱ[2]],
    running_cost=(X, uⁱ, p, t) -> sum(abs2, X[1]) + 0.1*sum(abs2, uⁱ)
)

player2 = Player(
    id=2, n=6, m=3, 
    x0=zeros(6), 
    dynamics=(xⁱ, uⁱ, p, t) -> [xⁱ[4:6]; uⁱ],
    running_cost=(X, uⁱ, p, t) -> sum(abs2, X[2]) + 0.1*sum(abs2, uⁱ)
)

# Collision avoidance constraint (couples players 1 and 2)
collision = ConstraintSpec(
    (X, U, p, t) -> [norm(X[1][1:2] - X[2][1:2]) - 0.5],  # maintain 0.5m separation
    INEQUALITY,
    1,
    [1, 2]  # involves players 1 and 2
)

# Create PD-GNEP
game = PDGNEProblem([player1, player2], [collision], 10.0, 0.1)
```
"""
struct PDGNEProblem{T, NP} <: AbstractGNEP{T}
    players::Vector{Player{T}}
    shared_constraints::Vector{ConstraintSpec}
    tf::T
    dt::T
    
    # Cached dimension information
    state_dims::Vector{Int}
    control_dims::Vector{Int}
    total_n::Int
    total_m::Int
    state_offsets::Vector{Int}      # Cumulative offsets for indexing
    control_offsets::Vector{Int}
    time_steps::Int
    
    function PDGNEProblem{T, NP}(
        players::Vector{Player{T}},
        shared_constraints::Vector{ConstraintSpec},
        tf::T,
        dt::T
    ) where {T, NP}
        @assert length(players) == NP "Number of players must match type parameter NP"
        @assert tf > 0 "Time horizon must be positive"
        @assert dt > 0 "Time step must be positive"
        @assert dt < tf "Time step must be less than time horizon"
        @assert allunique(player_id(p) for p in players) "Player IDs must be unique"
        
        # Validate shared constraints reference valid player IDs
        all_ids = Set(player_id(p) for p in players)
        for c in shared_constraints
            @assert all(id ∈ all_ids for id in c.involved_players) "Shared constraint references invalid player ID"
            @assert length(c.involved_players) >= 2 "Shared constraints must involve at least 2 players"
        end
        
        # Compute dimension information
        state_dims = [p.n for p in players]
        control_dims = [p.m for p in players]
        total_n = sum(state_dims)
        total_m = sum(control_dims)
        
        # Compute offsets for concatenated vectors (0-indexed for convenience)
        state_offsets = [0; cumsum(state_dims)[1:end-1]]
        control_offsets = [0; cumsum(control_dims)[1:end-1]]
        
        # Compute number of time steps
        time_steps = Int(ceil(tf / dt))
        
        new{T, NP}(players, shared_constraints, tf, dt,
                   state_dims, control_dims, total_n, total_m,
                   state_offsets, control_offsets, time_steps)
    end
end

# Convenience constructor (infers NP from player count)
function PDGNEProblem(
    players::Vector{Player{T}},
    shared_constraints::Vector{ConstraintSpec},
    tf::T,
    dt::T
) where T
    NP = length(players)
    return PDGNEProblem{T, NP}(players, shared_constraints, tf, dt)
end

# Constructor with default empty shared constraints
function PDGNEProblem(
    players::Vector{Player{T}},
    tf::T,
    dt::T
) where T
    return PDGNEProblem(players, ConstraintSpec[], tf, dt)
end




"""
    LQGameProblem{T, N, M, NP}

Concrete implementation of a finite-horizon linear-quadratic differential game.

# Type parameters
- `T` : numeric type (Float64, Float32, etc.)
- `N` : state dimension
- `M` : total control dimension (sum across all players)
- `NP` : number of players

# Fields
- `A::Matrix{T}` : n×n system dynamics matrix
- `B::Vector{Matrix{T}}` : control matrices [B₁, B₂, ..., Bₙₚ], each n×mᵢ
- `Q::Vector{Matrix{T}}` : state cost matrices [Q₁, Q₂, ..., Qₙₚ], each n×n
- `R::Vector{Matrix{T}}` : control cost matrices [R₁, R₂, ..., Rₙₚ], each mᵢ×mᵢ
- `Qf::Vector{Matrix{T}}` : terminal cost matrices [Qf₁, Qf₂, ..., Qfₙₚ], each n×n
- `x0::Vector{T}` : n-dimensional initial state
- `tf::T` : final time (time horizon)
- `control_dims::Vector{Int}` : control dimensions [m₁, m₂, ..., mₙₚ] for each player

# Notes
Represents the finite-horizon LQ game:
- Dynamics: ẋ(t) = Ax(t) + Σᵢ Bᵢuᵢ(t), x(0) = x0
- Cost for player i: Jᵢ = ∫₀ᵗᶠ [xᵀQᵢx + uᵢᵀRᵢuᵢ]dt + x(tf)ᵀQfᵢx(tf)
"""
struct LQGameProblem{T, N, M, NP} <: AbstractLQGame{T}
    A::Matrix{T}                  # n×n
    B::Vector{Matrix{T}}          # length NP, each n×mᵢ
    Q::Vector{Matrix{T}}          # length NP, each n×n
    R::Vector{Matrix{T}}          # length NP, each mᵢ×mᵢ
    Qf::Vector{Matrix{T}}         # length NP, each n×n
    x0::Vector{T}                 # length n
    tf::T                         # scalar
    control_dims::Vector{Int}     # length NP

    function LQGameProblem{T, N, M, NP}(
        A::Matrix{T},
        B::Vector{Matrix{T}},
        Q::Vector{Matrix{T}},
        R::Vector{Matrix{T}},
        Qf::Vector{Matrix{T}},
        x0::Vector{T},
        tf::T,
        control_dims::Vector{Int}
    ) where {T, N, M, NP}
        # Validation
        @assert size(A) == (N, N) "A must be n×n"
        @assert length(B) == NP "Must have NP control matrices"
        @assert length(Q) == NP "Must have NP state cost matrices"
        @assert length(R) == NP "Must have NP control cost matrices"
        @assert length(Qf) == NP "Must have NP terminal cost matrices"
        @assert length(x0) == N "x0 must have length n"
        @assert length(control_dims) == NP "Must specify control_dims for each player"
        @assert sum(control_dims) == M "Control dimensions must sum to M"
        @assert tf > 0 "Time horizon must be positive"
        
        # Validate dimensions
        for i in 1:NP
            @assert size(B[i], 1) == N "B[$i] must have $N rows"
            @assert size(B[i], 2) == control_dims[i] "B[$i] must have $(control_dims[i]) columns"
            @assert size(Q[i]) == (N, N) "Q[$i] must be n×n"
            @assert size(R[i]) == (control_dims[i], control_dims[i]) "R[$i] must be mᵢ×mᵢ"
            @assert size(Qf[i]) == (N, N) "Qf[$i] must be n×n"
        end
        
        new{T, N, M, NP}(A, B, Q, R, Qf, x0, tf, control_dims)
    end
end

# Convenience constructor
function LQGameProblem(
    A::Matrix{T},
    B::Vector{Matrix{T}},
    Q::Vector{Matrix{T}},
    R::Vector{Matrix{T}},
    Qf::Vector{Matrix{T}},
    x0::Vector{T},
    tf::T,
    control_dims::Vector{Int}
) where T
    N = size(A, 1)
    M = sum(control_dims)
    NP = length(B)
    return LQGameProblem{T, N, M, NP}(A, B, Q, R, Qf, x0, tf, control_dims)
end

"""
    PotentialGameProblem{T, N, M, NP}

Concrete implementation of a finite-horizon potential game.

# Type parameters
- `T` : numeric type
- `N` : state dimension
- `M` : total control dimension
- `NP` : number of players

# Fields
- `dynamics::Function` : dynamics function f(t, x, u) -> ẋ
- `potential::Function` : potential function P(t, x, u)
- `running_potential::Function` : running potential L(t, x, u)
- `terminal_potential::Function` : terminal potential Φ(x(tf))
- `constraints::Vector{Function}` : constraint functions
- `x0::Vector{T}` : initial state
- `tf::T` : final time
- `control_dims::Vector{Int}` : control dimensions per player

# Notes
Represents a potential game where:
- All player gradients derive from a single potential function
- ∇ᵤᵢJᵢ = ∇ᵤᵢP for all players i
- Total cost: P = ∫₀ᵗᶠ L(t,x,u)dt + Φ(x(tf))
"""
struct PotentialGameProblem{T, N, M, NP} <: AbstractPotentialGame{T}
    dynamics::Function
    running_potential::Function    # L(t, x, u)
    terminal_potential::Function   # Φ(x)
    constraints::Vector{Function}
    x0::Vector{T}
    tf::T
    control_dims::Vector{Int}
    
    function PotentialGameProblem{T, N, M, NP}(
        dynamics::Function,
        running_potential::Function,
        terminal_potential::Function,
        constraints::Vector{Function},
        x0::Vector{T},
        tf::T,
        control_dims::Vector{Int}
    ) where {T, N, M, NP}
        @assert length(x0) == N "x0 must have length N"
        @assert length(control_dims) == NP "Must specify control_dims for each player"
        @assert sum(control_dims) == M "Control dimensions must sum to M"
        @assert tf > 0 "Time horizon must be positive"
        
        new{T, N, M, NP}(dynamics, running_potential, terminal_potential, 
                         constraints, x0, tf, control_dims)
    end
end

# Convenience constructor
function PotentialGameProblem(
    dynamics::Function,
    running_potential::Function,
    terminal_potential::Function,
    constraints::Vector{Function},
    x0::Vector{T},
    tf::T,
    control_dims::Vector{Int}
) where T
    N = length(x0)
    M = sum(control_dims)
    NP = length(control_dims)
    return PotentialGameProblem{T, N, M, NP}(dynamics, running_potential, 
                                             terminal_potential, constraints, 
                                             x0, tf, control_dims)
end

"""
    LexicographicGameProblem{T, N, M, NP}

Concrete implementation of a finite-horizon lexicographic game.

# Type parameters
- `T` : numeric type
- `N` : state dimension
- `M` : total control dimension
- `NP` : number of players (priority levels)

# Fields
- `dynamics::Function` : dynamics function f(t, x, u) -> ẋ
- `running_costs::Vector{Function}` : ordered running costs [L₁(t,x,u₁), ...] by priority
- `terminal_costs::Vector{Function}` : ordered terminal costs [Φ₁(x), ...] by priority
- `priorities::Vector{Int}` : priority ordering (1 = highest priority)
- `constraints::Vector{Function}` : constraint functions
- `x0::Vector{T}` : initial state
- `tf::T` : final time
- `control_dims::Vector{Int}` : control dimensions per player
- `potential::Function` : equivalent potential function (computed from priorities)

# Notes
Lexicographic games optimize costs in hierarchical priority order:
- Player 1 optimizes J₁ first (highest priority)
- Player 2 optimizes J₂ subject to not degrading J₁
- And so on...
Recent work (Zargham et al.) shows these can be reformulated as potential games.
"""
struct LexicographicGameProblem{T, N, M, NP} <: AbstractLexicographicGame{T}
    dynamics::Function
    running_costs::Vector{Function}
    terminal_costs::Vector{Function}
    priorities::Vector{Int}
    constraints::Vector{Function}
    x0::Vector{T}
    tf::T
    control_dims::Vector{Int}
    potential::Function  # Derived from lexicographic structure
    
    function LexicographicGameProblem{T, N, M, NP}(
        dynamics::Function,
        running_costs::Vector{Function},
        terminal_costs::Vector{Function},
        priorities::Vector{Int},
        constraints::Vector{Function},
        x0::Vector{T},
        tf::T,
        control_dims::Vector{Int},
        potential::Function
    ) where {T, N, M, NP}
        @assert length(running_costs) == NP "Must have NP running cost functions"
        @assert length(terminal_costs) == NP "Must have NP terminal cost functions"
        @assert length(priorities) == NP "Must have NP priorities"
        @assert length(x0) == N "x0 must have length N"
        @assert length(control_dims) == NP "Must specify control_dims for each player"
        @assert sum(control_dims) == M "Control dimensions must sum to M"
        @assert tf > 0 "Time horizon must be positive"
        @assert all(1 .<= priorities .<= NP) "Priorities must be between 1 and NP"
        @assert length(unique(priorities)) == NP "Priorities must be unique"
        
        new{T, N, M, NP}(dynamics, running_costs, terminal_costs, priorities, 
                         constraints, x0, tf, control_dims, potential)
    end
end

# Convenience constructor with automatic potential function generation
function LexicographicGameProblem(
    dynamics::Function,
    running_costs::Vector{Function},
    terminal_costs::Vector{Function},
    priorities::Vector{Int},
    constraints::Vector{Function},
    x0::Vector{T},
    tf::T,
    control_dims::Vector{Int};
    potential::Union{Function, Nothing} = nothing
) where T
    N = length(x0)
    M = sum(control_dims)
    NP = length(running_costs)
    
    # Generate potential function if not provided
    # Using exponential weighting: P = Σᵢ exp(α(NP - priority[i])) * Jᵢ
    if isnothing(potential)
        α = 10.0  # Separation parameter (larger = stricter priority)
        potential = (t, x, u) -> begin
            val = zero(T)
            for i in 1:NP
                weight = exp(α * (NP - priorities[i]))
                val += weight * running_costs[i](t, x, u[control_dims[1:i-1] .+ 1:sum(control_dims[1:i])])
            end
            return val
        end
    end
    
    return LexicographicGameProblem{T, N, M, NP}(dynamics, running_costs, 
                                                  terminal_costs, priorities, 
                                                  constraints, x0, tf, 
                                                  control_dims, potential)
end

# Helper functions
num_players(::Union{GNEProblem{T,N,M,NP}, LQGameProblem{T,N,M,NP}, 
                    PotentialGameProblem{T,N,M,NP}, LexicographicGameProblem{T,N,M,NP}}) where {T,N,M,NP} = NP
state_dim(::Union{GNEProblem{T,N,M,NP}, LQGameProblem{T,N,M,NP}, 
                  PotentialGameProblem{T,N,M,NP}, LexicographicGameProblem{T,N,M,NP}}) where {T,N,M,NP} = N
control_dim(::Union{GNEProblem{T,N,M,NP}, LQGameProblem{T,N,M,NP}, 
                    PotentialGameProblem{T,N,M,NP}, LexicographicGameProblem{T,N,M,NP}}) where {T,N,M,NP} = M
control_dim(game::Union{GNEProblem, LQGameProblem, PotentialGameProblem, LexicographicGameProblem}, player::Int) = game.control_dims[player]

"""
    get_player(game::PDGNEProblem, id::Int)

Get player by ID.
"""
function get_player(game::PDGNEProblem, id::Int)
    idx = findfirst(p -> p.id == id, game.players)
    isnothing(idx) && error("Player with id=$id not found")
    return game.players[idx]
end