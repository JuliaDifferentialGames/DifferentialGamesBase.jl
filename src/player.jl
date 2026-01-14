"""
    Player{T}

Defines a single agent in a PD-GNEP with separable dynamics.

# Type Parameters
- `T` : Numeric type (Float64, Float32, etc.)

# Fields
- `id::Int` : Player identifier (1 to N)
- `n::Int` : State dimension for this player
- `m::Int` : Control dimension for this player
- `x0::Vector{T}` : Initial state (length n)
- `dynamics::Function` : Dynamics function fⁱ(xⁱ, uⁱ, p, t) -> ẋⁱ
- `running_cost::Function` : Running cost Lⁱ(X, uⁱ, p, t) where X is vector of all states
- `terminal_cost::Function` : Terminal cost Φⁱ(X) where X is vector of all states
- `private_constraints::Vector{ConstraintSpec}` : Constraints on (xⁱ, uⁱ) only
- `parameters::Any` : Optional parameters passed to dynamics and costs

# Notes
- `dynamics` has signature: fⁱ(xⁱ::AbstractVector, uⁱ::AbstractVector, p, t::Real) -> Vector
- `running_cost` has signature: Lⁱ(X::Vector{Vector}, uⁱ::AbstractVector, p, t::Real) -> Real
  where X = [x¹, x², ..., xᴺ] are all player states
- `terminal_cost` has signature: Φⁱ(X::Vector{Vector}) -> Real
- Private constraints apply only to player i's own state and control

# Example
```julia
player = Player(
    id=1,
    n=4,
    m=2,
    x0=[1.0, 0.0, 0.0, 0.0],
    dynamics=(xⁱ, uⁱ, p, t) -> [xⁱ[3], xⁱ[4], uⁱ[1], uⁱ[2]],
    running_cost=(X, uⁱ, p, t) -> sum(abs2, X[1]) + 0.1*sum(abs2, uⁱ),
    terminal_cost=(X) -> 10.0*sum(abs2, X[1])
)
```
"""
struct Player{T}
    id::Int
    n::Int                                    # State dimension
    m::Int                                    # Control dimension
    x0::Vector{T}                             # Initial state
    dynamics::Function                        # fⁱ(xⁱ, uⁱ, p, t) -> ẋⁱ
    running_cost::Function                    # Lⁱ(X, uⁱ, p, t) -> R
    terminal_cost::Function                   # Φⁱ(X) -> R
    private_constraints::Vector{ConstraintSpec}
    parameters::Any                           # User-defined parameters
    
    function Player{T}(
        id::Int,
        n::Int,
        m::Int,
        x0::Vector{T},
        dynamics::Function,
        running_cost::Function,
        terminal_cost::Function,
        private_constraints::Vector{ConstraintSpec} = ConstraintSpec[],
        parameters = nothing
    ) where T
        @assert n > 0 "State dimension must be positive"
        @assert m > 0 "Control dimension must be positive"
        @assert length(x0) == n "x0 must have length n=$n"
        @assert id > 0 "Player id must be positive"
        
        new{T}(id, n, m, x0, dynamics, running_cost, terminal_cost, 
               private_constraints, parameters)
    end
end

# Convenience constructor with keyword arguments
function Player(;
    id::Int,
    n::Int,
    m::Int,
    x0::Vector{T},
    dynamics::Function,
    running_cost::Function = (X, uⁱ, p, t) -> zero(T),
    terminal_cost::Function = (X) -> zero(T),
    private_constraints::Vector{ConstraintSpec} = ConstraintSpec[],
    parameters = nothing
) where T
    return Player{T}(id, n, m, x0, dynamics, running_cost, terminal_cost,
                     private_constraints, parameters)
end

# Helper functions for Player
state_dim(p::Player) = p.n
control_dim(p::Player) = p.m
player_id(p::Player) = p.id
initial_state(p::Player) = p.x0

"""
    add_constraint!(player::Player, constraint::ConstraintSpec)

Add a private constraint to a player.
"""
function add_constraint!(player::Player, constraint::ConstraintSpec)
    @assert is_private(constraint) "Use add_shared_constraint! for shared constraints"
    push!(player.private_constraints, constraint)
    return player
end