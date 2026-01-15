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