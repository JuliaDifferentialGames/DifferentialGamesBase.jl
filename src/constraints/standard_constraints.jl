# ============================================================================
# constraints/standard_constraints.jl
#
# Ergonomic constructors for common constraint patterns.
# Mirrors the standard_costs.jl API: named keyword constructors that return
# the appropriate concrete type without requiring knowledge of type names.
#
# All constructors follow the pattern:
#   constraint_name(player_or_players; keyword_args...) → AbstractConstraint
# ============================================================================

# ============================================================================
# Private convenience constructors
# ============================================================================

"""
    control_bounds(player; control_offset, control_dim, lower, upper) -> ControlBounds

Box constraint on player i's control: `lower ≤ uᵢ ≤ upper`.

# Example
```julia
# Player 1: thrust limited to ±5 N in 2D
c = control_bounds(1; control_offset=0, control_dim=2,
                   lower=fill(-5.0, 2), upper=fill(5.0, 2))
```
"""
function control_bounds(
    player::Int;
    control_offset::Int,
    control_dim::Int,
    lower::Vector{T},
    upper::Vector{T}
) where {T}
    ControlBounds(player, control_offset, control_dim, lower, upper)
end

"""
    state_bounds(player; state_offset, state_dim, lower, upper) -> StateBounds

Box constraint on player i's state: `lower ≤ xᵢ ≤ upper`.

# Example
```julia
# Player 2: position in [-10, 10]^2, velocity in [-5, 5]^2
c = state_bounds(2; state_offset=4, state_dim=4,
                 lower=[-10,-10,-5,-5.], upper=[10,10,5,5.])
```
"""
function state_bounds(
    player::Int;
    state_offset::Int,
    state_dim::Int,
    lower::Vector{T},
    upper::Vector{T}
) where {T}
    StateBounds(player, state_offset, state_dim, lower, upper)
end

# ============================================================================
# Shared convenience constructors
# ============================================================================

"""
    collision_avoidance(players; i_offset, j_offset, pos_dim, d_min, ε=1e-6)
        -> ProximityConstraint

Hard separation constraint: agents `players[1]` and `players[2]` must stay
at least `d_min` apart.

Uses regularised Euclidean distance for ForwardDiff compatibility.

# Example
```julia
# Players 1 and 2, 2D position in first 2 components of 4-state vector
c = collision_avoidance([1, 2]; i_offset=0, j_offset=4, pos_dim=2, d_min=0.5)
```
"""
function collision_avoidance(
    players::Vector{Int};
    i_offset::Int,
    j_offset::Int,
    pos_dim::Int,
    d_min::T,
    ε::Real = 1e-6
) where {T}
    ProximityConstraint(players, i_offset, j_offset, pos_dim, d_min; ε = T(ε))
end

"""
    keep_in_range(players; i_offset, j_offset, pos_dim, d_max, ε=1e-6)
        -> CommunicationConstraint

Maximum separation constraint: agents must stay within `d_max` of each other.
Used for communication-constrained or formation-keeping problems.

# Example
```julia
c = keep_in_range([1, 2]; i_offset=0, j_offset=4, pos_dim=2, d_max=10.0)
```
"""
function keep_in_range(
    players::Vector{Int};
    i_offset::Int,
    j_offset::Int,
    pos_dim::Int,
    d_max::T,
    ε::Real = 1e-6
) where {T}
    CommunicationConstraint(players, i_offset, j_offset, pos_dim, d_max; ε = T(ε))
end

"""
    linear_coupling(players, A, b) -> LinearCoupling

Shared linear inequality: `A * [x; u] ≤ b`.

# Example
```julia
# Shared thrust budget: u₁[1] + u₂[1] ≤ 10
c = linear_coupling([1, 2], reshape([0.0 0.0 1.0 0.0 0.0 1.0], 1, :), [10.0])
```
"""
function linear_coupling(
    players::Vector{Int},
    A::Matrix{T},
    b::Vector{T}
) where {T}
    LinearCoupling(players, A, b)
end