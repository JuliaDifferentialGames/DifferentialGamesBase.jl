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