abstract type DynamicsSpec{T} end

"""
Separable dynamics: ẋᵢ = fᵢ(xᵢ, uᵢ, p, t)

Each player's dynamics depend only on their own state and control.
Enables parallel propagation and decoupled Jacobians.
"""
struct SeparableDynamics{T, F} <: DynamicsSpec{T}
    player_dynamics::Vector{F}  # [f₁, f₂, ..., fₙ]
    state_dims::Vector{Int}
    control_dims::Vector{Int}
    
    function SeparableDynamics(
        player_dynamics::Vector{F},
        state_dims::Vector{Int},
        control_dims::Vector{Int}
    ) where {F}
        @assert length(player_dynamics) == length(state_dims)
        @assert length(player_dynamics) == length(control_dims)
        T = Float64  # Infer from usage or require explicit
        new{T, F}(player_dynamics, state_dims, control_dims)
    end
end

# Convenience constructor
SeparableDynamics(dynamics::Vector{F}) where {F} = 
    SeparableDynamics(dynamics, Int[], Int[])  # Infer dims from first call

"""
Coupled linear dynamics: ẋ = Ax + Σᵢ Bᵢuᵢ

Standard LQ game formulation with shared state space.
"""
struct LinearDynamics{T} <: DynamicsSpec{T}
    A::Matrix{T}
    B::Vector{Matrix{T}}  # Per-player control matrices
    
    function LinearDynamics(A::Matrix{T}, B::Vector{Matrix{T}}) where {T}
        n = size(A, 1)
        @assert size(A) == (n, n)
        @assert all(size(Bi, 1) == n for Bi in B)
        new{T}(A, B)
    end
end

"""
Coupled nonlinear dynamics: ẋ = f(x, u, p, t)

Most general case where all state components can depend on all controls.
Requires full Jacobian computation.
"""
struct CoupledNonlinearDynamics{T, F} <: DynamicsSpec{T}
    func::F  # f(x, u, p, t) -> ẋ
    n::Int
    m::Int
    jacobian::Union{Nothing, Function}  # Optional analytical Jacobian
    
    function CoupledNonlinearDynamics(
        func::F,
        n::Int,
        m::Int;
        jacobian::Union{Nothing, Function} = nothing
    ) where {F}
        T = Float64
        new{T, F}(func, n, m, jacobian)
    end
end

# Evaluation interface
"""Evaluate dynamics: ẋ = f(x, u, p, t)"""
function evaluate_dynamics end

"""Compute dynamics Jacobian: (∂f/∂x, ∂f/∂u)"""
function dynamics_jacobian end

# Implementations
function evaluate_dynamics(dyn::SeparableDynamics, x, u, p, t)
    # x, u are stacked vectors
    # Extract per-player slices and evaluate
    ẋ = similar(x)
    x_offset = 0
    u_offset = 0
    for (i, fi) in enumerate(dyn.player_dynamics)
        ni = dyn.state_dims[i]
        mi = dyn.control_dims[i]
        
        xi = x[x_offset .+ (1:ni)]
        ui = u[u_offset .+ (1:mi)]
        
        ẋ[x_offset .+ (1:ni)] = fi(xi, ui, p, t)
        
        x_offset += ni
        u_offset += mi
    end
    return ẋ
end

function evaluate_dynamics(dyn::LinearDynamics, x, u, p, t)
    return dyn.A * x + sum(Bi * ui for (Bi, ui) in zip(dyn.B, split_controls(u, control_dims)))
end

function evaluate_dynamics(dyn::CoupledNonlinearDynamics, x, u, p, t)
    return dyn.func(x, u, p, t)
end

# Jacobians (separable case is block-diagonal)
function dynamics_jacobian(dyn::SeparableDynamics, x, u, p, t)
    # Exploit separability: Jacobian is block-diagonal
    n_total = sum(dyn.state_dims)
    m_total = sum(dyn.control_dims)
    
    Jx = zeros(n_total, n_total)
    Ju = zeros(n_total, m_total)
    
    x_offset = 0
    u_offset = 0
    for (i, fi) in enumerate(dyn.player_dynamics)
        ni = dyn.state_dims[i]
        mi = dyn.control_dims[i]
        
        xi = x[x_offset .+ (1:ni)]
        ui = u[u_offset .+ (1:mi)]
        
        # Compute per-player Jacobian (small problem)
        Jxi, Jui = player_dynamics_jacobian(fi, xi, ui, p, t)
        
        # Place in block-diagonal structure
        Jx[x_offset .+ (1:ni), x_offset .+ (1:ni)] = Jxi
        Ju[x_offset .+ (1:ni), u_offset .+ (1:mi)] = Jui
        
        x_offset += ni
        u_offset += mi
    end
    
    return (Jx, Ju)
end