# Abstract base types
abstract type AbstractConstraint end
abstract type AbstractEqualityConstraint <: AbstractConstraint end
abstract type AbstractInequalityConstraint <: AbstractConstraint end
abstract type AbstractConvexConstraint <: AbstractInequalityConstraint end

# Player assignment wrappers
struct PrivateConstraint{C <: AbstractConstraint}
    constraint::C
    player::Int
    
    function PrivateConstraint(constraint::C, player::Int) where {C <: AbstractConstraint}
        @assert player > 0 "Player index must be positive"
        new{C}(constraint, player)
    end
end

struct SharedConstraint{C <: AbstractConstraint}
    constraint::C
    players::Vector{Int}  # All involved players
    
    function SharedConstraint(constraint::C, players::Vector{Int}) where {C <: AbstractConstraint}
        @assert !isempty(players) "Shared constraint must involve at least one player"
        @assert allunique(players) "Player indices must be unique"
        @assert all(p -> p > 0, players) "Player indices must be positive"
        new{C}(constraint, sort(players))
    end
end


"""
    LinearConstraint{T}

Represents linear inequality constraint: A*z ≤ b where z is either state x or control u.

# Mathematical Form
- applies_to = :x: Aₓ xᵢ ≤ b
- applies_to = :u: Aᵤ uᵢ ≤ b

# Fields
- `A::Matrix{T}`: Constraint matrix (m × n_z)
- `b::Vector{T}`: Constraint vector (m,)
- `applies_to::Symbol`: Either :x or :u

# Notes
For coupled state-control constraints, use NonlinearConstraint with explicit form.
Dimension n_z must match state dimension for :x, control dimension for :u.
"""
struct LinearConstraint{T} <: AbstractConvexConstraint
    A::Matrix{T}
    b::Vector{T}
    applies_to::Symbol  # :x or :u
    
    function LinearConstraint(A::Matrix{T}, b::Vector{T}, applies_to::Symbol) where {T}
        @assert size(A, 1) == length(b) "Constraint matrix and vector dimensions must match"
        @assert applies_to in (:x, :u) "applies_to must be :x or :u"
        new{T}(A, b, applies_to)
    end
end

# Convenience constructors
LinearConstraint(A::Matrix{T}, b::Vector{T}; applies_to::Symbol=:u) where {T} = 
    LinearConstraint(A, b, applies_to)

"""
    BoundConstraint{T}

Box constraints: lower ≤ z ≤ upper where z is either state x or control u.

# Mathematical Form
- applies_to = :x: x_min ≤ xᵢ ≤ x_max
- applies_to = :u: u_min ≤ uᵢ ≤ u_max

# Fields
- `lower::Vector{T}`: Lower bounds (use -Inf for unbounded)
- `upper::Vector{T}`: Upper bounds (use Inf for unbounded)
- `applies_to::Symbol`: Either :x or :u

# Notes
Extremely efficient to evaluate and project onto.
Dimensions must match state/control dimensions.
"""
struct BoundConstraint{T} <: AbstractConvexConstraint
    lower::Vector{T}
    upper::Vector{T}
    applies_to::Symbol  # :x or :u
    
    function BoundConstraint(lower::Vector{T}, upper::Vector{T}, applies_to::Symbol) where {T}
        @assert length(lower) == length(upper) "Lower and upper bounds must have same length"
        @assert all(lower .<= upper) "Lower bounds must be ≤ upper bounds"
        @assert applies_to in (:x, :u) "applies_to must be :x or :u"
        new{T}(lower, upper, applies_to)
    end
end

BoundConstraint(lower::Vector{T}, upper::Vector{T}; applies_to::Symbol=:u) where {T} = 
    BoundConstraint(lower, upper, applies_to)

"""
    NormConstraint{T, P}

Norm-based constraints: ||A*z||_p ≤ b where z is either state x or control u.

# Mathematical Form
- applies_to = :x: ||Aₓ xᵢ||_p ≤ b
- applies_to = :u: ||Aᵤ uᵢ||_p ≤ b

# Common Cases
- p = 2: Second-order cone (ellipsoidal constraints)
- p = 1: L1 penalty (sparsity-inducing)
- p = Inf: L∞ constraint (minimax)

# Fields
- `A::Matrix{T}`: Linear transformation matrix
- `b::T`: Norm bound
- `p::P`: Norm type (2, 1, Inf)
- `applies_to::Symbol`: Either :x or :u
"""
struct NormConstraint{T, P} <: AbstractConvexConstraint
    A::Matrix{T}
    b::T
    p::P  # Norm type: 1, 2, Inf
    applies_to::Symbol
    
    function NormConstraint(A::Matrix{T}, b::T, p::P, applies_to::Symbol) where {T, P}
        @assert b >= 0 "Norm bound must be non-negative"
        @assert p in (1, 2, Inf) "Only p ∈ {1, 2, ∞} norms supported"
        @assert applies_to in (:x, :u) "applies_to must be :x or :u"
        new{T, P}(A, b, p, applies_to)
    end
end

NormConstraint(A::Matrix{T}, b::T, p::P; applies_to::Symbol=:u) where {T, P} = 
    NormConstraint(A, b, p, applies_to)


"""
    NonlinearConstraint{F, J, H, CT}

General nonlinear constraint with optional analytical derivatives.

# Mathematical Form
- Equality: c(x, u, p, t) = 0
- Inequality: c(x, u, p, t) ≤ 0

# Fields
- `func::F`: Constraint function c(x, u, p, t) -> Vector{T}
- `jacobian::J`: Optional Jacobian ∇c(x, u, p, t) -> (∂c/∂x, ∂c/∂u)
- `hessian::H`: Optional Hessian for Lagrangian ∇²(λᵀc)
- `constraint_type::Symbol`: :equality or :inequality
- `dim::Int`: Output dimension of c
- `is_convex::Bool`: User assertion for convexity

# Jacobian Interface
If provided, jacobian should return tuple (J_x, J_u) where:
- J_x::Matrix{T} is (dim × n_x) - derivative w.r.t. state
- J_u::Matrix{T} is (dim × n_u) - derivative w.r.t. control

If Nothing, automatic differentiation will be used.

# Notes
For x = [x₁; x₂; ...; xₙ] (stacked state), ensure function expects full state vector.
Parameter p can contain time-varying parameters or coupling terms.
"""
struct NonlinearConstraint{F, J, H, CT} <: AbstractConstraint
    func::F
    jacobian::J  # Nothing or Function returning (J_x, J_u)
    hessian::H   # Nothing or Function for second-order
    constraint_type::Symbol  # :equality or :inequality
    dim::Int
    is_convex::Bool
    
    function NonlinearConstraint(
        func::F,
        jacobian::J,
        hessian::H,
        constraint_type::Symbol,
        dim::Int,
        is_convex::Bool
    ) where {F, J, H}
        @assert dim > 0 "Constraint dimension must be positive"
        @assert constraint_type in (:equality, :inequality) "Type must be :equality or :inequality"
        
        # Type parameter for dispatch
        CT = constraint_type
        new{F, J, H, CT}(func, jacobian, hessian, constraint_type, dim, is_convex)
    end
end

# Convenience constructors
function NonlinearConstraint(
    func::Function,
    dim::Int;
    constraint_type::Symbol = :inequality,
    jacobian::Union{Nothing, Function} = nothing,
    hessian::Union{Nothing, Function} = nothing,
    is_convex::Bool = false
)
    NonlinearConstraint(func, jacobian, hessian, constraint_type, dim, is_convex)
end

# Type checking helpers
is_equality(c::NonlinearConstraint{F,J,H,:equality}) where {F,J,H} = true
is_equality(c::NonlinearConstraint) = false
is_inequality(c::NonlinearConstraint{F,J,H,:inequality}) where {F,J,H} = true
is_inequality(c::NonlinearConstraint) = false


"""
    evaluate_constraint(c::AbstractConstraint, x, u, p, t)

Evaluate constraint function value.

Returns Vector{T} of dimension c.dim.
"""
function evaluate_constraint end

"""
    constraint_jacobian(c::AbstractConstraint, x, u, p, t)

Compute constraint Jacobian (∂c/∂x, ∂c/∂u).

Returns tuple (J_x, J_u) of sparse or dense matrices.
"""
function constraint_jacobian end

# Implementations for structured convex constraints

function evaluate_constraint(c::LinearConstraint{T}, x, u, p, t) where {T}
    z = (c.applies_to == :x) ? x : u
    return c.A * z - c.b  # Returns violation (≤ 0 satisfied)
end

function constraint_jacobian(c::LinearConstraint{T}, x, u, p, t) where {T}
    n_x, n_u = length(x), length(u)
    m = size(c.A, 1)
    
    if c.applies_to == :x
        return (c.A, zeros(T, m, n_u))
    else
        return (zeros(T, m, n_x), c.A)
    end
end

function evaluate_constraint(c::BoundConstraint{T}, x, u, p, t) where {T}
    z = (c.applies_to == :x) ? x : u
    
    # Stack lower and upper bound violations
    # Lower: lower - z ≤ 0  =>  z ≥ lower
    # Upper: z - upper ≤ 0  =>  z ≤ upper
    return vcat(c.lower - z, z - c.upper)
end

function constraint_jacobian(c::BoundConstraint{T}, x, u, p, t) where {T}
    n_x, n_u = length(x), length(u)
    n_z = (c.applies_to == :x) ? n_x : n_u
    m = 2 * n_z  # Both lower and upper bounds
    
    # Jacobian is [−I; I] for the relevant variable
    if c.applies_to == :x
        J_x = vcat(-I(n_z), I(n_z))
        J_u = zeros(T, m, n_u)
        return (J_x, J_u)
    else
        J_x = zeros(T, m, n_x)
        J_u = vcat(-I(n_z), I(n_z))
        return (J_x, J_u)
    end
end

function evaluate_constraint(c::NormConstraint{T, P}, x, u, p, t) where {T, P}
    z = (c.applies_to == :x) ? x : u
    Az = c.A * z
    
    # Return scalar: ||Az||_p - b ≤ 0
    norm_val = if c.p == 2
        norm(Az, 2)
    elseif c.p == 1
        norm(Az, 1)
    else  # p == Inf
        norm(Az, Inf)
    end
    
    return [norm_val - c.b]
end

function constraint_jacobian(c::NormConstraint{T, P}, x, u, p, t) where {T, P}
    z = (c.applies_to == :x) ? x : u
    Az = c.A * z
    n_x, n_u = length(x), length(u)
    
    # ∂||Az||_p/∂z depends on norm type
    grad_norm = if c.p == 2
        # ∂||Az||₂/∂z = Aᵀ(Az/||Az||₂)
        norm_val = norm(Az, 2)
        (norm_val > eps(T)) ? c.A' * (Az / norm_val) : zeros(T, size(c.A, 2))
    elseif c.p == 1
        # ∂||Az||₁/∂z = Aᵀ sign(Az)
        c.A' * sign.(Az)
    else  # p == Inf
        # Subdifferential of max - use sign of max element
        idx_max = argmax(abs.(Az))
        grad = zeros(T, length(Az))
        grad[idx_max] = sign(Az[idx_max])
        c.A' * grad
    end
    
    if c.applies_to == :x
        return (reshape(grad_norm, 1, :), zeros(T, 1, n_u))
    else
        return (zeros(T, 1, n_x), reshape(grad_norm, 1, :))
    end
end


"""
    automatic_differentiation_jacobian_sparse(func, x, u, p, t, sparsity_x, sparsity_u)

Sparse AD variant for large-scale problems with known sparsity structure.

# Arguments
- `sparsity_x`: Sparse matrix pattern for J_x (m × n_x)
- `sparsity_u`: Sparse matrix pattern for J_u (m × n_u)

# Notes
Requires SparseDiffTools.jl. Use when n_x or n_u > 100 and Jacobian is sparse.
For spacecraft formation flying with N agents and n-dimensional states,
collision constraints couple only pairs of agents, leading to sparsity O(N²) 
in constraint count but O(1) nonzeros per constraint row.
"""
function automatic_differentiation_jacobian_sparse(
    func::F, 
    x, 
    u, 
    p, 
    t,
    sparsity_x::SparseMatrixCSC,
    sparsity_u::SparseMatrixCSC
) where {F}
    # Placeholder - requires SparseDiffTools integration
    # See: https://github.com/JuliaDiff/SparseDiffTools.jl
    # Key idea: use matrix coloring to reduce number of AD passes
    # from O(n) to O(χ) where χ is chromatic number of sparsity graph
    
    error("Sparse AD not yet implemented. Use dense version or provide analytical Jacobian.")
end


"""
    automatic_differentiation_jacobian(func, x, u, p, t)

Compute constraint Jacobian using forward-mode automatic differentiation.

Returns tuple (J_x, J_u) where:
- J_x: (m × n_x) Jacobian w.r.t. state
- J_u: (m × n_u) Jacobian w.r.t. control

# Implementation Notes
Uses ForwardDiff.jacobian with dual number seeding for each variable block.
For functions with m outputs and n inputs, forward mode requires m passes.
Cost: O(m * n) in the worst case, but ForwardDiff uses efficient SIMD chunks.

# Numerical Considerations
- Chunk size automatically selected by ForwardDiff for SIMD efficiency
- For large n_x or n_u (>100), consider using sparse AD (see SparseDiffTools.jl)
- Thread-safety: ForwardDiff.jacobian allocates new dual number cache per call
"""
function automatic_differentiation_jacobian(func::F, x, u, p, t) where {F}
    n_x = length(x)
    n_u = length(u)
    
    # Create wrapper function that takes concatenated input [x; u]
    # This allows single jacobian call instead of separate ∂c/∂x and ∂c/∂u
    function augmented_func(z)
        x_local = z[1:n_x]
        u_local = z[n_x+1:end]
        return func(x_local, u_local, p, t)
    end
    
    # Compute full Jacobian w.r.t. [x; u]
    z = vcat(x, u)
    J_full = ForwardDiff.jacobian(augmented_func, z)
    
    # Extract blocks: J_full = [J_x | J_u]
    J_x = J_full[:, 1:n_x]
    J_u = J_full[:, n_x+1:end]
    
    return (J_x, J_u)
end

"""
    ForwardDiffConfig{N}

Pre-allocated configuration for repeated Jacobian evaluations.

Stores dual number cache to avoid repeated allocations in tight loops.
Useful when evaluating same constraint many times (e.g., in line search).

# Usage
```julia
config = ForwardDiffConfig(c, x, u, p, t)
J_x, J_u = constraint_jacobian(c, x, u, p, t, config)
```
"""
struct ForwardDiffConfig{N, T}
    chunk_size::Int
    cache::ForwardDiff.JacobianConfig{T, T, N, Vector{ForwardDiff.Dual{T, T, N}}}
end

function ForwardDiffConfig(func::F, x::Vector{T}, u::Vector{T}, p, t) where {F, T}
    n_x = length(x)
    n_u = length(u)
    z = vcat(x, u)
    
    # ForwardDiff automatically selects chunk size, but can override
    # Chunk size trades off memory vs computation:
    # - Larger chunks: more memory, fewer passes, better SIMD
    # - Smaller chunks: less memory, more passes
    chunk = ForwardDiff.Chunk(z)
    N = length(chunk)
    
    augmented_func = z_aug -> begin
        x_local = z_aug[1:n_x]
        u_local = z_aug[n_x+1:end]
        func(x_local, u_local, p, t)
    end
    
    cache = ForwardDiff.JacobianConfig(augmented_func, z, chunk)
    
    return ForwardDiffConfig{N, T}(N, cache)
end

# Specialized method using pre-allocated config
function constraint_jacobian(
    c::NonlinearConstraint, 
    x, 
    u, 
    p, 
    t,
    config::ForwardDiffConfig
)
    if c.jacobian !== nothing
        return c.jacobian(x, u, p, t)
    else
        n_x = length(x)
        n_u = length(u)
        z = vcat(x, u)
        
        augmented_func = z_aug -> begin
            x_local = z_aug[1:n_x]
            u_local = z_aug[n_x+1:end]
            c.func(x_local, u_local, p, t)
        end
        
        # Use pre-allocated cache
        J_full = ForwardDiff.jacobian(augmented_func, z, config.cache)
        
        J_x = J_full[:, 1:n_x]
        J_u = J_full[:, n_x+1:end]
        
        return (J_x, J_u)
    end
end


# For NonlinearConstraint, use provided jacobian or AD
function evaluate_constraint(c::NonlinearConstraint, x, u, p, t)
    return c.func(x, u, p, t)
end

function constraint_jacobian(c::NonlinearConstraint, x, u, p, t)
    if c.jacobian !== nothing
        return c.jacobian(x, u, p, t)
    else
        # AD fallbackl
        return automatic_differentiation_jacobian(c.func, x, u, p, t)
    end
end


struct ScaledConstraint{C <: AbstractConstraint} <: AbstractConstraint
    constraint::C
    scaling::Vector{Float64}  # Per-constraint-row scaling factors
end


mutable struct ConstraintViolation
    value::Float64
    is_active::Bool
    dual_variable::Float64  # Lagrange multiplier
end