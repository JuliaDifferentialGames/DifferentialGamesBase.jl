# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractCost

Base type for all cost functions in differential games.
"""
abstract type AbstractCost end

"""
    AbstractStageCost <: AbstractCost

Cost evaluated at each time step t ∈ [0, N-1].
"""
abstract type AbstractStageCost <: AbstractCost end

"""
    AbstractTerminalCost <: AbstractCost

Cost evaluated once at final time t = N.
"""
abstract type AbstractTerminalCost <: AbstractCost end

# ============================================================================
# NOTE: LQStageCost, LQTerminalCost, and PlayerObjective are defined in
# problems/GNEP.jl as parametric types supporting both LTI and LTV costs.
# All evaluation methods for those types are implemented below.
# ============================================================================

# ============================================================================
# DiagonalLQStageCost
# ============================================================================

"""
    DiagonalLQStageCost{T} <: AbstractStageCost

Diagonal LQ cost for computational efficiency: ℓ(x, u) = 1/2 xᵀ diag(qx) x + 1/2 uᵀ diag(ru) u + c

# Fields
- `qx::Vector{T}`: Diagonal state costs (n_x,), must be non-negative
- `ru::Vector{T}`: Diagonal control costs (n_u,), must be positive
- `c::T`: Constant offset
"""
struct DiagonalLQStageCost{T} <: AbstractStageCost
    qx::Vector{T}
    ru::Vector{T}
    c::T

    function DiagonalLQStageCost(qx::Vector{T}, ru::Vector{T}, c::T = zero(T)) where {T}
        @assert all(qx .>= 0) "Diagonal state costs must be non-negative"
        @assert all(ru .> 0) "Diagonal control costs must be positive"
        new{T}(qx, ru, c)
    end
end

# ============================================================================
# NonlinearStageCost
# ============================================================================

"""
    NonlinearStageCost{F, G, H} <: AbstractStageCost

General nonlinear stage cost with optional analytical derivatives.

# Fields
- `func::F`: Cost function ℓ(x, u, p, t) -> scalar
- `gradient::G`: Optional gradient function (x, u, p, t) -> (∇ₓℓ, ∇ᵤℓ)
- `hessian::H`: Optional Hessian function (x, u, p, t) -> (Hₓₓ, Hᵤᵤ, Hₓᵤ)
- `is_separable::Bool`: True if ℓ(xᵢ, uᵢ) only (enables sparse gradients)
"""
struct NonlinearStageCost{F, G, H} <: AbstractStageCost
    func::F
    gradient::G
    hessian::H
    is_separable::Bool

    function NonlinearStageCost(
        func::F,
        gradient::G,
        hessian::H,
        is_separable::Bool
    ) where {F, G, H}
        new{F, G, H}(func, gradient, hessian, is_separable)
    end
end

function NonlinearStageCost(
    func::Function;
    gradient::Union{Nothing, Function} = nothing,
    hessian::Union{Nothing, Function} = nothing,
    is_separable::Bool = false
)
    NonlinearStageCost(func, gradient, hessian, is_separable)
end

# ============================================================================
# DiagonalLQTerminalCost
# ============================================================================

"""
    DiagonalLQTerminalCost{T} <: AbstractTerminalCost

Diagonal terminal cost: φ(x) = 1/2 xᵀ diag(qf) x + cf
"""
struct DiagonalLQTerminalCost{T} <: AbstractTerminalCost
    qf::Vector{T}
    cf::T

    function DiagonalLQTerminalCost(qf::Vector{T}, cf::T = zero(T)) where {T}
        @assert all(qf .>= 0) "Diagonal terminal costs must be non-negative"
        new{T}(qf, cf)
    end
end

# ============================================================================
# NonlinearTerminalCost
# ============================================================================

"""
    NonlinearTerminalCost{F, G, H} <: AbstractTerminalCost

General nonlinear terminal cost with optional analytical derivatives.

# Fields
- `func::F`: Terminal cost function φ(x, p) -> scalar
- `gradient::G`: Optional gradient function (x, p) -> ∇ₓφ
- `hessian::H`: Optional Hessian function (x, p) -> Hₓₓ
"""
struct NonlinearTerminalCost{F, G, H} <: AbstractTerminalCost
    func::F
    gradient::G
    hessian::H

    function NonlinearTerminalCost(func::F, gradient::G, hessian::H) where {F, G, H}
        new{F, G, H}(func, gradient, hessian)
    end
end

function NonlinearTerminalCost(
    func::Function;
    gradient::Union{Nothing, Function} = nothing,
    hessian::Union{Nothing, Function} = nothing
)
    NonlinearTerminalCost(func, gradient, hessian)
end

# ============================================================================
# PlayerObjective
# ============================================================================
#
# Defined here (not in GNEP.jl) so that PlayerSpec in player_spec.jl can
# reference it. GNEP.jl loads after player_spec.jl; objectives.jl loads first.

"""
    PlayerObjective{SC <: AbstractStageCost, TC <: AbstractTerminalCost}

Bundles a player's stage and terminal cost functions.

# Fields
- `player_id`     : Player index (1-based)
- `stage_cost`    : Stage cost evaluated at each timestep
- `terminal_cost` : Terminal cost evaluated at x(tf)
- `scaling`       : Global cost scaling factor (default 1.0)
"""
struct PlayerObjective{SC <: AbstractStageCost, TC <: AbstractTerminalCost}
    player_id::Int
    stage_cost::SC
    terminal_cost::TC
    scaling::Float64

    function PlayerObjective(
        player_id::Int,
        stage_cost::SC,
        terminal_cost::TC,
        scaling::Float64 = 1.0
    ) where {SC <: AbstractStageCost, TC <: AbstractTerminalCost}
        @assert player_id > 0 "Player ID must be positive"
        @assert scaling > 0 "Cost scaling must be positive"
        new{SC, TC}(player_id, stage_cost, terminal_cost, scaling)
    end
end

# ============================================================================
# Cost Evaluation Interface (forward declarations)
# ============================================================================

function evaluate_stage_cost end
function evaluate_terminal_cost end
function stage_cost_gradient end
function terminal_cost_gradient end
function stage_cost_hessian end
function terminal_cost_hessian end

# ============================================================================
# DiagonalLQStageCost Implementations
# ============================================================================

function evaluate_stage_cost(cost::DiagonalLQStageCost, x, u, p, t)
    return 0.5 * (dot(cost.qx .* x, x) + dot(cost.ru .* u, u)) + cost.c
end

function stage_cost_gradient(cost::DiagonalLQStageCost, x, u, p, t)
    return (cost.qx .* x, cost.ru .* u)
end

function stage_cost_hessian(cost::DiagonalLQStageCost{T}, x, u, p, t) where {T}
    return (Diagonal(cost.qx), Diagonal(cost.ru), zeros(T, length(x), length(u)))
end

# ============================================================================
# NonlinearStageCost Implementations
# ============================================================================

function evaluate_stage_cost(cost::NonlinearStageCost, x, u, p, t)
    return cost.func(x, u, p, t)
end

function stage_cost_gradient(cost::NonlinearStageCost, x, u, p, t)
    if cost.gradient !== nothing
        return cost.gradient(x, u, p, t)
    else
        return automatic_differentiation_gradient(cost.func, x, u, p, t)
    end
end

function stage_cost_hessian(cost::NonlinearStageCost, x, u, p, t)
    if cost.hessian !== nothing
        return cost.hessian(x, u, p, t)
    else
        return automatic_differentiation_hessian(cost.func, x, u, p, t)
    end
end

# ============================================================================
# DiagonalLQTerminalCost Implementations
# ============================================================================

function evaluate_terminal_cost(cost::DiagonalLQTerminalCost, x, p)
    return 0.5 * dot(cost.qf .* x, x) + cost.cf
end

function terminal_cost_gradient(cost::DiagonalLQTerminalCost, x, p)
    return cost.qf .* x
end

function terminal_cost_hessian(cost::DiagonalLQTerminalCost, x, p)
    return Diagonal(cost.qf)
end

# ============================================================================
# NonlinearTerminalCost Implementations
# ============================================================================

function evaluate_terminal_cost(cost::NonlinearTerminalCost, x, p)
    return cost.func(x, p)
end

function terminal_cost_gradient(cost::NonlinearTerminalCost, x, p)
    if cost.gradient !== nothing
        return cost.gradient(x, p)
    else
        return ForwardDiff.gradient(x_var -> cost.func(x_var, p), x)
    end
end

function terminal_cost_hessian(cost::NonlinearTerminalCost, x, p)
    if cost.hessian !== nothing
        return cost.hessian(x, p)
    else
        return ForwardDiff.hessian(x_var -> cost.func(x_var, p), x)
    end
end

# ============================================================================
# Automatic Differentiation Helpers
# ============================================================================

function automatic_differentiation_gradient(func::F, x, u, p, t) where {F}
    n_x = length(x)
    z   = vcat(x, u)
    ∇z  = ForwardDiff.gradient(z_var -> func(z_var[1:n_x], z_var[n_x+1:end], p, t), z)
    return (∇z[1:n_x], ∇z[n_x+1:end])
end

function automatic_differentiation_hessian(func::F, x, u, p, t) where {F}
    n_x    = length(x)
    z      = vcat(x, u)
    H_full = ForwardDiff.hessian(z_var -> func(z_var[1:n_x], z_var[n_x+1:end], p, t), z)
    return (H_full[1:n_x, 1:n_x], H_full[n_x+1:end, n_x+1:end], H_full[1:n_x, n_x+1:end])
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    is_separable(cost::AbstractStageCost) -> Bool

Returns true if the stage cost depends only on player i's own (xᵢ, uᵢ).
LQStageCost dispatch is defined in problems/GNEP.jl after the struct.
"""
is_separable(::DiagonalLQStageCost) = true
is_separable(c::NonlinearStageCost) = c.is_separable