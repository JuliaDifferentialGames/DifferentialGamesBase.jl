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
# LQ Stage Costs (Analytical Derivatives)
# ============================================================================

"""
    LQStageCost{T} <: AbstractStageCost

Linear-quadratic stage cost: ℓ(x, u) = 1/2 [x; u]ᵀ H [x; u] + [q; r]ᵀ [x; u] + c

# Mathematical Form
```math
ℓ(x, u) = \\frac{1}{2} x^T Q x + \\frac{1}{2} u^T R u + x^T M u + q^T x + r^T u + c
```

Equivalent block matrix form:
```math
H = \\begin{bmatrix} Q & M \\\\ M^T & R \\end{bmatrix}, \\quad
g = \\begin{bmatrix} q \\\\ r \\end{bmatrix}
```

# Fields
- `Q::Matrix{T}`: State cost matrix (n_x × n_x), must be symmetric, Q ⪰ 0
- `R::Matrix{T}`: Control cost matrix (n_u × n_u), must be symmetric, R ≻ 0
- `M::Matrix{T}`: Cross-term matrix (n_x × n_u)
- `q::Vector{T}`: Linear state cost (n_x,)
- `r::Vector{T}`: Linear control cost (n_u,)
- `c::T`: Constant offset (scalar)

# Derivatives (Analytical)
- Gradient: ∇ₓℓ = Qx + Mu + q, ∇ᵤℓ = Rᵤ + Mᵀx + r
- Hessian blocks: Hₓₓ = Q, Hᵤᵤ = R, Hₓᵤ = M

# Notes
- R must be positive definite for player's controls (ensures unique best response)
- Q can be positive semidefinite (allows singular state costs)
- For stacked state x = [x₁; x₂; ...; xₙ], Q is typically sparse (block structure)
"""
struct LQStageCost{T} <: AbstractStageCost
    Q::Matrix{T}
    R::Matrix{T}
    M::Matrix{T}
    q::Vector{T}
    r::Vector{T}
    c::T
    
    function LQStageCost(
        Q::Matrix{T},
        R::Matrix{T},
        M::Matrix{T},
        q::Vector{T},
        r::Vector{T},
        c::T
    ) where {T}
        n_x, n_u = size(M)
        
        # Dimension checks
        @assert size(Q) == (n_x, n_x) "Q must be n_x × n_x"
        @assert size(R) == (n_u, n_u) "R must be n_u × n_u"
        @assert size(M) == (n_x, n_u) "M must be n_x × n_u"
        @assert length(q) == n_x "q must have length n_x"
        @assert length(r) == n_u "r must have length n_u"
        
        # Symmetry checks
        @assert issymmetric(Q) "Q must be symmetric"
        @assert issymmetric(R) "R must be symmetric"
        
        # Positive definiteness checks
        @assert isposdef(R) "R must be positive definite (unique best response requirement)"
        
        # Q can be positive semidefinite (not strictly positive)
        eigvals_Q = eigvals(Symmetric(Q))
        @assert all(eigvals_Q .>= -sqrt(eps(T)) * maximum(abs.(eigvals_Q))) "Q must be positive semidefinite"
        
        new{T}(Q, R, M, q, r, c)
    end
end

# Convenience constructors
function LQStageCost(Q::Matrix{T}, R::Matrix{T}; 
                     M::Union{Matrix{T}, Nothing} = nothing,
                     q::Union{Vector{T}, Nothing} = nothing,
                     r::Union{Vector{T}, Nothing} = nothing,
                     c::T = zero(T)) where {T}
    n_x = size(Q, 1)
    n_u = size(R, 1)
    
    M_mat = M === nothing ? zeros(T, n_x, n_u) : M
    q_vec = q === nothing ? zeros(T, n_x) : q
    r_vec = r === nothing ? zeros(T, n_u) : r
    
    return LQStageCost(Q, R, M_mat, q_vec, r_vec, c)
end

"""
    DiagonalLQStageCost{T} <: AbstractStageCost

Diagonal LQ cost for computational efficiency: ℓ(x, u) = 1/2 xᵀ diag(qx) x + 1/2 uᵀ diag(ru) u + c

# Mathematical Form
```math
ℓ(x, u) = \\frac{1}{2} \\sum_i q_{x,i} x_i^2 + \\frac{1}{2} \\sum_j r_{u,j} u_j^2 + c
```

# Fields
- `qx::Vector{T}`: Diagonal state costs (n_x,), must be non-negative
- `ru::Vector{T}`: Diagonal control costs (n_u,), must be positive
- `c::T`: Constant offset

# Notes
- Much more efficient than full LQStageCost (no matrix storage/multiplication)
- Use when state and control costs are decoupled
- Common in trajectory tracking, obstacle avoidance
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
# Nonlinear Stage Costs (AD Derivatives)
# ============================================================================

"""
    NonlinearStageCost{F, G, H} <: AbstractStageCost

General nonlinear stage cost with optional analytical derivatives.

# Mathematical Form
ℓ(x, u, p, t) -> scalar

# Fields
- `func::F`: Cost function ℓ(x, u, p, t) -> scalar
- `gradient::G`: Optional gradient function (x, u, p, t) -> (∇ₓℓ, ∇ᵤℓ)
- `hessian::H`: Optional Hessian function (x, u, p, t) -> (Hₓₓ, Hᵤᵤ, Hₓᵤ)
- `is_separable::Bool`: True if ℓ(xᵢ, uᵢ) only (enables sparse gradients)

# Gradient Interface
If provided, gradient(x, u, p, t) should return tuple (∇ₓℓ, ∇ᵤℓ) where:
- ∇ₓℓ::Vector{T} is (n_x,) gradient w.r.t. state
- ∇ᵤℓ::Vector{T} is (n_u,) gradient w.r.t. control

# Hessian Interface  
If provided, hessian(x, u, p, t) should return tuple (Hₓₓ, Hᵤᵤ, Hₓᵤ) where:
- Hₓₓ::Matrix{T} is (n_x × n_x) state-state block
- Hᵤᵤ::Matrix{T} is (n_u × n_u) control-control block
- Hₓᵤ::Matrix{T} is (n_x × n_u) cross-term (note: Hᵤₓ = Hₓᵤᵀ)

If Nothing, automatic differentiation will be used.

# Separability
- is_separable = true: Cost depends only on player i's (xᵢ, uᵢ)
  - Enables sparse gradient computation (∇ₓⱼ = 0, ∇ᵤⱼ = 0 for j ≠ i)
  - Common: control effort, state tracking
- is_separable = false: Cost couples multiple agents
  - Requires full gradient computation
  - Common: formation maintenance, consensus

# Parameter p Structure
User should pass named tuple with problem-specific data:
```julia
p = (
    state_indices = [1:6, 7:12, 13:18],   # State slices per player
    control_indices = [1:3, 4:6, 7:9],     # Control slices per player
    player = i,                             # Current player ID
    # ... other user-defined fields
)
```
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

# Convenience constructors
function NonlinearStageCost(
    func::Function;
    gradient::Union{Nothing, Function} = nothing,
    hessian::Union{Nothing, Function} = nothing,
    is_separable::Bool = false
)
    NonlinearStageCost(func, gradient, hessian, is_separable)
end

# ============================================================================
# Terminal Costs
# ============================================================================

"""
    LQTerminalCost{T} <: AbstractTerminalCost

Linear-quadratic terminal cost: φ(x) = 1/2 xᵀ Qf x + qfᵀ x + cf

# Mathematical Form
```math
φ(x) = \\frac{1}{2} x^T Q_f x + q_f^T x + c_f
```

# Fields
- `Qf::Matrix{T}`: Terminal state cost matrix (n_x × n_x), must be symmetric, Qf ⪰ 0
- `qf::Vector{T}`: Linear terminal cost (n_x,)
- `cf::T`: Constant offset

# Notes
- Qf can be singular (positive semidefinite sufficient)
- Common: penalize deviation from target state at final time
"""
struct LQTerminalCost{T} <: AbstractTerminalCost
    Qf::Matrix{T}
    qf::Vector{T}
    cf::T
    
    function LQTerminalCost(Qf::Matrix{T}, qf::Vector{T}, cf::T) where {T}
        n_x = size(Qf, 1)
        
        @assert size(Qf) == (n_x, n_x) "Qf must be square"
        @assert length(qf) == n_x "qf must have length n_x"
        @assert issymmetric(Qf) "Qf must be symmetric"
        
        eigvals_Qf = eigvals(Symmetric(Qf))
        @assert all(eigvals_Qf .>= -sqrt(eps(T)) * maximum(abs.(eigvals_Qf))) "Qf must be positive semidefinite"
        
        new{T}(Qf, qf, cf)
    end
end

# Convenience constructor
LQTerminalCost(Qf::Matrix{T}; qf::Union{Vector{T}, Nothing} = nothing, cf::T = zero(T)) where {T} =
    LQTerminalCost(Qf, qf === nothing ? zeros(T, size(Qf, 1)) : qf, cf)

"""
    DiagonalLQTerminalCost{T} <: AbstractTerminalCost

Diagonal terminal cost: φ(x) = 1/2 xᵀ diag(qf) x + cf

More efficient than full LQTerminalCost for diagonal costs.
"""
struct DiagonalLQTerminalCost{T} <: AbstractTerminalCost
    qf::Vector{T}
    cf::T
    
    function DiagonalLQTerminalCost(qf::Vector{T}, cf::T = zero(T)) where {T}
        @assert all(qf .>= 0) "Diagonal terminal costs must be non-negative"
        new{T}(qf, cf)
    end
end

"""
    NonlinearTerminalCost{F, G, H} <: AbstractTerminalCost

General nonlinear terminal cost with optional analytical derivatives.

# Mathematical Form
φ(x, p) -> scalar

# Fields
- `func::F`: Terminal cost function φ(x, p) -> scalar
- `gradient::G`: Optional gradient function (x, p) -> ∇ₓφ
- `hessian::H`: Optional Hessian function (x, p) -> Hₓₓ

# Note
No time parameter t needed (evaluated once at final time).
No control parameter u (terminal cost independent of final control).
"""
struct NonlinearTerminalCost{F, G, H} <: AbstractTerminalCost
    func::F
    gradient::G
    hessian::H
    
    function NonlinearTerminalCost(func::F, gradient::G, hessian::H) where {F, G, H}
        new{F, G, H}(func, gradient, hessian)
    end
end

# Convenience constructor
function NonlinearTerminalCost(
    func::Function;
    gradient::Union{Nothing, Function} = nothing,
    hessian::Union{Nothing, Function} = nothing
)
    NonlinearTerminalCost(func, gradient, hessian)
end

# ============================================================================
# Player Objective (Bundles Stage + Terminal Costs)
# ============================================================================

"""
    PlayerObjective{S <: AbstractStageCost, T <: AbstractTerminalCost}

Complete objective for a single player in differential game.

# Mathematical Form
```math
J_i = φ_i(x_N) + \\sum_{t=0}^{N-1} ℓ_i(x_t, u_t, p, t)
```

# Fields
- `player_id::Int`: Player index (must be positive)
- `stage_cost::S`: Cost evaluated at each time step
- `terminal_cost::T`: Cost evaluated at final time
- `scaling::Float64`: Optional cost scaling factor (default: 1.0)

# Notes
- Solver aggregates stage costs over trajectory
- User responsible for cost normalization (choose appropriate Q, R magnitudes)
- Use `scaling` parameter for global cost adjustment without modifying matrices
"""
struct PlayerObjective{S <: AbstractStageCost, T <: AbstractTerminalCost}
    player_id::Int
    stage_cost::S
    terminal_cost::T
    scaling::Float64
    
    function PlayerObjective(
        player_id::Int,
        stage_cost::S,
        terminal_cost::T,
        scaling::Float64 = 1.0
    ) where {S <: AbstractStageCost, T <: AbstractTerminalCost}
        @assert player_id > 0 "Player ID must be positive"
        @assert scaling > 0 "Cost scaling must be positive"
        new{S, T}(player_id, stage_cost, terminal_cost, scaling)
    end
end

# ============================================================================
# Cost Evaluation Interface
# ============================================================================

"""
    evaluate_stage_cost(cost::AbstractStageCost, x, u, p, t)

Evaluate stage cost value at time step t.

Returns scalar cost value.
"""
function evaluate_stage_cost end

"""
    evaluate_terminal_cost(cost::AbstractTerminalCost, x, p)

Evaluate terminal cost at final state.

Returns scalar cost value.
"""
function evaluate_terminal_cost end

"""
    stage_cost_gradient(cost::AbstractStageCost, x, u, p, t)

Compute stage cost gradient.

Returns tuple (∇ₓℓ, ∇ᵤℓ) of gradients w.r.t. state and control.
"""
function stage_cost_gradient end

"""
    terminal_cost_gradient(cost::AbstractTerminalCost, x, p)

Compute terminal cost gradient.

Returns ∇ₓφ gradient w.r.t. final state.
"""
function terminal_cost_gradient end

"""
    stage_cost_hessian(cost::AbstractStageCost, x, u, p, t)

Compute stage cost Hessian blocks.

Returns tuple (Hₓₓ, Hᵤᵤ, Hₓᵤ) of Hessian blocks.
"""
function stage_cost_hessian end

"""
    terminal_cost_hessian(cost::AbstractTerminalCost, x, p)

Compute terminal cost Hessian.

Returns Hₓₓ Hessian matrix.
"""
function terminal_cost_hessian end

# ============================================================================
# LQStageCost Implementations (Analytical)
# ============================================================================

function evaluate_stage_cost(cost::LQStageCost, x, u, p, t)
    return 0.5 * (x' * cost.Q * x + u' * cost.R * u + 2 * x' * cost.M * u) +
           cost.q' * x + cost.r' * u + cost.c
end

function stage_cost_gradient(cost::LQStageCost, x, u, p, t)
    ∇x = cost.Q * x + cost.M * u + cost.q
    ∇u = cost.R * u + cost.M' * x + cost.r
    return (∇x, ∇u)
end

function stage_cost_hessian(cost::LQStageCost, x, u, p, t)
    return (cost.Q, cost.R, cost.M)
end

# ============================================================================
# DiagonalLQStageCost Implementations (Optimized)
# ============================================================================

function evaluate_stage_cost(cost::DiagonalLQStageCost, x, u, p, t)
    return 0.5 * (dot(cost.qx .* x, x) + dot(cost.ru .* u, u)) + cost.c
end

function stage_cost_gradient(cost::DiagonalLQStageCost, x, u, p, t)
    ∇x = cost.qx .* x
    ∇u = cost.ru .* u
    return (∇x, ∇u)
end

function stage_cost_hessian(cost::DiagonalLQStageCost{T}, x, u, p, t) where {T}
    Hxx = Diagonal(cost.qx)
    Huu = Diagonal(cost.ru)
    Hxu = zeros(T, length(x), length(u))
    return (Hxx, Huu, Hxu)
end

# ============================================================================
# NonlinearStageCost Implementations (AD Fallback)
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
# LQTerminalCost Implementations (Analytical)
# ============================================================================

function evaluate_terminal_cost(cost::LQTerminalCost, x, p)
    return 0.5 * x' * cost.Qf * x + cost.qf' * x + cost.cf
end

function terminal_cost_gradient(cost::LQTerminalCost, x, p)
    return cost.Qf * x + cost.qf
end

function terminal_cost_hessian(cost::LQTerminalCost, x, p)
    return cost.Qf
end

# ============================================================================
# DiagonalLQTerminalCost Implementations (Optimized)
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
# NonlinearTerminalCost Implementations (AD Fallback)
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
# Automatic Differentiation Helpers for Stage Costs
# ============================================================================

"""
    automatic_differentiation_gradient(func, x, u, p, t)

Compute stage cost gradient using forward-mode AD.

Returns tuple (∇ₓℓ, ∇ᵤℓ).
"""
function automatic_differentiation_gradient(func::F, x, u, p, t) where {F}
    n_x = length(x)
    n_u = length(u)
    
    # Concatenate [x; u] for single gradient call
    z = vcat(x, u)
    
    # Gradient w.r.t. concatenated vector
    ∇z = ForwardDiff.gradient(z_var -> func(z_var[1:n_x], z_var[n_x+1:end], p, t), z)
    
    # Extract blocks
    ∇x = ∇z[1:n_x]
    ∇u = ∇z[n_x+1:end]
    
    return (∇x, ∇u)
end

"""
    automatic_differentiation_hessian(func, x, u, p, t)

Compute stage cost Hessian blocks using forward-mode AD.

Returns tuple (Hₓₓ, Hᵤᵤ, Hₓᵤ).

# Implementation Note
Uses ForwardDiff.hessian on concatenated [x; u] then extracts blocks.
For large problems (n_x + n_u > 100), consider sparse AD if Hessian is sparse.
"""
function automatic_differentiation_hessian(func::F, x, u, p, t) where {F}
    n_x = length(x)
    n_u = length(u)
    
    # Concatenate [x; u]
    z = vcat(x, u)
    
    # Full Hessian
    H_full = ForwardDiff.hessian(z_var -> func(z_var[1:n_x], z_var[n_x+1:end], p, t), z)
    
    # Extract blocks
    Hxx = H_full[1:n_x, 1:n_x]
    Huu = H_full[n_x+1:end, n_x+1:end]
    Hxu = H_full[1:n_x, n_x+1:end]
    
    return (Hxx, Huu, Hxu)
end

# ============================================================================
# Pre-allocated AD Configuration (Optional Performance Optimization)
# ============================================================================

"""
    ForwardDiffCostConfig{N, T}

Pre-allocated configuration for repeated gradient/Hessian evaluations.

Stores dual number cache to avoid allocations in tight loops.

# Usage
```julia
config = ForwardDiffCostConfig(cost, x, u, p, t)
∇x, ∇u = stage_cost_gradient(cost, x, u, p, t, config)
```
"""
struct ForwardDiffCostConfig{N, T}
    chunk_size::Int
    gradient_cache::ForwardDiff.GradientConfig{T, T, N, Vector{ForwardDiff.Dual{T, T, N}}}
    hessian_cache::ForwardDiff.HessianConfig{T, T, N, Vector{ForwardDiff.Dual{T, T, N}}}
end

function ForwardDiffCostConfig(func::F, x::Vector{T}, u::Vector{T}, p, t) where {F, T}
    z = vcat(x, u)
    chunk = ForwardDiff.Chunk(z)
    N = length(chunk)
    
    augmented_func = z_var -> func(z_var[1:length(x)], z_var[length(x)+1:end], p, t)
    
    grad_cache = ForwardDiff.GradientConfig(augmented_func, z, chunk)
    hess_cache = ForwardDiff.HessianConfig(augmented_func, z, chunk)
    
    return ForwardDiffCostConfig{N, T}(N, grad_cache, hess_cache)
end

# Specialized methods using pre-allocated config
function stage_cost_gradient(
    cost::NonlinearStageCost,
    x, u, p, t,
    config::ForwardDiffCostConfig
)
    if cost.gradient !== nothing
        return cost.gradient(x, u, p, t)
    else
        n_x = length(x)
        z = vcat(x, u)
        augmented_func = z_var -> cost.func(z_var[1:n_x], z_var[n_x+1:end], p, t)
        
        ∇z = ForwardDiff.gradient(augmented_func, z, config.gradient_cache)
        
        return (∇z[1:n_x], ∇z[n_x+1:end])
    end
end

function stage_cost_hessian(
    cost::NonlinearStageCost,
    x, u, p, t,
    config::ForwardDiffCostConfig
)
    if cost.hessian !== nothing
        return cost.hessian(x, u, p, t)
    else
        n_x = length(x)
        z = vcat(x, u)
        augmented_func = z_var -> cost.func(z_var[1:n_x], z_var[n_x+1:end], p, t)
        
        H_full = ForwardDiff.hessian(augmented_func, z, config.hessian_cache)
        
        Hxx = H_full[1:n_x, 1:n_x]
        Huu = H_full[n_x+1:end, n_x+1:end]
        Hxu = H_full[1:n_x, n_x+1:end]
        
        return (Hxx, Huu, Hxu)
    end
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    total_cost(obj::PlayerObjective, X::Vector, U::Vector, p)

Compute total trajectory cost for player.

# Arguments
- `X::Vector`: State trajectory [x₀, x₁, ..., xₙ]
- `U::Vector`: Control trajectory [u₀, u₁, ..., uₙ₋₁]
- `p`: Problem parameters

Returns scalar total cost J = Σₜ ℓ(xₜ, uₜ) + φ(xₙ)

# Note
Useful for debugging and logging. Solvers typically evaluate incrementally.
"""
function total_cost(obj::PlayerObjective, X::Vector, U::Vector, p)
    N = length(U)
    @assert length(X) == N + 1 "X must have length N+1 for N control steps"
    
    stage_sum = sum(evaluate_stage_cost(obj.stage_cost, X[t], U[t], p, t-1) for t in 1:N)
    terminal = evaluate_terminal_cost(obj.terminal_cost, X[end], p)
    
    return obj.scaling * (stage_sum + terminal)
end

"""
    diagnose_scaling(obj::PlayerObjective, X::Vector, U::Vector, p)

Analyze cost scaling for debugging ill-conditioned problems.

Returns named tuple with statistics:
- `stage_costs`: Vector of individual stage costs
- `terminal_cost`: Final cost value
- `stage_mean`, `stage_std`: Stage cost statistics
- `stage_to_terminal_ratio`: Relative magnitudes

# Recommendation
Stage and terminal costs should be similar order of magnitude.
If ratio >> 10 or << 0.1, consider rescaling Q, R, Qf matrices.
"""
function diagnose_scaling(obj::PlayerObjective, X::Vector, U::Vector, p)
    N = length(U)
    
    stage_costs = [evaluate_stage_cost(obj.stage_cost, X[t], U[t], p, t-1) for t in 1:N]
    terminal_cost = evaluate_terminal_cost(obj.terminal_cost, X[end], p)
    
    stage_sum = sum(stage_costs)
    stage_mean = stage_sum / N
    stage_std = std(stage_costs)
    
    return (
        stage_costs = stage_costs,
        terminal_cost = terminal_cost,
        stage_mean = stage_mean,
        stage_std = stage_std,
        stage_min = minimum(stage_costs),
        stage_max = maximum(stage_costs),
        stage_to_terminal_ratio = stage_sum / (terminal_cost + eps()),
        total_cost = obj.scaling * (stage_sum + terminal_cost)
    )
end

"""
    is_separable(cost::AbstractStageCost)

Check if stage cost is separable (depends only on player i's variables).

Returns true for DiagonalLQStageCost and NonlinearStageCost with is_separable=true.
"""
is_separable(::DiagonalLQStageCost) = true
is_separable(::LQStageCost) = false  # General LQ can have coupling through Q
is_separable(cost::NonlinearStageCost) = cost.is_separable