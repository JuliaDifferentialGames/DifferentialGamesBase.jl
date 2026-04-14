# ============================================================================
# trajectory_expansion.jl
#
# Phase 3: trajectory linearization and quadraticization infrastructure.
#
# Provides three types and three functions:
#
#   DynamicsExpansion{T}    — per-timestep (A_k, B_k, c_k) with optional block structure
#   CostExpansion{T}        — per-timestep, per-player (Q, R, M, q, r, ℓ) + terminal
#   TrajectoryExpansion{T}  — bundles both with the reference (X, U)
#
#   expand(game, X, U, da)  → TrajectoryExpansion{T}
#   assemble_lq_game(exp)   → GameProblem{T} with LinearDynamics + LQStageCost
#   reference_trajectory(game, strategy, da) → (X, U, times)
#
# Architecture rationale:
#   iLQGames / FNELQ consume assemble_lq_game() → GameProblem.
#   FALCON / SQP-DG consume TrajectoryExpansion directly (raw matrices).
#   The block structure in DynamicsExpansion is exploited only when the
#   underlying dynamics are SeparableDynamics; other callers use A_full.
#
# This file lives in DifferentialGamesBaseSolvers (solver infrastructure),
# not in DifferentialGamesBase (problem specification). It depends on:
#   DifferentialGamesBase: GameProblem, DynamicsSpec, DiscreteApproximation,
#   evaluate_dynamics, dynamics_jacobian, rollout, rollout_strategy, da_step,
#   jacobian (on DiscreteApproximation), LQGameProblem, LQStageCost, etc.
# ============================================================================

using LinearAlgebra
using ForwardDiff
using SparseArrays

# ============================================================================
# DynamicsExpansion{T}
# ============================================================================

"""
    DynamicsExpansion{T}

First-order expansion of dynamics along a reference trajectory:
    x(k+1) ≈ Aₖ x(k) + Bₖ u(k) + cₖ

where cₖ = x_ref(k+1) - Aₖ x_ref(k) - Bₖ u_ref(k) is the affine defect
(zero when the reference trajectory is dynamically feasible).

# Fields
- `A_full`   : `Vector{Matrix{T}}` length N, A_full[k] is (n × n) at step k
- `B_full`   : `Vector{Matrix{T}}` length N, B_full[k] is (n × m_total) at step k
- `c`        : `Vector{Vector{T}}` length N, affine defect per step
- `A_blocks` : `Vector{Vector{Matrix{T}}}` or `nothing`
               If not nothing: A_blocks[i][k] is (nᵢ × nᵢ) block for player i at step k.
               Only populated for `SeparableDynamics` — exploits block-diagonal structure.
- `B_blocks` : `Vector{Vector{Matrix{T}}}` or `nothing`
               If not nothing: B_blocks[i][k] is (nᵢ × mᵢ) block for player i at step k.
- `n`, `m`, `N` : state dim, total control dim, horizon length
- `is_separable` : true iff block structure is populated

# Usage
Solver-agnostic consumers use `A_full[k]`, `B_full[k]`.
PD-GNEP solvers (FALCON, SQP-DG) use `A_blocks[i][k]`, `B_blocks[i][k]`
to exploit the block-diagonal sparsity of separable dynamics, avoiding
O(n²) operations on off-diagonal zeros.
"""
struct DynamicsExpansion{T}
    A_full::Vector{Matrix{T}}
    B_full::Vector{Matrix{T}}
    c::Vector{Vector{T}}

    # Block structure — only for SeparableDynamics
    # A_blocks[i][k] : (nᵢ × nᵢ) — ∂fᵢ/∂xᵢ at step k
    # B_blocks[i][k] : (nᵢ × mᵢ) — ∂fᵢ/∂uᵢ at step k
    A_blocks::Union{Nothing, Vector{Vector{Matrix{T}}}}
    B_blocks::Union{Nothing, Vector{Vector{Matrix{T}}}}

    n::Int
    m::Int
    N::Int
    is_separable::Bool

    function DynamicsExpansion{T}(
        A_full, B_full, c,
        A_blocks, B_blocks,
        n, m, N, is_separable
    ) where {T}
        @assert length(A_full) == N
        @assert length(B_full) == N
        @assert length(c)      == N
        new{T}(A_full, B_full, c, A_blocks, B_blocks, n, m, N, is_separable)
    end
end

# ============================================================================
# CostExpansion{T}
# ============================================================================

"""
    CostExpansion{T}

Second-order expansion of all players' costs along a reference trajectory.

For each player i and timestep k, stores the full quadratic approximation
of the stage cost around (x_ref(k), u_ref(k)):

    ℓᵢ(x, uᵢ) ≈ ½ δxᵀ Hxx[i][k] δx  +  δuᵢᵀ Huu[i][k] δuᵢ
               +  δxᵀ Hxu[i][k] δuᵢ   +  gx[i][k]ᵀ δx  +  gu[i][k]ᵀ δuᵢ
               +  ℓ_val[i][k]

The Hessian and gradient are w.r.t. the FULL joint state x (dimension n),
and player i's control uᵢ (dimension mᵢ). This is the general form needed
by FALCON and SQP-DG. FNELQ and iLQGames access the same fields through
`assemble_lq_game`, which extracts (Hxx, Huu, Hxu) as (Q, R, M).

# Fields
- `Hxx`    : `Vector{Vector{Matrix{T}}}` — Hxx[i][k] is (n × n)
- `Huu`    : `Vector{Vector{Matrix{T}}}` — Huu[i][k] is (mᵢ × mᵢ)
- `Hxu`    : `Vector{Vector{Matrix{T}}}` — Hxu[i][k] is (n × mᵢ)
- `gx`     : `Vector{Vector{Vector{T}}}` — gx[i][k] is (n,)
- `gu`     : `Vector{Vector{Vector{T}}}` — gu[i][k] is (mᵢ,)
- `ℓ_val`  : `Vector{Vector{T}}`         — ℓ_val[i][k] is scalar

Terminal cost (at step N+1):
- `Hxx_f`  : `Vector{Matrix{T}}` — Hxx_f[i] is (n × n)
- `gx_f`   : `Vector{Vector{T}}` — gx_f[i] is (n,)
- `ℓ_f`    : `Vector{T}`         — ℓ_f[i] is scalar

# Notes
All Hessian blocks are regularised to be PSD by the expansion routines —
indefinite second-order terms (from non-convex costs) are projected to the
PSD cone via `max(eigval, 0)` regularisation. This ensures the assembled LQ
game is solvable by FNELQ even when the original cost is non-convex.
"""
struct CostExpansion{T}
    Hxx::Vector{Vector{Matrix{T}}}
    Huu::Vector{Vector{Matrix{T}}}
    Hxu::Vector{Vector{Matrix{T}}}
    gx::Vector{Vector{Vector{T}}}
    gu::Vector{Vector{Vector{T}}}
    ℓ_val::Vector{Vector{T}}

    # Terminal
    Hxx_f::Vector{Matrix{T}}
    gx_f::Vector{Vector{T}}
    ℓ_f::Vector{T}

    n_players::Int
    n::Int
    control_dims::Vector{Int}
    N::Int
end

# ============================================================================
# TrajectoryExpansion{T}
# ============================================================================

"""
    TrajectoryExpansion{T}

Complete first- and second-order expansion of a game along a reference
trajectory. The primary output of `expand`.

# Fields
- `dynamics`  : `DynamicsExpansion{T}`
- `costs`     : `CostExpansion{T}`
- `X`         : `Matrix{T}` (n × N+1) — reference state trajectory
- `U`         : `Matrix{T}` (m × N)   — reference control trajectory
- `times`     : `Vector{T}` (N+1,)    — time vector

# Usage
```julia
# Build a reference trajectory and expand around it
X0, U0 = reference_trajectory(game, zero_strategy, da)
exp    = expand(game, X0, U0, da)

# Use directly in FALCON/SQP
Aₖ = exp.dynamics.A_full[k]
Qᵢₖ = exp.costs.Hxx[i][k]

# Or assemble an LQ game for FNELQ/iLQGames
lq_game = assemble_lq_game(exp, game)
sol = solve(lq_game, FNELQ())
```
"""
struct TrajectoryExpansion{T}
    dynamics::DynamicsExpansion{T}
    costs::CostExpansion{T}
    X::Matrix{T}
    U::Matrix{T}
    times::Vector{T}
end

# ============================================================================
# linearize_dynamics — DynamicsExpansion from a trajectory
# ============================================================================

"""
    linearize_dynamics(dyn, X, U, da, times) -> DynamicsExpansion{T}

Linearize `dyn` along the reference trajectory `(X, U)` at each timestep.

Uses `jacobian(da, ...)` on the `DiscreteApproximation` — NOT `dynamics_jacobian`.
This ensures the linearization is consistent with the discrete map used in
the solver's constraint layer, preventing the artificial infeasibility floor.

# Arguments
- `dyn`   : `DynamicsSpec{T}`
- `X`     : (n × N+1) reference state trajectory
- `U`     : (m × N)   reference control trajectory
- `da`    : `DiscreteApproximation` — must match `dyn`
- `times` : (N+1,) time vector

# Returns
`DynamicsExpansion{T}` with:
- `A_full`, `B_full` : dense Jacobians at each step (always populated)
- `A_blocks`, `B_blocks` : block-diagonal factors (only for `SeparableDynamics`)
- `c` : affine defect x_ref(k+1) - Aₖ x_ref(k) - Bₖ u_ref(k)

# Sparsity
For `SeparableDynamics`, the full Jacobian `∂f/∂x` is block-diagonal by
construction. Rather than extracting blocks from the dense matrix (which
would require threshold-based sparsity detection), we re-evaluate each
player's dynamics Jacobian independently at their private state/control slice.
This gives exact block structure at the cost of n_players ForwardDiff calls
instead of one — generally worthwhile for n_players ≥ 2.
"""
function linearize_dynamics(
    dyn::DynamicsSpec{T},
    X::AbstractMatrix,
    U::AbstractMatrix,
    da::DiscreteApproximation,
    times::AbstractVector
) where {T}
    n = total_state_dim(dyn)
    m = total_control_dim(dyn)
    N = size(U, 2)

    A_full = Vector{Matrix{T}}(undef, N)
    B_full = Vector{Matrix{T}}(undef, N)
    c      = Vector{Vector{T}}(undef, N)

    for k in 1:N
        xk = X[:, k]; uk = U[:, k]
        t  = _step_time(times, k, dyn)
        Ak, Bk = jacobian(da, xk, uk, nothing, t)
        A_full[k] = Ak
        B_full[k] = Bk
        # Affine defect: how far the reference trajectory deviates from the
        # linearized model. Zero for dynamically feasible trajectories.
        c[k] = X[:, k+1] .- Ak * xk .- Bk * uk
    end

    # Block structure for SeparableDynamics
    A_blocks, B_blocks = _compute_block_jacobians(dyn, X, U, times, N, T)

    return DynamicsExpansion{T}(
        A_full, B_full, c,
        A_blocks, B_blocks,
        n, m, N,
        dyn isa SeparableDynamics
    )
end

# Discrete block Jacobians for SeparableDynamics.
#
# A_blocks[i][k] = ∂xᵢ(k+1)/∂xᵢ(k) — discrete-map Jacobian for player i.
# B_blocks[i][k] = ∂xᵢ(k+1)/∂uᵢ(k) — discrete-map control Jacobian.
#
# These are obtained by differentiating through _rk4_step applied to player
# i's subsystem only, using the private state/control slices. This gives the
# exact same value as the corresponding block of A_full/B_full (since the
# dynamics are separable, cross-player terms are zero), but avoids extracting
# blocks from the dense matrix — exact by construction, no numerical threshold.
#
# Note: this differs from differentiating the continuous RHS fi directly,
# which gives ∂ẋᵢ/∂xᵢ (continuous Jacobian) — a different quantity.
function _compute_block_jacobians(
    dyn::SeparableDynamics{T},
    X, U, times, N, ::Type{T}
) where {T}
    np      = length(dyn.player_dynamics)
    s_offs  = [0; cumsum(dyn.state_dims)]
    c_offs  = [0; cumsum(dyn.control_dims)]

    A_blocks = [[Matrix{T}(undef, dyn.state_dims[i], dyn.state_dims[i]) for _ in 1:N]
                for i in 1:np]
    B_blocks = [[Matrix{T}(undef, dyn.state_dims[i], dyn.control_dims[i]) for _ in 1:N]
                for i in 1:np]

    for k in 1:N
        t  = times[k]
        dt = times[k+1] - times[k]
        for i in 1:np
            sr  = s_offs[i]+1:s_offs[i+1]
            cr  = c_offs[i]+1:c_offs[i+1]
            fi  = dyn.player_dynamics[i]
            xi  = X[sr, k]; ui = U[cr, k]
            ni  = dyn.state_dims[i]; mi = dyn.control_dims[i]
            # Wrap player i's dynamics in a CoupledNonlinearDynamics-compatible
            # closure and differentiate through _rk4_step
            fi_rhs = (xx, uu, p, tt) -> fi(xx, uu, p, tt)
            J = ForwardDiff.jacobian(
                z -> _rk4_step_fn(fi_rhs, z[1:ni], z[ni+1:end], nothing, t, dt),
                vcat(xi, ui)
            )
            A_blocks[i][k] = J[:, 1:ni]
            B_blocks[i][k] = J[:, ni+1:end]
        end
    end
    return A_blocks, B_blocks
end

function _compute_block_jacobians(
    ::Union{LinearDynamics, CoupledNonlinearDynamics},
    X, U, times, N, ::Type{T}
) where {T}
    return nothing, nothing
end

# Pure-function RK4 step for a player's subsystem.
# Takes the RHS as a plain function rather than a DynamicsSpec so that
# ForwardDiff can propagate dual numbers through it cleanly.
@inline function _rk4_step_fn(f::F, x, u, p, t, dt) where {F}
    k1 = f(x,             u, p, t)
    k2 = f(x + dt/2 * k1, u, p, t + dt/2)
    k3 = f(x + dt/2 * k2, u, p, t + dt/2)
    k4 = f(x + dt    * k3, u, p, t + dt)
    return x + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# ============================================================================
# quadraticize_costs — CostExpansion from a trajectory
# ============================================================================

"""
    quadraticize_costs(game, X, U, times) -> CostExpansion{T}

Quadraticize all players' stage and terminal costs along `(X, U)`.

For each player i and timestep k, computes the second-order Taylor expansion
of ℓᵢ(x, uᵢ) around (X[:,k], Uᵢ[:,k]) using ForwardDiff.

For LQStageCost objectives the expansion is exact (analytical Hessians are
used directly via `stage_cost_hessian`, avoiding ForwardDiff overhead).
For NonlinearStageCost and cost-term DSL objectives, ForwardDiff is used.

# Regularisation
Hxx and Huu blocks are symmetrised and projected to the PSD cone:
    H_reg = (H + Hᵀ)/2  then  H_reg += max(-λ_min, 0)·I
This ensures the assembled LQ game is solvable regardless of local
non-convexity in the original cost, at the price of a first-order error
in non-convex regions. Callers that need the raw (possibly indefinite)
Hessian should set `regularize=false`.
"""
function quadraticize_costs(
    game::GameProblem{T},
    X::AbstractMatrix,
    U::AbstractMatrix,
    times::AbstractVector;
    regularize::Bool = true
) where {T}
    np           = num_players(game)
    n            = total_state_dim(game.dynamics)
    control_dims = game.dynamics.control_dims
    state_dims   = game.metadata.state_dims
    c_offs       = [0; cumsum(control_dims)]
    # State offsets per player: for shared-state games all are 0;
    # for PD-GNEP each player has a private state block.
    s_offs = if game.dynamics isa SeparableDynamics
        [0; cumsum(game.dynamics.state_dims)[1:end-1]]
    else
        zeros(Int, np)
    end
    N            = size(U, 2)

    Hxx   = [[zeros(T, n, n)   for _ in 1:N] for _ in 1:np]
    Huu   = [[zeros(T, control_dims[i], control_dims[i]) for _ in 1:N] for i in 1:np]
    Hxu   = [[zeros(T, n, control_dims[i]) for _ in 1:N] for i in 1:np]
    gx    = [[zeros(T, n)              for _ in 1:N] for _ in 1:np]
    gu    = [[zeros(T, control_dims[i]) for _ in 1:N] for i in 1:np]
    ℓ_val = [[zero(T) for _ in 1:N] for _ in 1:np]
    Hxx_f = [zeros(T, n, n)   for _ in 1:np]
    gx_f  = [zeros(T, n)      for _ in 1:np]
    ℓ_f   = zeros(T, np)

    for i in 1:np
        obj = get_objective(game, i)
        mi  = control_dims[i]
        coi = c_offs[i]     # control offset for player i
        soi = s_offs[i]     # state offset for player i
        n_xi = game.dynamics isa SeparableDynamics ?
               game.dynamics.state_dims[i] : n

        for k in 1:N
            xk  = X[:, k]
            uk  = U[:, k]
            uik = uk[coi+1:coi+mi]    # player i's control slice
            xik = xk[soi+1:soi+n_xi]  # player i's state slice
            tk  = Int(k)

            # Stage cost Hessians
            Hxxk, Huuk, Hxuk = _stage_hessian(obj.stage_cost, xk, xik, uik, soi, n_xi, mi, n, nothing, tk)
            gxk, guk         = _stage_gradient(obj.stage_cost, xk, xik, uik, soi, n_xi, mi, n, nothing, tk)
            ℓk               = evaluate_stage_cost(obj.stage_cost, xik, uik, nothing, tk)

            Hxx[i][k]   = regularize ? _psd_project(Hxxk) : Hxxk
            Huu[i][k]   = regularize ? _psd_project(Huuk) : Huuk
            Hxu[i][k]   = Hxuk
            gx[i][k]    = gxk
            gu[i][k]    = guk
            ℓ_val[i][k] = ℓk
        end

        # Terminal cost
        xN   = X[:, N+1]
        xiN  = xN[soi+1:soi+n_xi]
        Hxx_fk = _terminal_hessian(obj.terminal_cost, xiN, soi, n_xi, n, nothing)
        gx_fk  = _terminal_gradient(obj.terminal_cost, xiN, soi, n_xi, n, nothing)
        ℓ_fk   = evaluate_terminal_cost(obj.terminal_cost, xiN, nothing)

        Hxx_f[i] = regularize ? _psd_project(Hxx_fk) : Hxx_fk
        gx_f[i]  = gx_fk
        ℓ_f[i]   = ℓ_fk
    end

    return CostExpansion{T}(
        Hxx, Huu, Hxu, gx, gu, ℓ_val,
        Hxx_f, gx_f, ℓ_f,
        np, n, control_dims, N
    )
end

# ============================================================================
# Stage Hessian dispatch — analytical for LQ, ForwardDiff otherwise
#
# Signature: (cost, x_full, xi, ui, soi, n_xi, mi, n, p, t)
#   x_full : full joint state (for nonlinear costs via evaluate_stage_cost)
#   xi     : player i's state slice (for LQ analytical path)
#   ui     : player i's control
#   soi    : zero-based state offset for player i in x_full
#   n_xi   : player i's state dimension
#   mi     : player i's control dimension
#   n      : total joint state dimension
# ============================================================================

function _stage_hessian(
    cost::LQStageCost,
    x_full, xi, ui, soi::Int, n_xi::Int, mi::Int, n::Int, p, t::Int
)
    # stage_cost_hessian for LQStageCost returns (Q, R, M) — matrices only,
    # independent of (x, u). Pass xi and ui for signature compatibility.
    Qk, Rk, Mk = stage_cost_hessian(cost, xi, ui, p, t)
    # Embed the (n_xi × n_xi) Hessian block at player i's state offset.
    Hxx = zeros(eltype(x_full), n, n)
    Hxx[soi+1:soi+n_xi, soi+1:soi+n_xi] = Qk
    Hxu = zeros(eltype(x_full), n, mi)
    Hxu[soi+1:soi+n_xi, :] = Mk
    return Hxx, Rk, Hxu
end

function _stage_hessian(
    cost::AbstractStageCost,
    x_full, xi, ui, soi::Int, n_xi::Int, mi::Int, n::Int, p, t
)
    # ForwardDiff through evaluate_stage_cost with the player's state slice.
    # evaluate_stage_cost receives (xi, ui) — not the joint state — for
    # NonlinearStageCost and cost-term DSL objectives.
    z = vcat(xi, ui)
    H = ForwardDiff.hessian(
        z_var -> evaluate_stage_cost(cost, z_var[1:n_xi], z_var[n_xi+1:end], p, t),
        z
    )
    # Embed blocks into joint dimensions
    Hxx = zeros(eltype(x_full), n, n)
    Hxx[soi+1:soi+n_xi, soi+1:soi+n_xi] = H[1:n_xi, 1:n_xi]
    Hxu = zeros(eltype(x_full), n, mi)
    Hxu[soi+1:soi+n_xi, :] = H[1:n_xi, n_xi+1:end]
    Huu = H[n_xi+1:end, n_xi+1:end]
    return Hxx, Huu, Hxu
end

# ============================================================================
# Stage Gradient dispatch
# ============================================================================

function _stage_gradient(
    cost::LQStageCost,
    x_full, xi, ui, soi::Int, n_xi::Int, mi::Int, n::Int, p, t::Int
)
    ∇xi, ∇ui = stage_cost_gradient(cost, xi, ui, p, t)
    ∇x = zeros(eltype(x_full), n)
    ∇x[soi+1:soi+n_xi] = ∇xi
    return ∇x, ∇ui
end

function _stage_gradient(
    cost::AbstractStageCost,
    x_full, xi, ui, soi::Int, n_xi::Int, mi::Int, n::Int, p, t
)
    z = vcat(xi, ui)
    g = ForwardDiff.gradient(
        z_var -> evaluate_stage_cost(cost, z_var[1:n_xi], z_var[n_xi+1:end], p, t),
        z
    )
    ∇x = zeros(eltype(x_full), n)
    ∇x[soi+1:soi+n_xi] = g[1:n_xi]
    return ∇x, g[n_xi+1:end]
end

# ============================================================================
# Terminal Hessian and Gradient dispatch
#
# Signature: (cost, xi, soi, n_xi, n, p)
#   xi   : player i's terminal state slice
#   soi  : zero-based state offset
#   n_xi : player i's state dimension
#   n    : total joint state dimension
# ============================================================================

function _terminal_hessian(cost::LQTerminalCost, xi, soi::Int, n_xi::Int, n::Int, p)
    Hxi = terminal_cost_hessian(cost, xi, p)    # (n_xi × n_xi)
    Hxx = zeros(eltype(xi), n, n)
    Hxx[soi+1:soi+n_xi, soi+1:soi+n_xi] = Hxi
    return Hxx
end

function _terminal_hessian(cost::AbstractTerminalCost, xi, soi::Int, n_xi::Int, n::Int, p)
    Hxi = ForwardDiff.hessian(x_var -> evaluate_terminal_cost(cost, x_var, p), xi)
    Hxx = zeros(eltype(xi), n, n)
    Hxx[soi+1:soi+n_xi, soi+1:soi+n_xi] = Hxi
    return Hxx
end

function _terminal_gradient(cost::LQTerminalCost, xi, soi::Int, n_xi::Int, n::Int, p)
    ∇xi = terminal_cost_gradient(cost, xi, p)
    ∇x  = zeros(eltype(xi), n)
    ∇x[soi+1:soi+n_xi] = ∇xi
    return ∇x
end

function _terminal_gradient(cost::AbstractTerminalCost, xi, soi::Int, n_xi::Int, n::Int, p)
    ∇xi = ForwardDiff.gradient(x_var -> evaluate_terminal_cost(cost, x_var, p), xi)
    ∇x  = zeros(eltype(xi), n)
    ∇x[soi+1:soi+n_xi] = ∇xi
    return ∇x
end

# ============================================================================
# PSD projection — ensures assembled LQ game is solvable
# ============================================================================

"""
    _psd_project(H::Matrix{T}) -> Matrix{T}

Project H to the positive semidefinite cone: symmetrise, then clamp
negative eigenvalues to zero. O(n³) but called only during expansion,
not in the inner QP loop.
"""
function _psd_project(H::Matrix{T}) where {T}
    Hs = (H + H') / 2
    F  = eigen(Symmetric(Hs))
    λ  = max.(F.values, zero(T))
    return F.vectors * Diagonal(λ) * F.vectors'
end

# ============================================================================
# expand — main entry point
# ============================================================================

"""
    expand(game, X, U, da; regularize=true) -> TrajectoryExpansion{T}

Linearize dynamics and quadraticize costs along the reference trajectory
`(X, U)`, returning a `TrajectoryExpansion` for use by solvers.

This is the single entry point for all second-order solver infrastructure.
Internally calls `linearize_dynamics` and `quadraticize_costs`.

# Arguments
- `game`       : `GameProblem{T}`
- `X`          : (n × N+1) reference state trajectory (from `rollout`)
- `U`          : (m × N)   reference control trajectory
- `da`         : `DiscreteApproximation` — must be consistent with `game.dynamics`
- `regularize` : Project Hessian blocks to PSD cone (default true)

# Returns
`TrajectoryExpansion{T}`.

# Example
```julia
times = collect(range(0.0, game.time_horizon.tf, length=N+1))
da    = discretize(game.dynamics, game.time_horizon.dt)
X0    = rollout(game.dynamics, game.initial_state, zeros(m, N), nothing, times)
U0    = zeros(m, N)
exp   = expand(game, X0, U0, da)
```
"""
function expand(
    game::GameProblem{T},
    X::AbstractMatrix,
    U::AbstractMatrix,
    da::DiscreteApproximation;
    regularize::Bool = true
) where {T}
    N     = size(U, 2)
    N_th  = n_steps(game)
    @assert(N == N_th, "U has $N columns but game has $N_th steps")

    th    = game.time_horizon
    times = collect(range(zero(T), th.tf, length=N+1))

    dyn_exp  = linearize_dynamics(game.dynamics, X, U, da, times)
    cost_exp = quadraticize_costs(game, X, U, times; regularize)

    return TrajectoryExpansion{T}(dyn_exp, cost_exp, Matrix{T}(X), Matrix{T}(U), times)
end

# ============================================================================
# assemble_lq_game — TrajectoryExpansion → GameProblem (LTV LQ)
# ============================================================================

"""
    assemble_lq_game(exp, game) -> GameProblem{T}

Assemble an LTV LQ game from a `TrajectoryExpansion`. The result has:
- `LinearDynamics` with LTV A_seq, B_seq from `exp.dynamics`
- `LQStageCost` (LTV) with Hxx/Huu/Hxu/gx/gu per player per step
- `LQTerminalCost` with Hxx_f/gx_f per player
- Same initial state, constraints, and metadata as `game`

Used by FNELQ (iLQGames inner loop) and any solver requiring a linear-
quadratic game. FALCON and SQP-DG use `TrajectoryExpansion` directly.

# Notes
The assembled game is an LTV game over the same horizon as `game`.
`is_lq_game(assembled)` returns true. `validate_game_problem` passes.
"""
function assemble_lq_game(
    exp::TrajectoryExpansion{T},
    game::GameProblem{T}
) where {T}
    N  = exp.dynamics.N
    n  = exp.dynamics.n
    np = num_players(game)
    control_dims = game.dynamics.control_dims
    c_offs       = [0; cumsum(control_dims)]

    # Build LTV LinearDynamics from A_full, B_full
    # B is stored as B_seq[i][k] = B_full[k][:, c_offs[i]+1:c_offs[i+1]]
    A_seq = exp.dynamics.A_full
    B_seq = [
        [exp.dynamics.B_full[k][:, c_offs[i]+1:c_offs[i+1]] for k in 1:N]
        for i in 1:np
    ]
    dyn_lq = LinearDynamics(A_seq, B_seq)

    # Build LTV LQStageCost and LQTerminalCost per player
    objectives = map(1:np) do i
        original_obj = get_objective(game, i)
        stage_cost = LQStageCost(
            exp.costs.Hxx[i],    # Q_seq  : (n×n per step)
            exp.costs.Huu[i],    # R_seq  : (mᵢ×mᵢ per step)
            exp.costs.Hxu[i],    # M_seq  : (n×mᵢ per step)
            exp.costs.gx[i],     # q_seq  : (n per step)
            exp.costs.gu[i]      # r_seq  : (mᵢ per step)
        )
        terminal_cost = LQTerminalCost(
            exp.costs.Hxx_f[i],
            exp.costs.gx_f[i]
        )
        PlayerObjective(i, stage_cost, terminal_cost, original_obj.scaling)
    end

    # Rebuild metadata for the new LinearDynamics
    state_offsets   = [0]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    cost_coupling   = sparse(trues(np, np))
    coupling_graph  = CouplingGraph(cost_coupling, Vector{Int}[], nothing)
    metadata = GameMetadata(
        [n], control_dims, state_offsets, control_offsets,
        coupling_graph, false, nothing
    )

    return GameProblem{T}(
        np,
        objectives,
        dyn_lq,
        game.initial_state,
        game.private_constraints,
        game.shared_constraints,
        game.time_horizon,
        metadata
    )
end

# ============================================================================
# reference_trajectory — warm-start helper for solvers
# ============================================================================

"""
    reference_trajectory(game, strategy, da) -> (X, U, times)

Produce a reference trajectory by rolling out `strategy` against `game.dynamics`.
Wraps `rollout_strategy` with the time vector from `game.time_horizon`.

This is the standard entry point for warm-starting iterative solvers:
```julia
X, U, times = reference_trajectory(game, zero_strategy, da)
for iter in 1:max_iter
    exp      = expand(game, X, U, da)
    lq_game  = assemble_lq_game(exp, game)
    sol      = solve(lq_game, FNELQ())
    strategy = get_strategy(sol)
    X, U, _  = reference_trajectory(game, strategy, da)
end
```

# Arguments
- `game`     : `GameProblem{T}`
- `strategy` : `AbstractStrategy{T}` — open-loop or feedback
- `da`       : unused here (rollout uses the game's dynamics directly), retained
               for API consistency with `expand`

# Returns
`(X, U, times)` — (n×N+1), (m×N), (N+1,)
"""
function reference_trajectory(
    game::GameProblem{T},
    strategy::AbstractStrategy{T},
    da::DiscreteApproximation
) where {T}
    th    = game.time_horizon
    N     = n_steps(game)
    times = collect(range(zero(T), th.tf, length=N+1))
    X, U  = rollout_strategy(game.dynamics, game.initial_state, strategy, nothing)
    return X, U, times
end

# Zero warm start — all controls zero
function reference_trajectory(
    game::GameProblem{T},
    ::Nothing,
    da::DiscreteApproximation
) where {T}
    th    = game.time_horizon
    N     = n_steps(game)
    m     = total_control_dim(game.dynamics)
    times = collect(range(zero(T), th.tf, length=N+1))
    U0    = zeros(T, m, N)
    X0    = rollout(game.dynamics, game.initial_state, U0, nothing, times)
    return X0, U0, times
end

# ============================================================================
# Utilities
# ============================================================================

# Return the correct time argument for a dynamics step:
# LinearDynamics uses Int timestep index; continuous dynamics use Real time.
@inline function _step_time(times::AbstractVector, k::Int, ::LinearDynamics)
    return k
end

@inline function _step_time(times::AbstractVector, k::Int, ::DynamicsSpec)
    return times[k]
end