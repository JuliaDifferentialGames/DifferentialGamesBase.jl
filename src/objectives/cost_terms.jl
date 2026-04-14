using ForwardDiff
using LinearAlgebra

# ============================================================================
# AbstractCostTerm + composition via +
#
# Enables the cost DSL:
#   stage = track_goal(xg, Q) + regularize_input(R) + avoid_proximity(j, d, w)
#   obj   = minimize(stage, terminal=track_goal(xg, Qf), player_id=1)
#
# Cross-agent coupling in costs:
#
#   Per-player terms (AbstractCostTerm):
#     evaluate_cost_term(term, x, u, p, t) where x is the FULL joint state.
#     A separable term only reads xᵢ from x via its stored index slice.
#     A coupling term reads both xᵢ and xⱼ from x.
#     This is the correct design — costs always receive the joint state;
#     the term decides what to look at. DifferentialGame passes joint x.
#
#   This avoids the need for a separate CoupledCostTerm type: coupling is
#   expressed naturally by terms that read multiple player slices from x.
#   The `player_slice` helper makes this ergonomic (see below).
#
# File location: src/objectives/cost_terms.jl
# Standard concrete terms: src/objectives/standard_costs.jl
# ============================================================================

# ============================================================================
# Abstract types
# ============================================================================

"""
    AbstractCostTerm

A single additive term in a player's stage cost.

# Call Signature
```
evaluate_cost_term(term, x, u, p, t) -> scalar
```
where `x` is the **full joint state** vector and `u` is the **full joint
control** vector. Terms that depend only on player i's slice are separable;
terms that read multiple players' slices encode cross-agent coupling.

# Separation vs Coupling
The full joint `(x, u)` is always passed. A separable term ignores other
players' slices; a coupling term reads them explicitly. The `player_slice`
helper (defined below) makes slice extraction ergonomic and ForwardDiff-safe.

All terms must be ForwardDiff-compatible: `evaluate_cost_term` will be called
with dual-number `x` and `u` during quadraticization.

# Required Interface
- `evaluate_cost_term(term, x, u, p, t) -> scalar`

# Optional Interface (fall back to ForwardDiff if not provided)
- `cost_term_gradient(term, x, u, p, t) -> (∇ₓ, ∇ᵤ)`
- `cost_term_hessian(term, x, u, p, t) -> (Hₓₓ, Hᵤᵤ, Hₓᵤ)`

# Trait Queries
- `is_quadratic(term)       -> Bool` — exactly quadratic in (x, u)
- `is_separable_term(term)  -> Bool` — reads only player i's slice of (x, u)
"""
abstract type AbstractCostTerm end

"""
    AbstractTerminalCostTerm

A single additive term in a player's terminal cost.
Receives the full joint state `x`.

# Required Interface
- `evaluate_cost_term(term, x, p) -> scalar`
"""
abstract type AbstractTerminalCostTerm end

# ============================================================================
# Trait defaults
# ============================================================================

"""
    is_quadratic(term) -> Bool

True if the term is exactly quadratic in (x, u). Conservative default: false.
Quadratic concrete types override to true, allowing DifferentialGame to
classify the game as an LQ game and use FNELQ directly.
"""
is_quadratic(::AbstractCostTerm)         = false
is_quadratic(::AbstractTerminalCostTerm) = false

"""
    is_separable_term(term::AbstractCostTerm) -> Bool

True if the term depends only on a single player's slice of (x, u), not
on any other player's state or control. Conservative default: false.
"""
is_separable_term(::AbstractCostTerm) = false

# ============================================================================
# player_slice — ergonomic, ForwardDiff-safe slice extraction
# ============================================================================

"""
    player_slice(x, offset::Int, dim::Int) -> view

Extract player i's state or control slice from a joint vector.
Uses `view` to avoid allocation; safe with ForwardDiff dual numbers.

# Arguments
- `x`      : Joint state or control vector (plain or dual-number)
- `offset` : Zero-based offset into `x` for player i (from `control_offsets[i]`)
- `dim`    : Slice length (state_dim or control_dim for player i)

# Example — proximity cost reading two players' positions
```julia
struct ProximityCostTerm <: AbstractCostTerm
    i_offset::Int; i_dim::Int   # player i state slice
    j_offset::Int; j_dim::Int   # player j state slice
    d_min::Float64; weight::Float64
end

function evaluate_cost_term(t::ProximityCostTerm, x, u, p, ts)
    xᵢ = player_slice(x, t.i_offset, t.i_dim)
    xⱼ = player_slice(x, t.j_offset, t.j_dim)
    Δ  = xᵢ[1:2] - xⱼ[1:2]
    d  = sqrt(dot(Δ, Δ) + 1e-6)   # softened distance, ForwardDiff-safe
    return t.weight * max(t.d_min - d, 0.0)^2
end
```
"""
@inline player_slice(x::AbstractVector, offset::Int, dim::Int) =
    view(x, offset+1 : offset+dim)

# ============================================================================
# CompositeCostTerm — produced by +; flat tuple for zero-overhead dispatch
# ============================================================================

"""
    CompositeCostTerm{TS <: Tuple} <: AbstractCostTerm

Sum of `AbstractCostTerm`s. Produced by the `+` operator.

The tuple type parameter `TS` fully specializes evaluation at compile time —
no boxing or dynamic dispatch. Tuples are kept flat: `(a + b) + c` produces
`CompositeCostTerm{Tuple{A,B,C}}`, not nested `CompositeCostTerm{...,C}`.
"""
struct CompositeCostTerm{TS <: Tuple} <: AbstractCostTerm
    terms::TS
end

"""
    CompositeTerminalCostTerm{TS <: Tuple} <: AbstractTerminalCostTerm

Sum of `AbstractTerminalCostTerm`s. Produced by `+`.
"""
struct CompositeTerminalCostTerm{TS <: Tuple} <: AbstractTerminalCostTerm
    terms::TS
end

# ============================================================================
# + operator — flat tuple construction
# ============================================================================

Base.:+(a::AbstractCostTerm, b::AbstractCostTerm) =
    CompositeCostTerm((a, b))

Base.:+(a::CompositeCostTerm, b::AbstractCostTerm) =
    CompositeCostTerm((a.terms..., b))

Base.:+(a::AbstractCostTerm, b::CompositeCostTerm) =
    CompositeCostTerm((a, b.terms...))

Base.:+(a::CompositeCostTerm, b::CompositeCostTerm) =
    CompositeCostTerm((a.terms..., b.terms...))

Base.:+(a::AbstractTerminalCostTerm, b::AbstractTerminalCostTerm) =
    CompositeTerminalCostTerm((a, b))

Base.:+(a::CompositeTerminalCostTerm, b::AbstractTerminalCostTerm) =
    CompositeTerminalCostTerm((a.terms..., b))

Base.:+(a::AbstractTerminalCostTerm, b::CompositeTerminalCostTerm) =
    CompositeTerminalCostTerm((a, b.terms...))

Base.:+(a::CompositeTerminalCostTerm, b::CompositeTerminalCostTerm) =
    CompositeTerminalCostTerm((a.terms..., b.terms...))

# ============================================================================
# evaluate_cost_term — forward declaration + composite implementations
# ============================================================================

"""
    evaluate_cost_term(term, x, u, p, t) -> scalar   [stage]
    evaluate_cost_term(term, x, p)       -> scalar   [terminal]

Evaluate a cost term. `x` is always the full joint state. `u` is the full
joint control vector. Must accept ForwardDiff dual numbers.
"""
function evaluate_cost_term end

function evaluate_cost_term(term::CompositeCostTerm, x, u, p, t)
    # `trm` avoids shadowing the time argument `t`
    result = zero(eltype(x))
    for trm in term.terms
        result += evaluate_cost_term(trm, x, u, p, t)
    end
    return result
end

function evaluate_cost_term(term::CompositeTerminalCostTerm, x, p)
    result = zero(eltype(x))
    for trm in term.terms
        result += evaluate_cost_term(trm, x, p)
    end
    return result
end

# ============================================================================
# Gradient and Hessian — ForwardDiff fallback for all composite terms
# ============================================================================

"""
    cost_term_gradient(term::AbstractCostTerm, x, u, p, t) -> (∇ₓ, ∇ᵤ)

Gradient of stage cost w.r.t. full joint (x, u). Default: ForwardDiff.
Concrete terms with analytical gradients should override this.
"""
function cost_term_gradient(term::AbstractCostTerm, x, u, p, t)
    nx = length(x)
    z  = vcat(x, u)
    g  = ForwardDiff.gradient(
        z_var -> evaluate_cost_term(term, z_var[1:nx], z_var[nx+1:end], p, t),
        z
    )
    return (g[1:nx], g[nx+1:end])
end

"""
    cost_term_hessian(term::AbstractCostTerm, x, u, p, t) -> (Hₓₓ, Hᵤᵤ, Hₓᵤ)

Hessian blocks of stage cost term w.r.t. full joint (x, u). Default: ForwardDiff.

Returns the (n×n), (m×m), and (n×m) blocks. For quadratic terms this is exact;
for nonlinear terms it is the exact second-order Taylor coefficient at (x, u).
"""
function cost_term_hessian(term::AbstractCostTerm, x, u, p, t)
    nx = length(x)
    z  = vcat(x, u)
    H  = ForwardDiff.hessian(
        z_var -> evaluate_cost_term(term, z_var[1:nx], z_var[nx+1:end], p, t),
        z
    )
    return (H[1:nx, 1:nx], H[nx+1:end, nx+1:end], H[1:nx, nx+1:end])
end

"""
    cost_term_gradient(term::AbstractTerminalCostTerm, x, p) -> ∇ₓ
"""
function cost_term_gradient(term::AbstractTerminalCostTerm, x, p)
    return ForwardDiff.gradient(x_var -> evaluate_cost_term(term, x_var, p), x)
end

"""
    cost_term_hessian(term::AbstractTerminalCostTerm, x, p) -> Hₓₓ
"""
function cost_term_hessian(term::AbstractTerminalCostTerm, x, p)
    return ForwardDiff.hessian(x_var -> evaluate_cost_term(term, x_var, p), x)
end

# Composite trait propagation
is_quadratic(term::CompositeCostTerm)         = all(is_quadratic, term.terms)
is_quadratic(term::CompositeTerminalCostTerm) = all(is_quadratic, term.terms)
is_separable_term(term::CompositeCostTerm)    = all(is_separable_term, term.terms)

# ============================================================================
# minimize() — converts cost terms into PlayerObjective
# ============================================================================

"""
    minimize(stage; terminal=nothing, player_id, scaling=1.0) -> PlayerObjective

Convert cost terms into a `PlayerObjective` consumable by `DifferentialGame`
and all existing solvers (FNELQ, iLQGames, FALCON).

# Arguments
- `stage`     : `AbstractCostTerm` or `CompositeCostTerm` — stage cost
- `terminal`  : `AbstractTerminalCostTerm` or `CompositeTerminalCostTerm`;
  if `nothing`, a zero terminal cost is used
- `player_id` : Player index (1-based)
- `scaling`   : Global cost scaling factor (default 1.0)

# Returns
`PlayerObjective` wrapping `NonlinearStageCost` / `NonlinearTerminalCost`.

# Example
```julia
# Player 1: track goal, penalize inputs, avoid player 2 (cross-agent coupling)
obs = minimize(
    track_goal(xg, Q, state_offset=0, state_dim=4) +
    regularize_input(R, control_offset=0, control_dim=2) +
    avoid_proximity(j_offset=4, j_dim=4, d_min=1.0, weight=100.0),
    terminal = track_goal(xg, Qf, state_offset=0, state_dim=4),
    player_id = 1
)
```
"""
function minimize(
    stage::AbstractCostTerm;
    terminal::Union{Nothing, AbstractTerminalCostTerm} = nothing,
    player_id::Int,
    scaling::Float64 = 1.0
)
    @assert player_id > 0  "player_id must be positive"
    @assert scaling > 0.0  "scaling must be positive"

    stage_cost = NonlinearStageCost(
        (x, u, p, t) -> evaluate_cost_term(stage, x, u, p, t);
        gradient     = (x, u, p, t) -> cost_term_gradient(stage, x, u, p, t),
        hessian      = (x, u, p, t) -> cost_term_hessian(stage, x, u, p, t),
        is_separable = is_separable_term(stage)
    )

    term_cost = if terminal === nothing
        NonlinearTerminalCost(
            (x, p) -> zero(eltype(x));
            gradient = (x, p) -> zeros(eltype(x), length(x)),
            hessian  = (x, p) -> zeros(eltype(x), length(x), length(x))
        )
    else
        NonlinearTerminalCost(
            (x, p) -> evaluate_cost_term(terminal, x, p);
            gradient = (x, p) -> cost_term_gradient(terminal, x, p),
            hessian  = (x, p) -> cost_term_hessian(terminal, x, p)
        )
    end

    return PlayerObjective(player_id, stage_cost, term_cost, scaling)
end