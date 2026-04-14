# ============================================================================
# constraints/base_constraints.jl
#
# Abstract constraint hierarchy and evaluation interface.
#
# Design principles:
#
#   1. Player ownership lives inside the constraint, not in a wrapper.
#      AbstractPrivateConstraint carries `player::Int`.
#      AbstractSharedConstraint carries `players::Vector{Int}`.
#      No PrivateConstraint{C}/SharedConstraint{C} wrapper types.
#
#   2. Constraint type (equality vs inequality) is encoded via the abstract
#      subtype hierarchy, not a phantom type parameter or runtime symbol.
#      is_equality / is_inequality dispatch on the abstract type.
#
#   3. Every constraint receives the full joint (x, u) at evaluation time.
#      Private constraints ignore irrelevant slices (same convention as costs).
#      player_slice from cost_terms.jl is available for slice extraction.
#
#   4. Jacobians default to ForwardDiff through the same evaluate_constraint
#      function. Analytical overrides are stored as an optional field.
#      This guarantees Jacobian consistency — same principle as DiscreteApproximation.
#
#   5. ForwardDiff compatibility is a contract: all constraint functions must
#      accept dual-number x and u. Use player_slice (view-based) not indexing.
#
# Solver consumption:
#   Private constraints  → accessed via get_player(c) :: Int
#   Shared constraints   → accessed via get_players(c) :: Vector{Int}
#   Both                 → evaluate_constraint(c, x, u, p, t) → Vector
#                          constraint_jacobian(c, x, u, p, t) → (Jx, Ju)
#                          is_active(c, x, u, p, t, tol) → Bool
#
# PDGNEProblem coupling graph:
#   CouplingGraph expects Vector{Int} per shared constraint.
#   Use: [get_players(c) for c in shared_constraints]
# ============================================================================

# ============================================================================
# Abstract hierarchy
# ============================================================================

"""
    AbstractConstraint

Root abstract type for all game constraints.

# Subtype Tree
```
AbstractConstraint
├── AbstractPrivateConstraint   — involves exactly one player (player::Int field)
│   ├── AbstractPrivateInequality  ← c(x,u) ≤ 0
│   └── AbstractPrivateEquality    ← c(x,u) = 0
└── AbstractSharedConstraint    — couples ≥2 players (players::Vector{Int} field)
    ├── AbstractSharedInequality   ← c(x,u) ≤ 0
    └── AbstractSharedEquality     ← c(x,u) = 0
```

# Required Interface (all subtypes)
- `evaluate_constraint(c, x, u, p, t) → AbstractVector`
- `get_player(c)  → Int`           (AbstractPrivateConstraint only)
- `get_players(c) → Vector{Int}`   (AbstractSharedConstraint only)

# Optional Interface (fall back to ForwardDiff if not provided)
- `constraint_jacobian(c, x, u, p, t) → (Jx::Matrix, Ju::Matrix)`
"""
abstract type AbstractConstraint end

# ── Private branch ────────────────────────────────────────────────────────────

"""
    AbstractPrivateConstraint <: AbstractConstraint

Constraint involving a single player. Must carry a `player::Int` field.
Solvers access it via `get_player(c)`.
"""
abstract type AbstractPrivateConstraint <: AbstractConstraint end

"""
    AbstractPrivateInequality <: AbstractPrivateConstraint

Private inequality constraint: `c(x, u, p, t) ≤ 0`.
"""
abstract type AbstractPrivateInequality <: AbstractPrivateConstraint end

"""
    AbstractPrivateEquality <: AbstractPrivateConstraint

Private equality constraint: `c(x, u, p, t) = 0`.
"""
abstract type AbstractPrivateEquality <: AbstractPrivateConstraint end

# ── Shared branch ─────────────────────────────────────────────────────────────

"""
    AbstractSharedConstraint <: AbstractConstraint

Constraint coupling multiple players. Must carry a `players::Vector{Int}` field.
Evaluated once per timestep (not once per player).
Solvers access involved players via `get_players(c)`.
"""
abstract type AbstractSharedConstraint <: AbstractConstraint end

"""
    AbstractSharedInequality <: AbstractSharedConstraint

Shared inequality constraint: `c(x, u, p, t) ≤ 0`.
"""
abstract type AbstractSharedInequality <: AbstractSharedConstraint end

"""
    AbstractSharedEquality <: AbstractSharedConstraint

Shared equality constraint: `c(x, u, p, t) = 0`.
"""
abstract type AbstractSharedEquality <: AbstractSharedConstraint end

# ============================================================================
# Trait queries — dispatch on abstract type, zero runtime cost
# ============================================================================

"""
    is_private(c::AbstractConstraint) -> Bool
"""
is_private(::AbstractPrivateConstraint) = true
is_private(::AbstractSharedConstraint)  = false

"""
    is_shared(c::AbstractConstraint) -> Bool
"""
is_shared(::AbstractSharedConstraint)  = true
is_shared(::AbstractPrivateConstraint) = false

"""
    is_inequality(c::AbstractConstraint) -> Bool
"""
is_inequality(::AbstractPrivateInequality) = true
is_inequality(::AbstractSharedInequality)  = true
is_inequality(::AbstractPrivateEquality)   = false
is_inequality(::AbstractSharedEquality)    = false

"""
    is_equality(c::AbstractConstraint) -> Bool
"""
is_equality(::AbstractPrivateEquality)  = true
is_equality(::AbstractSharedEquality)   = true
is_equality(::AbstractPrivateInequality) = false
is_equality(::AbstractSharedInequality)  = false

# ============================================================================
# Player accessor interface
# ============================================================================

"""
    get_player(c::AbstractPrivateConstraint) -> Int

Return the player index this constraint applies to.
Default implementation reads the `player` field; override if your type
uses a different field name.
"""
get_player(c::AbstractPrivateConstraint) = c.player

"""
    get_players(c::AbstractSharedConstraint) -> Vector{Int}

Return the player indices involved in this shared constraint, sorted ascending.
Default implementation reads the `players` field.
"""
get_players(c::AbstractSharedConstraint) = c.players

# ============================================================================
# Evaluation interface — forward declarations
# ============================================================================

"""
    evaluate_constraint(c, x, u, p, t) -> AbstractVector

Evaluate the constraint at joint state `x` and joint control `u`.

Returns a vector of constraint values. For inequality constraints, satisfaction
means all values ≤ 0. For equality constraints, satisfaction means all values = 0.

All implementations must be ForwardDiff-compatible: accept dual-number x and u.
Use `player_slice(x, offset, dim)` for slice extraction rather than direct indexing.
"""
function evaluate_constraint end

# ============================================================================
# Jacobian — ForwardDiff default, analytical override pattern
# ============================================================================

"""
    constraint_jacobian(c, x, u, p, t) -> (Jx::Matrix, Ju::Matrix)

Compute constraint Jacobian at (x, u, p, t).

Returns `(Jx, Ju)` where:
- `Jx` : (m × n) — ∂c/∂x, full joint state
- `Ju` : (m × m_total) — ∂c/∂u, full joint control

Default: ForwardDiff through `evaluate_constraint`. Concrete types with
analytical Jacobians should override this method or store a `jacobian`
function field and dispatch to it (see PrivateNonlinear, SharedNonlinear).

The ForwardDiff default guarantees Jacobian consistency: the linearization
and the constraint evaluation always use the same function.
"""
function constraint_jacobian(c::AbstractConstraint, x, u, p, t)
    nx = length(x); nu = length(u)
    z  = vcat(x, u)
    J  = ForwardDiff.jacobian(
        z_var -> evaluate_constraint(c, z_var[1:nx], z_var[nx+1:end], p, t),
        z
    )
    return (J[:, 1:nx], J[:, nx+1:end])
end

# ============================================================================
# Activity check
# ============================================================================

"""
    is_active(c, x, u, p, t; tol=1e-6) -> Bool

Returns true if any constraint component is within `tol` of its bound.
Useful for active-set methods and warm-starting dual variables.
"""
function is_active(c::AbstractConstraint, x, u, p, t; tol::Real = 1e-6)
    v = evaluate_constraint(c, x, u, p, t)
    return any(vi -> abs(vi) <= tol, v)
end

"""
    constraint_violation(c, x, u, p, t) -> Float64

Maximum violation: max(max(v), 0) for inequality, max(|v|) for equality.
"""
function constraint_violation(c::AbstractConstraint, x, u, p, t)
    v = evaluate_constraint(c, x, u, p, t)
    if is_inequality(c)
        return maximum(max.(v, zero(eltype(v))))
    else
        return maximum(abs.(v))
    end
end