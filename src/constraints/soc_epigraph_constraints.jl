# ============================================================================
# constraints/soc_epigraph_constraints.jl
#
# SOCEpigraphConstraint — a hard, EXACT second-order-cone epigraph constraint
# ‖u[norm_indices]‖₂ ≤ u[epigraph_index], for use as the standard convex-
# optimization "epigraph trick" for minimum-fuel-style costs: introduce a
# scalar slack s = u[epigraph_index], constrain ‖Δv‖ ≤ s, and make the cost
# LINEAR in s. Unlike a generic PrivateNonlinearInequality marked
# is_convex=true (which FALCON re-linearizes via a single ForwardDiff
# Jacobian per SCP iterate — exact only for affine constraints), this type
# is detected and handled specially by FALCON's `_linearise_hcvx`: since both
# the norm part and the epigraph scalar are RAW components of the local
# control vector (not a nonlinear function of them), the resulting row is an
# EXACT affine embedding for every reference iterate, pushed as a genuine
# `Clarabel.SecondOrderConeT` block — not merely a locally-exact supporting
# hyperplane.
#
# Design notes:
#   - Subtypes AbstractPrivateInequality: involves one player.
#   - evaluate_constraint/constraint_jacobian are provided for API
#     compatibility only (e.g. generic diagnostics) — they are NEVER called
#     in FALCON's normal solve path, which bypasses them entirely in favor of
#     the exact SOC-block construction in `_linearise_hcvx`. This matters
#     because the generic ForwardDiff Jacobian of a norm is NaN at the
#     origin (`Δv=0`) — precisely the regime this constraint needs to handle
#     correctly, which is why FALCON special-cases it instead.
#   - is_convex = true: required so this constraint is excluded from every
#     AL-coupling path (which already skips `is_convex(pc)==true` private
#     constraints uniformly) — solvers must additionally special-case
#     `isa(pc, SOCEpigraphConstraint)` in the generic h_cvx loop to bypass
#     ITS ForwardDiff path too (that loop's guard is the opposite sense,
#     `is_convex(pc) || continue`).
# ============================================================================

"""
    SOCEpigraphConstraint <: AbstractPrivateInequality

Hard cone constraint `‖u[norm_indices]‖₂ ≤ u[epigraph_index]` for `player`,
where both `norm_indices` and `epigraph_index` index into the LOCAL per-knot
control vector `u`.

# Fields
- `player`         : Player index (1-based)
- `norm_indices`   : 1-based indices into `u` forming the vector part of the norm
- `epigraph_index` : 1-based index into `u` for the scalar epigraph slack

# Example
```julia
# Player 1: ‖Δv‖ ≤ s, with Δv = u[1:3], s = u[4]
c = soc_epigraph_constraint(1; norm_indices=[1,2,3], epigraph_index=4)
```
"""
struct SOCEpigraphConstraint <: AbstractPrivateInequality
    player::Int
    norm_indices::Vector{Int}
    epigraph_index::Int

    function SOCEpigraphConstraint(player::Int, norm_indices::Vector{Int}, epigraph_index::Int)
        @assert player > 0 "player must be positive"
        @assert !isempty(norm_indices) && all(>(0), norm_indices) "norm_indices must be positive"
        @assert epigraph_index > 0 "epigraph_index must be positive"
        @assert epigraph_index ∉ norm_indices "epigraph_index must not overlap norm_indices"
        new(player, norm_indices, epigraph_index)
    end
end

# Vestigial API-compat only — see module docstring above. Never called by
# FALCON's normal solve path (bypassed in favor of the exact SOC block).
function evaluate_constraint(c::SOCEpigraphConstraint, x, u, p, t)
    return [norm(u[c.norm_indices]) - u[c.epigraph_index]]
end

function constraint_jacobian(c::SOCEpigraphConstraint, x, u, p, t)
    nx = length(x); nu = length(u)
    z  = vcat(x, u)
    J  = ForwardDiff.jacobian(
        z_var -> evaluate_constraint(c, z_var[1:nx], z_var[nx+1:end], p, t), z
    )
    return (J[:, 1:nx], J[:, nx+1:end])
end

is_convex(::SOCEpigraphConstraint) = true
depends_on_control(::SOCEpigraphConstraint) = true
constraint_output_dim(::SOCEpigraphConstraint) = 1

# ── Ergonomic constructor ─────────────────────────────────────────────────────

"""
    soc_epigraph_constraint(player; norm_indices, epigraph_index) -> SOCEpigraphConstraint

Hard cone constraint `‖u[norm_indices]‖₂ ≤ u[epigraph_index]` for `player`.
See [`SOCEpigraphConstraint`](@ref).
"""
function soc_epigraph_constraint(player::Int; norm_indices::Vector{Int}, epigraph_index::Int)
    SOCEpigraphConstraint(player, norm_indices, epigraph_index)
end
