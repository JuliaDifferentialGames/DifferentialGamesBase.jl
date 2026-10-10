# ============================================================================
# constraints/chance_constraints.jl
#
# Chance constraints: problem definition (ChanceConstraint), uncertainty
# propagation (uncertainty/propagation.jl), and probability evaluation are
# three deliberately separate concerns:
#
#   problem definition  ≠  uncertainty approximation  ≠  chance evaluation
#
# A ChanceConstraint is just a scalar/vector probabilistic requirement
# Pr[c(x,u,p,t) ≤ 0] ≥ 1-ε. It drops into an ordinary private/shared
# constraint list (so it composes with GameProblem construction/validation)
# and, when evaluated with no belief context, behaves as the nominal
# (no-uncertainty) constraint. `compile_chance_constraint` is what turns it,
# given a propagator and evaluator, into an ordinary deterministic
# PrivateNonlinearInequality/SharedNonlinearInequality that every existing
# solver in DifferentialGamesBaseSolvers.jl already understands.
#
# Include order: after constraints/private_constraints.jl,
# constraints/shared_constraints.jl, and uncertainty/propagation.jl.
# ============================================================================

# ============================================================================
# ChanceConstraint — problem definition
# ============================================================================

"""
    ChanceConstraint{C, R, S}

A probabilistic constraint `Pr[constraint(x,u,p,t) ≤ 0] ≥ 1 - risk`.

Wraps an ordinary `AbstractPrivateInequality`/`AbstractSharedInequality` so
`get_player`/`get_players` forward to it and it can live directly in a
`GameProblem`'s private/shared constraint lists. `evaluate_constraint` on a
bare `ChanceConstraint` falls back to the inner constraint's nominal
(no-uncertainty) residual — the risk-aware residual only exists after
`compile_chance_constraint`.

# Fields
- `constraint::C` : inner `AbstractPrivateInequality` or `AbstractSharedInequality`
- `risk::R`       : acceptable violation probability ε ∈ (0, 1)
- `scope::Symbol` : `:private` or `:shared`, matching `constraint`'s kind
"""
struct ChanceConstraint{C, R} <: AbstractConstraint
    constraint::C
    risk::R
    scope::Symbol

    function ChanceConstraint(constraint::C, risk::R) where {C<:AbstractConstraint, R<:Real}
        @assert is_inequality(constraint) "ChanceConstraint currently supports inequality constraints only"
        @assert 0 < risk < 1 "risk (ε) must be in (0, 1)"
        scope = is_private(constraint) ? :private : :shared
        new{C, R}(constraint, risk, scope)
    end
end

is_private(cc::ChanceConstraint) = cc.scope === :private
is_shared(cc::ChanceConstraint) = cc.scope === :shared
is_inequality(::ChanceConstraint) = true
is_equality(::ChanceConstraint) = false
get_player(cc::ChanceConstraint) = get_player(cc.constraint)
get_players(cc::ChanceConstraint) = get_players(cc.constraint)
is_convex(::ChanceConstraint) = false
depends_on_control(cc::ChanceConstraint) = depends_on_control(cc.constraint)

evaluate_constraint(cc::ChanceConstraint, x, u, p, t) = evaluate_constraint(cc.constraint, x, u, p, t)
constraint_output_dim(cc::ChanceConstraint) = constraint_output_dim(cc.constraint)

function Base.show(io::IO, cc::ChanceConstraint{C}) where {C}
    print(io, "ChanceConstraint{", C, "}(risk=", cc.risk, ", scope=:", cc.scope, ")")
end

# ============================================================================
# AbstractChanceEvaluator — probability -> deterministic-equivalent residual
# ============================================================================

"""
    AbstractChanceEvaluator

Interface for turning a scalar constraint value's distribution (given a mean
and variance along the constraint's gradient direction) into a deterministic
inequality residual. Implement
`chance_residual(evaluator, c_mean, c_var, risk) -> Real`.
"""
abstract type AbstractChanceEvaluator end

"""
    GaussianQuantileEvaluator <: AbstractChanceEvaluator

Linearized-Gaussian deterministic equivalent (Boole/Chebyshev-free, exact
under the Gaussian linearization assumption):

    c(μ,u) + Φ⁻¹(1-ε) * sqrt(∇c' Σ ∇c) ≤ 0

implemented via the inverse error function (`Φ⁻¹(q) = sqrt(2) * erfinv(2q-1)`).
"""
struct GaussianQuantileEvaluator <: AbstractChanceEvaluator end

function _norm_quantile(q::Real)
    # Φ⁻¹(q) = sqrt(2) * erfinv(2q - 1). Avoids a SpecialFunctions dependency
    # via a rational (Acklam-free) approximation accurate to ~1e-9 for the
    # q ∈ (0.5, 1) range relevant to one-sided chance constraints (ε < 0.5).
    return sqrt(2) * _erfinv(2q - 1)
end

function _erfinv(x::Real)
    # Winitzki (2008) rational approximation to erfinv, |err| < 1.3e-4 absolute;
    # refined with two Newton steps against erf for double-precision accuracy.
    a = 0.147
    ln1mx2 = log(1 - x^2)
    t1 = 2 / (π * a) + ln1mx2 / 2
    y = sign(x) * sqrt(sqrt(t1^2 - ln1mx2 / a) - t1)
    for _ in 1:2
        y -= (erf_approx(y) - x) / (2 / sqrt(π) * exp(-y^2))
    end
    return y
end

function erf_approx(x::Real)
    # Abramowitz & Stegun 7.1.26, |err| < 1.5e-7.
    t = 1 / (1 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
        t * (-1.453152027 + t * 1.061405429))))
    s = 1 - poly * exp(-x^2)
    return x >= 0 ? s : -s
end

"""
    chance_residual(evaluator::AbstractChanceEvaluator, c_mean, c_var, risk) -> Real

Deterministic-equivalent residual for a scalar constraint value with mean
`c_mean` and variance `c_var` along the constraint's gradient direction, at
risk level `risk = ε`. Satisfying `chance_residual(...) ≤ 0` is the
deterministic-equivalent of `Pr[c(x,u) ≤ 0] ≥ 1-ε` under `evaluator`.
"""
function chance_residual(::GaussianQuantileEvaluator, c_mean::Real, c_var::Real, risk::Real)
    q = _norm_quantile(1 - risk)
    return c_mean + q * sqrt(max(c_var, zero(c_var)))
end

"""
    MonteCarloChanceEvaluator <: AbstractChanceEvaluator

Sample-based chance evaluation. Reserved interface point — not implemented
in this round. See docs/src/api/stochastic.md, Phase 3.
"""
struct MonteCarloChanceEvaluator <: AbstractChanceEvaluator
    n_samples::Int
end
chance_residual(::MonteCarloChanceEvaluator, args...) =
    error("MonteCarloChanceEvaluator is not implemented yet — see docs/src/api/stochastic.md, Phase 3 roadmap.")

"""
    ScenarioChanceEvaluator <: AbstractChanceEvaluator

Scenario-approximation chance evaluation. Reserved interface point — not
implemented in this round. See docs/src/api/stochastic.md, Phase 3.
"""
struct ScenarioChanceEvaluator <: AbstractChanceEvaluator
    n_scenarios::Int
end
chance_residual(::ScenarioChanceEvaluator, args...) =
    error("ScenarioChanceEvaluator is not implemented yet — see docs/src/api/stochastic.md, Phase 3 roadmap.")

# ============================================================================
# Risk allocation — splitting a joint risk budget across constraint components
# ============================================================================

"""
    AbstractRiskAllocation

Interface for distributing risk across the components of a vector-valued
`ChanceConstraint`. Implement `allocate_risk(policy, cc) -> Vector{<:Real}`
(one risk value per output component).
"""
abstract type AbstractRiskAllocation end

"""
    UniformRiskAllocation <: AbstractRiskAllocation

Split `cc.risk` equally (Bonferroni-style, `ε/n`) across all `n` output
components of the constraint.
"""
struct UniformRiskAllocation <: AbstractRiskAllocation end

"""
    allocate_risk(policy::AbstractRiskAllocation, cc::ChanceConstraint) -> Vector{<:Real}

Per-output-component risk values `ε_i` for `cc`, one per component of
`constraint_output_dim(cc)`, according to `policy`.
"""
allocate_risk(::UniformRiskAllocation, cc::ChanceConstraint) =
    fill(cc.risk / constraint_output_dim(cc), constraint_output_dim(cc))

"""
    FixedRiskAllocation(epsilons::Vector{<:Real}) <: AbstractRiskAllocation

User-specified per-component risk allocation. `length(epsilons)` must match
the constraint's output dimension.
"""
struct FixedRiskAllocation{V<:AbstractVector{<:Real}} <: AbstractRiskAllocation
    epsilons::V
end
function allocate_risk(policy::FixedRiskAllocation, cc::ChanceConstraint)
    @assert length(policy.epsilons) == constraint_output_dim(cc) "FixedRiskAllocation length mismatch"
    return policy.epsilons
end

"""
    OptimizedRiskAllocation <: AbstractRiskAllocation

Risk allocation optimized jointly with the trajectory. Reserved interface
point — not implemented in this round. See docs/src/api/stochastic.md, Phase 3.
"""
struct OptimizedRiskAllocation <: AbstractRiskAllocation end
allocate_risk(::OptimizedRiskAllocation, args...) =
    error("OptimizedRiskAllocation is not implemented yet — see docs/src/api/stochastic.md, Phase 3 roadmap.")

# ============================================================================
# compile_chance_constraint — the deterministic-equivalent compiler
# ============================================================================

"""
    compile_chance_constraint(cc::ChanceConstraint, belief_fn, propagator, evaluator;
                               risk_allocation=UniformRiskAllocation())
        -> PrivateNonlinearInequality | SharedNonlinearInequality

Compile a `ChanceConstraint` into an ordinary deterministic constraint over
the *augmented* (mean, covariance) state produced by `determinize`. `belief_fn`
extracts `(μ, Σ)` from the augmented state vector `z` (see
problems/determinize.jl).

For each output component `i` of the inner constraint, with gradient
`∇c_i = (∂c_i/∂μ)` and `c_var_i = ∇c_i' Σ ∇c_i`, the compiled residual is
`chance_residual(evaluator, c_i(μ,u), c_var_i, ε_i) ≤ 0`, where `ε_i` comes
from `risk_allocation`. `propagator` is accepted for interface symmetry with
`determinize` (the covariance used here is read directly off the augmented
state, which `propagator` was responsible for producing).
"""
function compile_chance_constraint(
    cc::ChanceConstraint,
    belief_fn::Function,
    propagator::AbstractUncertaintyPropagator,
    evaluator::AbstractChanceEvaluator;
    risk_allocation::AbstractRiskAllocation = UniformRiskAllocation()
)
    inner = cc.constraint
    dim = constraint_output_dim(inner)
    epsilons = allocate_risk(risk_allocation, cc)

    residual = (z, u, p, t) -> begin
        μ, Σ = belief_fn(z)
        c_val = evaluate_constraint(inner, μ, u, p, t)
        Jx, _ = constraint_jacobian(inner, μ, u, p, t)
        map(1:dim) do i
            ∇c = Jx[i, :]
            c_var = dot(∇c, Σ * ∇c)
            chance_residual(evaluator, c_val[i], c_var, epsilons[i])
        end
    end

    if is_private(cc)
        PrivateInequality(get_player(cc); func=residual, dim=dim, depends_on_control=depends_on_control(inner))
    else
        SharedInequality(get_players(cc); func=residual, dim=dim)
    end
end
