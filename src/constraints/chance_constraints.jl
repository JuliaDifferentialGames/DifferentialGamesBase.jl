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

# Standard-normal quantile Φ⁻¹(q): Wichura's algorithm AS241 (PPND16), relative
# error ≈ 1e-16 over (0, 1). Implemented locally to avoid a SpecialFunctions dependency.
function _norm_quantile(q::Real)
    (0 < q < 1) || return q == 0 ? -Inf : q == 1 ? Inf : NaN
    p = q - 0.5
    if abs(p) <= 0.425
        r = 0.180625 - p * p
        return p * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
            67265.770927008700853) * r + 45921.953931549871457) * r +
            13731.693765509461125) * r + 1971.5909503065514427) * r +
            133.14166789178437745) * r + 3.387132872796366608) /
            (((((((5226.495278852545925 * r + 28729.085735721942674) * r +
            39307.89580009271061) * r + 21213.794301586595867) * r +
            5394.1960214247511077) * r + 687.1870074920579083) * r +
            42.313330701600911252) * r + 1.0)
    end
    r = p < 0 ? q : 1 - q
    r = sqrt(-log(r))
    if r <= 5.0
        r -= 1.6
        val = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
            0.24178072517745061177) * r + 1.27045825245236838258) * r +
            3.64784832476320460504) * r + 5.7694972214606914055) * r +
            4.6303378461565452959) * r + 1.42343711074968357734) /
            (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
            0.0151986665636164571966) * r + 0.14810397642748007459) * r +
            0.68976733498510000455) * r + 1.6763848301838038494) * r +
            2.05319162663775882187) * r + 1.0)
    else
        r -= 5.0
        val = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
            0.0012426609473880784386) * r + 0.026532189526576123093) * r +
            0.29656057182850489123) * r + 1.7848265399172913358) * r +
            5.4637849111641143699) * r + 6.6579046435011037772) /
            (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
            1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
            0.0148753612908506148525) * r + 0.13692988092273580531) * r +
            0.59983220655588793769) * r + 1.0)
    end
    return p < 0 ? -val : val
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
    risk_allocation::AbstractRiskAllocation = UniformRiskAllocation(),
    step_risks::Union{Nothing, AbstractMatrix} = nothing,
    time_index::Function = t -> t isa Integer ? Int(t) : 1
)
    inner = cc.constraint
    dim = constraint_output_dim(inner)
    epsilons = allocate_risk(risk_allocation, cc)
    # With a horizon-wide budget, ε depends on the step: step_risks is dim × steps.
    ε_at(i, t) = step_risks === nothing ? epsilons[i] :
                 step_risks[i, clamp(time_index(t), 1, size(step_risks, 2))]

    residual = (z, u, p, t) -> begin
        μ, Σ = belief_fn(z)
        c_val = evaluate_constraint(inner, μ, u, p, t)
        Jx, _ = constraint_jacobian(inner, μ, u, p, t)
        map(1:dim) do i
            ∇c = Jx[i, :]
            c_var = dot(∇c, Σ * ∇c)
            chance_residual(evaluator, c_val[i], c_var, ε_at(i, t))
        end
    end

    if is_private(cc)
        PrivateInequality(get_player(cc); func=residual, dim=dim, depends_on_control=depends_on_control(inner))
    else
        SharedInequality(get_players(cc); func=residual, dim=dim)
    end
end

# ============================================================================
# Guarantee classification
# ============================================================================

"""
    is_affine_constraint(c) -> Bool

`true` for constraints known to be affine in `(x, u)`. Under a Gaussian law an
affine scalar chance constraint is evaluated *exactly* by the quantile
formula; for anything else the formula is a first-order approximation.
"""
is_affine_constraint(::AbstractConstraint) = false
is_affine_constraint(::ControlBounds) = true
is_affine_constraint(::StateBounds) = true
is_affine_constraint(::LinearCoupling) = true
is_affine_constraint(cc::ChanceConstraint) = is_affine_constraint(cc.constraint)

"""
    guarantee_class(cc::ChanceConstraint, evaluator) -> Symbol

How strong the probability statement `Pr[c ≤ 0] ≥ 1-ε` is when `cc` is
enforced through `evaluator` under a Gaussian state/control law:
`:exact_affine_gaussian` or `:linearized_approximation`. (`:certified_upper_bound`
is reserved for evaluators that supply a conservative bound.)
"""
guarantee_class(cc::ChanceConstraint, ::GaussianQuantileEvaluator) =
    is_affine_constraint(cc) ? :exact_affine_gaussian : :linearized_approximation

# ============================================================================
# Standard-normal CDF and violation probabilities
# ============================================================================

# erfc to ~1e-14: Taylor series of erf for |x| ≤ 2.5, Lentz continued fraction beyond.
function _erfc(x::Real)
    ax = abs(x)
    if ax <= 2.5
        term = ax; s = ax; x2 = ax * ax
        for n in 1:200
            term *= -x2 / n
            δ = term / (2n + 1)
            s += δ
            abs(δ) < 1e-17 * abs(s) && break
        end
        erfax = 2 / sqrt(π) * s
        return x >= 0 ? 1 - erfax : 1 + erfax
    end
    # erfc(x) = exp(-x²)/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + …)))))
    f = ax
    for k in 60:-1:1
        f = ax + (k / 2) / f
    end
    c = exp(-ax * ax) / (sqrt(π) * f)
    return x >= 0 ? c : 2 - c
end

"Standard-normal CDF Φ(z)."
_norm_cdf(z::Real) = 0.5 * _erfc(-z / sqrt(2))

"""
    gaussian_violation_probability(c_mean, c_var) -> Real

`Pr[c ≤ 0 fails] = Pr[c > 0]` for `c ~ N(c_mean, c_var)`. Exact for an affine
constraint under a Gaussian law; zero variance gives the deterministic indicator.
"""
function gaussian_violation_probability(c_mean::Real, c_var::Real)
    c_var <= 0 && return c_mean > 0 ? 1.0 : 0.0
    return 1 - _norm_cdf(-c_mean / sqrt(c_var))
end

# ============================================================================
# Horizon-wide joint risk allocation (union bound)
# ============================================================================

"""
    JointRiskBudget(total)

A trajectory-wide failure budget: `Pr[some constraint is violated at some step]
≤ total` is guaranteed (by the union bound) once the per-event risks `ε_{c,k,m}`
sum to at most `total` AND every individual chance statement is valid.
"""
struct JointRiskBudget{R<:Real}
    total::R
    function JointRiskBudget(total::R) where {R<:Real}
        @assert 0 < total < 1 "joint risk budget must be in (0, 1)"
        new{R}(total)
    end
end

"""
    allocate_horizon_risk(policy, ccs, steps, budget::JointRiskBudget) -> Vector{Matrix}

Per-event risks `ε_{c,k,m}` for every constraint `c ∈ ccs`, step `k = 1:steps`
and output component `m`, returned as one `dim_c × steps` matrix per constraint.
`UniformRiskAllocation` gives each event `total / (#events)`.
"""
function allocate_horizon_risk(::UniformRiskAllocation, ccs::AbstractVector, steps::Int,
                               budget::JointRiskBudget)
    n_events = sum(constraint_output_dim(cc) for cc in ccs) * steps
    ε = budget.total / n_events
    return [fill(ε, constraint_output_dim(cc), steps) for cc in ccs]
end

"Union-bound failure probability `Σ ε_{c,k,m}` implied by an allocation."
joint_risk_bound(eps::AbstractVector{<:AbstractMatrix}) = sum(sum, eps)

"""
    check_risk_budget(eps, budget) -> Bool

Whether the allocated per-event risks respect the joint budget.
"""
check_risk_budget(eps::AbstractVector{<:AbstractMatrix}, budget::JointRiskBudget) =
    joint_risk_bound(eps) <= budget.total * (1 + 1e-12)

# ============================================================================
# Checking chance constraints on Gaussian trajectory moments (no solver needed)
# ============================================================================

"""
    ChanceCheck

Result of evaluating one chance event `(constraint c, step k, component m)`.
`violation_probability` is exact when `guarantee == :exact_affine_gaussian`
and a linearized approximation otherwise.
"""
struct ChanceCheck
    constraint::Int
    step::Int
    component::Int
    risk::Float64
    residual::Float64
    violation_probability::Float64
    satisfied::Bool
    guarantee::Symbol
end

"""
    check_chance_constraints(ccs, x_mean, u_mean, Σx, Σu, Σxu; risk, evaluator) -> Vector{ChanceCheck}

Evaluate each `ChanceConstraint` at every step from Gaussian trajectory moments:
`x_mean[:, k]`, `u_mean[:, k]`, `Σx[k]`, `Σu[k]`, `Σxu[k] = Cov(x_k, u_k)`.
`risk[c]` is the `dim_c × steps` allocation (default: each constraint's own
`risk` split uniformly over its components). Use [`joint_risk_bound`](@ref) on
the allocation for the trajectory-wide union bound.
"""
function check_chance_constraints(ccs::AbstractVector, x_mean::AbstractMatrix, u_mean::AbstractMatrix,
        Σx, Σu, Σxu; risk = nothing, evaluator::AbstractChanceEvaluator = GaussianQuantileEvaluator())
    steps = size(u_mean, 2)
    out = ChanceCheck[]
    for (ci, cc) in enumerate(ccs)
        inner = cc.constraint
        dim = constraint_output_dim(inner)
        ε = risk === nothing ? reshape(allocate_risk(UniformRiskAllocation(), cc), dim, 1) .* ones(1, steps) : risk[ci]
        for k in 1:steps
            x = x_mean[:, k]; u = u_mean[:, k]
            c = evaluate_constraint(inner, x, u, nothing, k)
            Jx, Ju = constraint_jacobian(inner, x, u, nothing, k)
            for m in 1:dim
                gx = Jx[m, :]; gu = Ju[m, :]
                v = dot(gx, Σx[k] * gx) + 2 * dot(gx, Σxu[k] * gu) + dot(gu, Σu[k] * gu)
                res = chance_residual(evaluator, c[m], v, ε[m, k])
                push!(out, ChanceCheck(ci, k, m, ε[m, k], res, gaussian_violation_probability(c[m], v),
                                       res <= 0, guarantee_class(cc, evaluator)))
            end
        end
    end
    return out
end

"""
    joint_violation_bound(checks) -> Float64

Union bound `min(1, Σ Pr[violation])` on the probability of any violation along
the trajectory, valid when each event probability is valid (not merely
approximate — see `ChanceCheck.guarantee`).
"""
joint_violation_bound(checks::AbstractVector{ChanceCheck}) = min(1.0, sum(c.violation_probability for c in checks; init = 0.0))

# ============================================================================
# Empirical (simulation) validation
# ============================================================================

"""
    empirical_violation_rates(g; z = 1.96) -> NamedTuple

`g` is an `events × samples` matrix of constraint values (`g ≤ 0` is feasible).
Returns per-event violation frequencies, the frequency of *any* violation per
sample, the sample count, and Wilson score intervals at normal quantile `z`.
A zero count is an estimate, not a safety guarantee.
"""
function empirical_violation_rates(g::AbstractMatrix; z::Real = 1.96)
    ne, ns = size(g)
    viol = g .> 0
    per_event = vec(sum(viol; dims = 2)) ./ ns
    any_rate = count(j -> any(@view viol[:, j]), 1:ns) / ns
    wilson(p) = begin
        den = 1 + z^2 / ns
        mid = (p + z^2 / (2ns)) / den
        half = z * sqrt(p * (1 - p) / ns + z^2 / (4ns^2)) / den
        (max(0.0, mid - half), min(1.0, mid + half))
    end
    return (per_event = per_event, per_event_interval = wilson.(per_event),
            any_violation = any_rate, any_violation_interval = wilson(any_rate), samples = ns)
end
