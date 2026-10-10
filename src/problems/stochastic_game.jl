# ============================================================================
# problems/stochastic_game.jl
#
# Concrete AbstractStochasticGame subtype, now expressed in terms of the
# general ItoSDEDynamics/StochasticDiscreteApproximation pipeline
# (dynamics/stochastic_dynamics.jl, dynamics/stochastic_discretization.jl)
# rather than a standalone additive-noise vocabulary. `ProcessNoiseModel` is
# kept as a backward-compatible convenience constructor: it builds the
# `ItoSDEDynamics` that reproduces its old semantics exactly (additive
# Gaussian noise on the joint state), so existing call sites (and the
# SFNELQ stochastic-LQ Nash solver in DifferentialGamesBaseSolvers.jl, which
# only ever consumed additive/constant-diffusion noise) keep working.
#
# Still wraps rather than duplicates — dynamics, objectives, and constraints
# all still come from `nominal_game`.
# ============================================================================

"""
    ProcessNoiseModel{T}

Backward-compatible data holder for additive Gaussian process noise on the
joint state: `x_{k+1} = f(x_k, u_k) + w_k`, `w_k ~ N(0, Σ)`. On its own it
carries no drift information; `StochasticGameProblem(nominal_game,
::ProcessNoiseModel)` is what turns it into the `ItoSDEDynamics` that
reproduces this exact semantics, using `nominal_game.dynamics` (must be
`LinearDynamics`) as the drift.

Distinct from `SharedObservation`/`PrivateObservation`'s noise fields, which
model *measurement* noise (what a player observes about the state), not
*dynamics* noise (how the true state evolves) — see the design note in
`abstract/information_structure.jl`.

# Fields
- `Σ::Union{Matrix{T}, Vector{Matrix{T}}}` : Process noise covariance, either
  constant (LTI, supported) or one matrix per time step (LTV — not yet
  supported by the `ItoSDEDynamics` conversion; raises an error, see
  docs/src/api/stochastic.md Phase roadmap).
"""
struct ProcessNoiseModel{T}
    Σ::Union{Matrix{T}, Vector{Matrix{T}}}

    function ProcessNoiseModel{T}(Σ::Matrix{T}) where {T}
        @assert issymmetric(Σ) "Process noise covariance must be symmetric"
        @assert isposdef(Σ) "Process noise covariance must be positive definite"
        new{T}(Σ)
    end

    function ProcessNoiseModel{T}(Σ::Vector{Matrix{T}}) where {T}
        @assert !isempty(Σ) "Process noise covariance sequence must be non-empty"
        for Σk in Σ
            @assert issymmetric(Σk) "Each process noise covariance must be symmetric"
            @assert isposdef(Σk) "Each process noise covariance must be positive definite"
        end
        new{T}(Σ)
    end
end

ProcessNoiseModel(Σ::Matrix{T}) where {T} = ProcessNoiseModel{T}(Σ)
ProcessNoiseModel(Σ::Vector{Matrix{T}}) where {T} = ProcessNoiseModel{T}(Σ)

is_ltv(model::ProcessNoiseModel) = model.Σ isa Vector
covariance(model::ProcessNoiseModel, k::Int) = is_ltv(model) ? model.Σ[k] : model.Σ

"""
    _ito_sde_from_process_noise(drift, pnm::ProcessNoiseModel) -> ItoSDEDynamics

Build the additive-noise `ItoSDEDynamics` equivalent to a (legacy)
`ProcessNoiseModel`.
"""
function _ito_sde_from_process_noise(drift::DynamicsSpec{T}, pnm::ProcessNoiseModel{T}) where {T}
    is_ltv(pnm) && error("LTV ProcessNoiseModel is not yet supported by the ItoSDEDynamics conversion — see docs/src/api/stochastic.md Phase roadmap.")
    Σ = pnm.Σ
    G = Matrix(cholesky(Symmetric(Σ)).L)
    ItoSDEDynamics(drift, constant_diffusion(G), size(G, 2))
end

"""
    StochasticGameProblem{T} <: AbstractStochasticGame{T}

A deterministic `nominal_game` with stochastic dynamics layered on top of
its drift. Nothing about the nominal game's dynamics/objectives/constraints
is duplicated; `stochastic_dynamics` wraps `nominal_game.dynamics` as the
drift of an `ItoSDEDynamics` (or, for the Phase 1 POMDPs.jl adapter, is a
`MarkovKernelDynamics`).

# Fields
- `nominal_game::AbstractDeterministicGame{T}`
- `stochastic_dynamics::AbstractStochasticDynamics{T}`
"""
struct StochasticGameProblem{T} <: AbstractStochasticGame{T}
    nominal_game::AbstractDeterministicGame{T}
    stochastic_dynamics::AbstractStochasticDynamics{T}

    function StochasticGameProblem{T}(
        nominal_game::AbstractDeterministicGame{T},
        stochastic_dynamics::AbstractStochasticDynamics{T}
    ) where {T}
        @assert total_state_dim(stochastic_dynamics) == total_state_dim(nominal_game.dynamics) "stochastic_dynamics state dim must match nominal_game"
        new{T}(nominal_game, stochastic_dynamics)
    end
end

StochasticGameProblem(
    nominal_game::AbstractDeterministicGame{T},
    stochastic_dynamics::AbstractStochasticDynamics{T}
) where {T} = StochasticGameProblem{T}(nominal_game, stochastic_dynamics)

"""
    StochasticGameProblem(nominal_game, pnm::ProcessNoiseModel) -> StochasticGameProblem

Backward-compatible constructor matching the old call pattern. Converts
`pnm` into the `ItoSDEDynamics` that reproduces its additive-noise semantics,
using `nominal_game.dynamics` (must be `LinearDynamics`) as the drift.
"""
function StochasticGameProblem(nominal_game::AbstractDeterministicGame{T}, pnm::ProcessNoiseModel{T}) where {T}
    sde = _ito_sde_from_process_noise(nominal_game.dynamics, pnm)
    StochasticGameProblem{T}(nominal_game, sde)
end

n_players(g::StochasticGameProblem) = n_players(g.nominal_game)
time_horizon(g::StochasticGameProblem) = time_horizon(g.nominal_game)

"""
    get_W(g::StochasticGameProblem) -> Matrix

Noise covariance `GGᵀ` for the wrapped `ItoSDEDynamics`'s diffusion, read
off `.diffusion` rather than a bare `ProcessNoiseModel`. Consumed by
SFNELQ (`DifferentialGamesBaseSolvers.jl`).
"""
function get_W(g::StochasticGameProblem)
    sde = g.stochastic_dynamics
    @assert sde isa ItoSDEDynamics "get_W requires ItoSDEDynamics stochastic dynamics"
    G = diffusion_matrix(sde, g.nominal_game.initial_state,
        zeros(total_control_dim(sde)), nothing, 1)
    return G * G'
end

"""
    has_multiplicative_noise(g::StochasticGameProblem) -> Bool

`true` unless the diffusion is provably state/control-independent
(`ConstantDiffusion`, i.e. additive noise — the certainty-equivalent case).
"""
has_multiplicative_noise(g::StochasticGameProblem) = !is_constant_diffusion(g.stochastic_dynamics)

"""
    is_certainty_equivalent(g::StochasticGameProblem) -> Bool

`true` iff the noise is additive (constant diffusion) — the case in which
the stochastic-LQ Nash feedback gains coincide with the deterministic LQ
Nash gains (only the value-function constant shifts).
"""
is_certainty_equivalent(g::StochasticGameProblem) = !has_multiplicative_noise(g)

function Base.show(io::IO, g::StochasticGameProblem{T}) where {T}
    print(io, "StochasticGameProblem{$T} wrapping ", g.nominal_game,
          " [", typeof(g.stochastic_dynamics).name.name, "]")
end
