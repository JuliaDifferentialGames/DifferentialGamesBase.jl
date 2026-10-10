# ============================================================================
# dynamics/stochastic_dynamics.jl
#
# Stochastic dynamics as a first-class, but not universal, representation.
#
# `ItoSDEDynamics` is the right model for continuous-time controlled Itô
# diffusions (robotics/aerospace noise): dx = f(x,u,t)dt + G(x,u,t)dW. It
# *wraps* an existing deterministic `DynamicsSpec` as its drift, so all of
# the existing Jacobian/dimension/control-slicing machinery for that drift
# type is reused unchanged — only the diffusion term is new.
#
# `MarkovKernelDynamics` exists because POMDPs.jl permits arbitrary
# transition distributions (discrete, hybrid, multimodal, generative) with
# no valid SDE whose discretization reproduces them. It is the landing spot
# for the POMDPs.jl adapter (ext/DifferentialGamesBasePOMDPsExt.jl); kept
# deliberately thin until that adapter needs more.
#
# Include order: must come after dynamics/dynamics.jl (DynamicsSpec,
# concrete drift types) and dynamics/dynamics_interface.jl (evaluate_dynamics,
# dynamics_jacobian) since ItoSDEDynamics forwards to both.
# ============================================================================

"""
    AbstractStochasticDynamics{T} <: DynamicsSpec{T}

Abstract base type for stochastic dynamics representations. Concrete
subtypes: `ItoSDEDynamics` (controlled Itô diffusion) and
`MarkovKernelDynamics` (arbitrary transition kernel, e.g. from POMDPs.jl).
"""
abstract type AbstractStochasticDynamics{T} <: DynamicsSpec{T} end

# ============================================================================
# ItoSDEDynamics — dx = f(x,u,p,t)dt + G(x,u,p,t)dW
# ============================================================================

"""
    ItoSDEDynamics{T, D, G} <: AbstractStochasticDynamics{T}

Controlled Itô diffusion `dx = f(x,u,p,t)dt + G(x,u,p,t)dW`, where the drift
`f` is an *existing* deterministic `DynamicsSpec` (`CoupledNonlinearDynamics`,
`SeparableDynamics`, or `LinearDynamics`) and `diffusion` is a new function
`(x,u,p,t) -> G` with `G` of size `(total_state_dim(drift), noise_dim)`.

Wrapping the drift (rather than reimplementing it) means every existing
Jacobian/AD/dimension accessor for the drift type keeps working unchanged.

# Fields
- `drift::D`       : deterministic `DynamicsSpec{T}` giving f(x,u,p,t)
- `diffusion::G`    : `(x,u,p,t) -> Matrix` giving the diffusion matrix
- `noise_dim::Int`  : dimension of the driving Wiener process
"""
struct ItoSDEDynamics{T, D<:DynamicsSpec{T}, G} <: AbstractStochasticDynamics{T}
    drift::D
    diffusion::G
    noise_dim::Int
end

total_state_dim(dyn::ItoSDEDynamics) = total_state_dim(dyn.drift)
total_control_dim(dyn::ItoSDEDynamics) = total_control_dim(dyn.drift)
n_players(dyn::ItoSDEDynamics) = n_players(dyn.drift)
is_ltv(dyn::ItoSDEDynamics) = is_ltv(dyn.drift)

"""
    noise_dim(dyn::ItoSDEDynamics) -> Int

Dimension of the driving Wiener process `dW`.
"""
noise_dim(dyn::ItoSDEDynamics) = dyn.noise_dim

"""
    diffusion_matrix(dyn::ItoSDEDynamics, x, u, p, t) -> Matrix

Evaluate the diffusion matrix `G(x,u,p,t)`.
"""
diffusion_matrix(dyn::ItoSDEDynamics, x, u, p, t) = dyn.diffusion(x, u, p, t)

"""
    ConstantDiffusion{T}

Callable wrapper marking a diffusion matrix as state/control/time-independent.
`ItoSDEDynamics` dispatches on this type (rather than inspecting closures) to
select the exact Van-Loan-style discretization path for `LinearDynamics`
drift with additive noise — the path that reproduces today's
`ProcessNoiseModel` semantics exactly.
"""
struct ConstantDiffusion{T}
    G::Matrix{T}
end
(c::ConstantDiffusion)(x, u, p, t) = c.G

"""
    constant_diffusion(G::Matrix) -> ItoSDEDynamics helper

Build a `diffusion` callable for `ItoSDEDynamics` from a constant matrix `G`,
tagged so `is_constant_diffusion` / the exact discretization path can detect it.
"""
constant_diffusion(G::Matrix{T}) where {T} = ConstantDiffusion{T}(G)

"""
    is_constant_diffusion(dyn::ItoSDEDynamics) -> Bool

Best-effort check for whether `diffusion` is state/control-independent
(constructed via `constant_diffusion`). Used to pick the exact closed-form
discretization path over the linearized one.
"""
is_constant_diffusion(::ItoSDEDynamics) = false
is_constant_diffusion(::ItoSDEDynamics{T, D, <:ConstantDiffusion}) where {T, D<:DynamicsSpec{T}} = true

function Base.show(io::IO, dyn::ItoSDEDynamics{T}) where {T}
    print(io, "ItoSDEDynamics{$T} [drift=", typeof(dyn.drift).name.name,
          ", noise_dim=", dyn.noise_dim, "]")
end

# ============================================================================
# MarkovKernelDynamics — arbitrary transition kernel (POMDPs.jl bridge)
# ============================================================================

"""
    MarkovKernelDynamics{T, K} <: AbstractStochasticDynamics{T}

An arbitrary discrete-time Markov transition kernel `kernel(x, u, p, t) ->
distribution-like-object`, for stochastic dynamics that need not arise from
any SDE (discrete, hybrid, multimodal, or generative transitions — the
POMDPs.jl case). Kept thin: the POMDPs.jl adapter is the primary consumer.

# Fields
- `kernel::K`        : `(x,u,p,t) -> next-state distribution` (POMDPs.jl-style)
- `state_dim::Int`
- `control_dim::Int`
"""
struct MarkovKernelDynamics{T, K} <: AbstractStochasticDynamics{T}
    kernel::K
    state_dim::Int
    control_dim::Int
end

# T cannot be inferred from the fields (kernel/state_dim/control_dim carry no
# T-typed value), so the convenience constructor defaults it to Float64.
MarkovKernelDynamics(kernel::K, state_dim::Int, control_dim::Int; T::Type = Float64) where {K} =
    MarkovKernelDynamics{T, K}(kernel, state_dim, control_dim)

total_state_dim(dyn::MarkovKernelDynamics) = dyn.state_dim
total_control_dim(dyn::MarkovKernelDynamics) = dyn.control_dim
is_ltv(::MarkovKernelDynamics) = false

function Base.show(io::IO, dyn::MarkovKernelDynamics{T}) where {T}
    print(io, "MarkovKernelDynamics{$T} [state_dim=", dyn.state_dim,
          ", control_dim=", dyn.control_dim, "]")
end

# ============================================================================
# POMDPs.jl adapter stubs — implemented by DifferentialGamesBasePOMDPsExt
# ============================================================================

"""
    from_mdp(mdp; horizon, state_embedding=identity, action_embedding=identity,
              chance_constraints=[]) -> StochasticGameProblem

Requires `using POMDPs` to load `DifferentialGamesBasePOMDPsExt`. See that
extension's docstring for details; scoped to continuous/vector-state MDPs
this round (docs/src/api/stochastic.md Phase 5 covers the general case).
"""
function from_mdp(args...; kwargs...)
    error("from_mdp requires `using POMDPs` to load DifferentialGamesBasePOMDPsExt.")
end

"""
    from_pomdp(pomdp; updater, initial_belief, horizon, kwargs...) -> POGameProblem

Requires `using POMDPs` to load `DifferentialGamesBasePOMDPsExt`. Belief-space
game construction is not implemented this round — see
docs/src/api/stochastic.md Phase 2.
"""
function from_pomdp(args...; kwargs...)
    error("from_pomdp requires `using POMDPs` to load DifferentialGamesBasePOMDPsExt.")
end
