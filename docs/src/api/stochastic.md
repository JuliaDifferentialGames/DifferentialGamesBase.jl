# Stochastic Dynamics, Chance Constraints, and POMDPs.jl Interoperability

All problem definitions — including stochastic dynamics, chance constraints, and
the POMDPs.jl adapter — live in this package. `DifferentialGamesBaseSolvers.jl`
supplies solvers; `DifferentialGamesMARL.jl` supplies multi-agent RL algorithms
and environments. POMDPs.jl is used only as an interface for *defining*
MDP/POMDP games, never as a dependency of the solve path.

## Design principle

Problem definition, uncertainty propagation, and chance-probability evaluation
are three separate concerns, composed rather than merged:

```
problem definition  ≠  uncertainty approximation  ≠  chance evaluation
```

A [`ChanceConstraint`](@ref) names a probabilistic requirement; an
[`AbstractUncertaintyPropagator`](@ref) evolves a belief; an
[`AbstractChanceEvaluator`](@ref) turns a probability requirement into a
deterministic residual. Any propagator/evaluator pair compiles the same
`ChanceConstraint` via [`compile_chance_constraint`](@ref).

SDEs ([`ItoSDEDynamics`](@ref)) are the right model for continuous-time
controlled Itô diffusions and reuse all of the existing deterministic
`DynamicsSpec` machinery as drift. POMDPs.jl permits arbitrary transition
kernels with no valid SDE representation — that case is
[`MarkovKernelDynamics`](@ref). Both map into the common
[`StochasticDiscreteApproximation`](@ref) representation; never the other
way around.

## Abstract Hierarchy

```@docs
AbstractStochasticDynamics
```

## Stochastic Dynamics

```@docs
ItoSDEDynamics
MarkovKernelDynamics
diffusion_matrix
noise_dim
is_constant_diffusion
constant_diffusion
ConstantDiffusion
```

## Stochastic Discretization

```@docs
StochasticDiscreteApproximation
discretize_stochastic
LinearizedGaussianDiscretization
mean_step
sample_step
process_covariance
transition_distribution
```

## Stochastic / Robust / Partially Observable Game Wrappers

```@docs
StochasticGameProblem
ProcessNoiseModel
RobustGameProblem
DisturbanceSet
POGameProblem
get_W
has_multiplicative_noise
is_certainty_equivalent
```

## Uncertainty Propagation

```@docs
GaussianBelief
mean
covariance
AbstractUncertaintyPropagator
LinearizedGaussianPropagation
UnscentedPropagation
ParticlePropagation
propagate
```

## Chance Constraints

```@docs
ChanceConstraint
AbstractChanceEvaluator
GaussianQuantileEvaluator
MonteCarloChanceEvaluator
ScenarioChanceEvaluator
chance_residual
AbstractRiskAllocation
UniformRiskAllocation
FixedRiskAllocation
OptimizedRiskAllocation
allocate_risk
compile_chance_constraint
```

## Determinization

```@docs
determinize
```

## POMDPs.jl Adapter

Requires `using POMDPs` to load `DifferentialGamesBasePOMDPsExt`.

```@docs
from_mdp
from_pomdp
```

## Roadmap

Phase 0 (the wrapper types above) and Phase 1 (everything else on this page)
are implemented. Later phases are tracked here so future work has a fixed
reference point:

- **Phase 2 — Belief-space POMDPs**: `GaussianBelief`-based EKF/UKF updaters
  (optionally interoperating with `GaussianFilters.jl`),
  `ExpectedBeliefPropagation`/`StochasticBeliefPropagation`, full `from_pomdp`
  belief-game construction, solver support for `POGameProblem`. This is the
  natural place to converge `PrivateObservation`/`SharedObservation`
  (`src/abstract/information_structure.jl`) with a concrete belief updater.
- **Phase 3 — Richer uncertainty propagation/evaluation**: `UnscentedPropagation`,
  `ParticlePropagation`, `MonteCarloChanceEvaluator`, `ScenarioChanceEvaluator`,
  `OptimizedRiskAllocation`, and a `validate(sol, ccmdp; evaluator=...)` workflow.
- **Phase 4 — SciML stochastic backend**: `DifferentialGamesBaseStochasticDiffEqExt.jl`
  (weak dep on `StochasticDiffEq`), `SDESolverDiscretization(alg)` for
  `sample_step`/trajectory sampling/Monte-Carlo validation, mirroring
  `DifferentialGamesBaseDiffEqExt.jl`'s map/Jacobian-consistency pattern.
- **Phase 5 — General `MarkovKernelDynamics`**: arbitrary discrete/hybrid POMDPs.jl
  models beyond Phase 1's continuous-vector-state restriction.
- **Phase 6 — CommonRLInterface interop via POMDPTools**: two-way wrapping to expose
  solved games as RL environments — additive to, not a replacement for,
  `DifferentialGamesMARL.jl`'s own `CommonRLInterface.AbstractEnv` usage.

## Three noise/observation vocabularies

See the design note in `src/abstract/information_structure.jl` for how
`ProcessNoiseModel` (dynamics noise), `PrivateObservation`/`SharedObservation`
(measurement noise, per-player), and `ObservationModel`/`NoisyObservation`
(`src/problems/IGNEP.jl`, inverse-game measurement model) relate. Full
unification is Phase 2 work.
