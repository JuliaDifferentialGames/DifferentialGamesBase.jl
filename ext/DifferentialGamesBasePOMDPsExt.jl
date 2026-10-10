module DifferentialGamesBasePOMDPsExt

# ============================================================================
# Package extension: DifferentialGamesBase + POMDPs.jl
#
# POMDPs.jl is used only as the *interface for defining* MDP/POMDP games —
# not as a dependency of the solve path. `from_mdp`/`from_pomdp` build an
# ordinary `StochasticGameProblem`/`POGameProblem`; solving proceeds entirely
# through DifferentialGamesBaseSolvers.jl, unaware POMDPs.jl was involved.
#
# Scoped to continuous/vector `statetype`/`actiontype` MDPs this round — see
# docs/src/api/stochastic.md Phase 5 for arbitrary discrete/hybrid models.
# ============================================================================

using DifferentialGamesBase
using POMDPs
using SparseArrays: sparse

import DifferentialGamesBase:
    from_mdp, from_pomdp,
    MarkovKernelDynamics, StochasticGameProblem,
    GameProblem, PlayerObjective, NonlinearStageCost, NonlinearTerminalCost,
    DiscreteTime, GameMetadata, CouplingGraph

"""
    from_mdp(mdp; horizon, state_embedding=identity, action_embedding=identity,
              chance_constraints=Function[]) -> StochasticGameProblem

Build a one-player `StochasticGameProblem` from a `POMDPs.jl` MDP whose
`statetype`/`actiontype` are (or embed into) `AbstractVector{<:Real}`.

`horizon` truncates infinite-horizon/discounted MDPs to a finite number of
steps; the discount `γ^k` is folded into the per-step stage cost rather than
silently treated as finite-horizon. `state_embedding`/`action_embedding`
convert between the MDP's native state/action types and plain vectors —
required whenever those types are not already `AbstractVector{<:Real}`.
"""
function from_mdp(
    mdp;
    horizon::Int,
    state_embedding::Function = identity,
    action_embedding::Function = identity,
    chance_constraints::Vector = []
)
    s0 = state_embedding(rand(POMDPs.initialstate(mdp)))
    n = length(s0)
    a0 = action_embedding(first(POMDPs.actions(mdp)))
    m = length(a0)
    γ = POMDPs.discount(mdp)

    kernel = (x, u, p, t) -> begin
        s = state_embedding isa typeof(identity) ? x : error(
            "from_mdp requires an invertible state_embedding to call back into POMDPs.jl transition(); provide one via the `state_embedding` keyword.")
        POMDPs.transition(mdp, s, action_embedding isa typeof(identity) ? u : error(
            "from_mdp requires an invertible action_embedding to call back into POMDPs.jl transition(); provide one via the `action_embedding` keyword."))
    end
    dyn = MarkovKernelDynamics(kernel, n, m)

    stage = NonlinearStageCost((x, u, p, t) ->
        -γ^Int(round(t)) * POMDPs.reward(mdp, x, u))
    terminal = NonlinearTerminalCost((x, p) -> 0.0)
    objective = PlayerObjective(1, stage, terminal)

    metadata = GameMetadata([n], [m], [0], [0], CouplingGraph(
        sparse(Bool[true]), Vector{Int}[], nothing), false, nothing, nothing)
    th = DiscreteTime(Float64(horizon), 1.0)

    nominal = GameProblem{Float64}(1, [objective], _mdp_placeholder_drift(n, m), s0,
        Vector(chance_constraints), [], th, metadata)

    StochasticGameProblem(nominal, dyn)
end

# `StochasticGameProblem` asserts its stochastic_dynamics' state dim matches
# nominal_game.dynamics' state dim — nominal_game.dynamics is otherwise
# unused for a MarkovKernelDynamics-backed game (the kernel is the sole
# source of truth for transitions), so a zero dynamics placeholder suffices.
function _mdp_placeholder_drift(n::Int, m::Int)
    DifferentialGamesBase.CoupledNonlinearDynamics((x, u, p, t) -> zeros(eltype(x), n), n, m)
end

"""
    from_pomdp(pomdp; updater, initial_belief, horizon, state_embedding=identity,
               action_embedding=identity) -> POGameProblem

Build a one-player `POGameProblem` wrapping `from_mdp(pomdp; ...)`, wiring
`updater`/`initial_belief` through to the existing `POGameProblem` fields.

Belief-update/filtering math (turning `updater`/`initial_belief` into an
actual belief-space game) is **not implemented this round** — see
docs/src/api/stochastic.md Phase 2.
"""
function from_pomdp(pomdp; updater, initial_belief, horizon::Int, kwargs...)
    error("""
    from_pomdp belief-space game construction is not implemented yet.
    from_mdp (continuous/vector-state MDPs) is available now; full POMDP
    belief-game support is Phase 2 — see docs/src/api/stochastic.md.
    """)
end

end # module
