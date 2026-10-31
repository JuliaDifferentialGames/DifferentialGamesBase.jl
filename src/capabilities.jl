# ============================================================================
# capabilities.jl
#
# Canonical, closed vocabulary of solver-capability symbols. Reconciles the
# 7-symbol list documented (but not enforced) on `solver_capabilities` against
# what solvers in DifferentialGamesBaseSolvers.jl already emit today, by
# widening rather than renaming — every symbol currently in use anywhere is a
# member of this set, so no existing solver needs to change.
# ============================================================================

"""
    CANONICAL_CAPABILITIES

The closed set of capability symbols `solver_capabilities`/`required_capabilities`
may use. `validate_capabilities` checks membership; use it in tests to keep new
solvers/problem types from silently drifting into an ad hoc vocabulary.
"""
const CANONICAL_CAPABILITIES = Set{Symbol}([
    # Emitted by `required_capabilities(::GameProblem)` today
    :LQGame, :NonlinearGame, :PotentialGame, :GNEP, :ConstrainedGame,
    :SharedConstraints, :SeparableDynamics, :DiscreteTime,
    # Emitted by DifferentialGamesBaseSolvers.jl solvers today
    :FeedbackPolicies, :OpenLoopPolicies, :UnconstrainedGame,
    :RobustMPC, :TubeBased, :ConvexGame, :VariationalNash, :GeneralizedNash,
    :LexicographicGame, :InverseLQGame, :InfiniteHorizon, :MultiAgent, :RecedingHorizon,
    # New for Phase 0 problem types (A(b)) and continuous-time games
    :ContinuousTime, :Stochastic, :RobustDisturbance, :PartiallyObservable, :ChanceConstrained,
])

"""
    validate_capabilities(caps::Vector{Symbol}) -> nothing

Assert every symbol in `caps` is a member of `CANONICAL_CAPABILITIES`. Throws with
the offending symbols listed on failure.
"""
function validate_capabilities(caps::Vector{Symbol})
    bad = setdiff(caps, CANONICAL_CAPABILITIES)
    isempty(bad) || error(
        "Non-canonical capability symbols: $bad — add to CANONICAL_CAPABILITIES " *
        "in DifferentialGamesBase.jl or fix the solver/problem emitting them."
    )
    return nothing
end
