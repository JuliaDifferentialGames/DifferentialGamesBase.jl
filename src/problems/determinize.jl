# ============================================================================
# problems/determinize.jl
#
# Moment/belief-state augmentation: compile a StochasticGameProblem into an
# ordinary deterministic GameProblem over the augmented state z = (μ, Σ_flat),
# so every existing DifferentialGamesBaseSolvers.jl solver can solve it
# unchanged. This is the key deliverable of the Phase 1 stochastic pipeline.
#
# Σ is stored row-major-flattened (not the minimal vech) for implementation
# simplicity — n² extra scalars instead of n(n+1)/2, a constant-factor cost,
# not a correctness issue.
#
# Include order: after problems/stochastic_game.jl, problems/GNEP.jl,
# constraints/chance_constraints.jl, uncertainty/propagation.jl.
# ============================================================================

# ½tr(QΣ) correction that turns a cost evaluated at the mean into the expected
# cost of a quadratic cost under x ~ N(μ, Σ). Zero for non-quadratic costs.
_expected_cost_trace_term(cost::LQStageCost, Σ, t) = 0.5 * tr(get_Q(cost, t) * Σ)
_expected_cost_trace_term(cost::LQTerminalCost, Σ, _) = 0.5 * tr(cost.Qf * Σ)
_expected_cost_trace_term(cost, Σ, t) = zero(eltype(Σ))

_unflatten_belief(z::AbstractVector, n::Int) = (z[1:n], reshape(z[n+1:n+n^2], n, n))
_flatten_belief(μ::AbstractVector, Σ::AbstractMatrix) = vcat(μ, vec(Matrix(Σ)))

"""
    determinize(sg::StochasticGameProblem{T};
                propagator=LinearizedGaussianPropagation(),
                evaluator=GaussianQuantileEvaluator(),
                risk_allocation=UniformRiskAllocation()) -> GameProblem{T}

Build the augmented-state `(μ, Σ)` deterministic game equivalent to `sg`
under `propagator`. Every `ChanceConstraint` present in
`sg.nominal_game`'s private/shared constraint lists is compiled via
`compile_chance_constraint`; every other constraint is carried over
unchanged (evaluated against the `μ` slice of the augmented state). Requires
`sg.nominal_game`'s time horizon to be `DiscreteTime` (dt is needed to
discretize the stochastic dynamics).

`solve(determinize(sg; ...), AnyExistingSolver())` requires no changes to
`DifferentialGamesBaseSolvers.jl`.
"""
function determinize(
    sg::StochasticGameProblem{T};
    propagator::AbstractUncertaintyPropagator = LinearizedGaussianPropagation(),
    evaluator::AbstractChanceEvaluator = GaussianQuantileEvaluator(),
    risk_allocation::AbstractRiskAllocation = sg.risk_allocation
) where {T}
    nominal = sg.nominal_game
    th = time_horizon(nominal)
    @assert th isa DiscreteTime "determinize requires a DiscreteTime horizon (dt needed to discretize stochastic dynamics)"
    dt = th.dt

    n = total_state_dim(sg.stochastic_dynamics)
    sd = discretize_stochastic(sg.stochastic_dynamics, dt)
    belief_fn = z -> _unflatten_belief(z, n)

    # Augmented dynamics: z_{k+1} = (mean_step(μ_k,u_k), flatten(propagate(μ_k,Σ_k,u_k)))
    aug_func = (z, u, p, t) -> begin
        μ, Σ = belief_fn(z)
        belief = GaussianBelief(collect(μ), collect(Σ))
        next = propagate(propagator, sd, belief, u, p, t)
        _flatten_belief(mean(next), covariance(next))
    end
    aug_dim = n + n^2
    m = total_control_dim(nominal.dynamics)
    aug_dynamics = CoupledNonlinearDynamics(aug_func, aug_dim, m)

    # Augmented initial state: nominal x0 as the mean and sg's initial covariance.
    sg.process_covariances === nothing ||
        error("determinize does not support time-varying process covariances yet")
    z0 = _flatten_belief(nominal.initial_state, sg.initial_covariance)

    # Wrap each player's objective so it sees the μ slice of z. For quadratic
    # costs the expected cost is exact: E[½xᵀQx] = ½μᵀQμ + ½tr(QΣ), so the
    # covariance term is added back. Other cost types are evaluated at the mean
    # only (a first-order approximation).
    aug_objectives = map(nominal.objectives) do obj
        stage = NonlinearStageCost(
            (z, u, p, t) -> begin
                μ, Σ = belief_fn(z)
                evaluate_stage_cost(obj.stage_cost, μ, u, p, t) +
                    _expected_cost_trace_term(obj.stage_cost, Σ, t)
            end)
        terminal = NonlinearTerminalCost(
            (z, p) -> begin
                μ, Σ = belief_fn(z)
                evaluate_terminal_cost(obj.terminal_cost, μ, p) +
                    _expected_cost_trace_term(obj.terminal_cost, Σ, nothing)
            end)
        PlayerObjective(obj.player_id, stage, terminal, obj.scaling)
    end

    # Compile ChanceConstraints; carry other constraints over, evaluated on μ.
    # Horizon-wide budget: one ε per (constraint, step, component), union-bound split.
    all_ccs = all_chance_constraints(sg)
    budget_eps = sg.joint_risk === nothing ? nothing :
        allocate_horizon_risk(risk_allocation, all_ccs, n_steps(nominal), sg.joint_risk)
    step_risks_of(c) = begin
        budget_eps === nothing && return nothing
        idx = findfirst(cc -> cc === c, all_ccs)
        idx === nothing ? nothing : budget_eps[idx]
    end
    time_index = t -> t isa Integer ? Int(t) : round(Int, t / dt) + 1
    compile_or_wrap(c::ChanceConstraint) =
        compile_chance_constraint(c, belief_fn, propagator, evaluator; risk_allocation,
                                  step_risks = step_risks_of(c), time_index)
    compile_or_wrap(c::AbstractPrivateInequality) =
        PrivateInequality(get_player(c);
            func = (z, u, p, t) -> evaluate_constraint(c, belief_fn(z)[1], u, p, t),
            dim = constraint_output_dim(c), depends_on_control = depends_on_control(c))
    compile_or_wrap(c::AbstractPrivateEquality) =
        PrivateEquality(get_player(c);
            func = (z, u, p, t) -> evaluate_constraint(c, belief_fn(z)[1], u, p, t),
            dim = constraint_output_dim(c))
    compile_or_wrap(c::AbstractSharedInequality) =
        SharedInequality(get_players(c);
            func = (z, u, p, t) -> evaluate_constraint(c, belief_fn(z)[1], u, p, t),
            dim = constraint_output_dim(c))
    compile_or_wrap(c::AbstractSharedEquality) =
        SharedEquality(get_players(c);
            func = (z, u, p, t) -> evaluate_constraint(c, belief_fn(z)[1], u, p, t),
            dim = constraint_output_dim(c))

    declared = sg.chance_constraints
    aug_private = [compile_or_wrap(c) for c in Iterators.flatten((nominal.private_constraints,
                                                                  filter(is_private, declared)))]
    aug_shared = [compile_or_wrap(c) for c in Iterators.flatten((nominal.shared_constraints,
                                                                 filter(is_shared, declared)))]

    aug_metadata = GameMetadata(
        fill(aug_dim, 1), nominal.metadata.control_dims,
        [0], nominal.metadata.control_offsets,
        nominal.metadata.coupling_graph, nominal.metadata.is_potential,
        nominal.metadata.potential_function, nominal.metadata.communication)

    GameProblem{T}(
        nominal.n_players, aug_objectives, aug_dynamics, z0,
        aug_private, aug_shared, th, aug_metadata)
end
