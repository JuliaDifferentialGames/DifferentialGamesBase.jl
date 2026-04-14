module DifferentialGamesBase

using LinearAlgebra
using SparseArrays
using ForwardDiff

# ============================================================================
# Include order — dependency constraints:
#
#  1. abstract/game_problem.jl         — AbstractGameProblem hierarchy
#  2. abstract/information_structure.jl — AbstractInformationStructure
#  3. abstract/player_dynamics.jl      — AbstractPlayerDynamics
#  4. objectives/base_objectives.jl    — AbstractCost, NonlinearStageCost,
#                                        PlayerObjective
#  5. constraints.jl                   — AbstractConstraint hierarchy
#  6. metadata.jl                      — CouplingGraph, GameMetadata
#  7. dynamics/dynamics.jl             — LinearDynamics, SeparableDynamics,
#                                        CoupledNonlinearDynamics, accessors
#  8. time_horizon.jl                  — DiscreteTime, ContinuousTime
#  9. problems/player_spec.jl          — PlayerSpec
# 10. problems/GNEP.jl                 — LQStageCost, LQTerminalCost,
#                                        GameProblem, constructors,
#                                        validate_game_problem
# 11. problems/player_based_game.jl    — Player alias, DifferentialGame,
#                                        remake
# 12. abstract/strategy.jl             — AbstractStrategy, OpenLoopStrategy,
#                                        FeedbackStrategy
# 13. dynamics/discretization.jl       — DiscreteApproximation, discretize()
# 14. dynamics/dynamics_interface.jl   — evaluate_dynamics, rollout,
#                                        rollout_strategy
# 15. expansion/trajectory_expansion.jl — TrajectoryExpansion, expand(),
#                                        assemble_lq_game()
# 16. objectives/cost_terms.jl         — AbstractCostTerm, CompositeCostTerm,
#                                        player_slice, minimize()
# 17. objectives/standard_costs.jl     — QuadraticStateCost, ProximityCost,
#                                        track_goal, etc.
# 18. solutions/gnep_solutions.jl      — GNEPSolution, AbstractSolution,
#                                        Trajectory
# 19. solve.jl                         — GameSolver, solve(), WarmstartData
# 20. problems/IGNEP.jl                — (pending overhaul, commented out)
# 21. solutions/ignep_solutions.jl     — (pending overhaul, commented out)
# ============================================================================

include("abstract/game_problem.jl")
include("abstract/information_structure.jl")
include("abstract/player_dynamics.jl")
include("objectives/base_objectives.jl")
include("constraints/base_constraints.jl")
include("constraints/private_constraints.jl")
include("constraints/shared_constraints.jl")
include("constraints/standard_constraints.jl")
include("metadata.jl")
include("dynamics/dynamics.jl")
include("time_horizon.jl")
include("problems/player_spec.jl")
include("problems/GNEP.jl")
include("problems/player_based_game.jl")
include("abstract/strategy.jl")
include("dynamics/discretization.jl")
include("dynamics/dynamics_interface.jl")
include("expansion/trajectory_expansion.jl")
include("objectives/cost_terms.jl")
include("objectives/standard_costs.jl")
include("solutions/gnep_solutions.jl")
include("solve.jl")
# include("problems/IGNEP.jl")
# include("solutions/ignep_solutions.jl")

# ============================================================================
# Exports — Abstract Game Hierarchy
# ============================================================================
export
    AbstractGameProblem,
    AbstractDeterministicGame,
    AbstractStochasticGame,
    AbstractPartiallyObservableGame,
    AbstractInverseGameProblem,
    is_deterministic,
    is_stochastic,
    is_partially_observable,
    is_inverse

# ============================================================================
# Exports — Information Structure
# ============================================================================
export
    AbstractInformationStructure,
    PerfectStateInformation,
    OpenLoopInformation,
    PrivateObservation,
    SharedObservation,
    AsymmetricInformation,
    requires_belief_state,
    is_feedback_compatible,
    is_open_loop,
    _infer_game_class

# ============================================================================
# Exports — Abstract Player Dynamics
# ============================================================================
export
    AbstractPlayerDynamics,
    ContinuousPlayerDynamics,
    DiscretePlayerDynamics,
    CoupledPlayerDynamics,
    LinearPlayerDynamics,
    is_continuous,
    is_discrete,
    is_linear,
    is_separable_dynamics,
    evaluate_player_dynamics,
    player_dynamics_jacobian

# ============================================================================
# Exports — Strategy Hierarchy
# ============================================================================
export
    AbstractStrategy,
    OpenLoopStrategy,
    FeedbackStrategy,
    zero_open_loop_strategy,
    zero_feedback_strategy,
    apply_strategy,
    to_open_loop,
    get_nominal_control,
    get_gain,
    get_feedforward,
    get_nominal_state,
    get_times,
    get_control_dims,
    control_offsets,
    state_dim

# ============================================================================
# Exports — Discretization
# ============================================================================
export
    AbstractDiscretizationMethod,
    ZOHDiscretization,
    MatrixExpDiscretization,
    DiffEqDiscretization,
    DiscreteApproximation,
    discretize,
    da_step,
    jacobian,
    validate_discretization

# ============================================================================
# Exports — Dynamics Interface
# ============================================================================
export
    evaluate_dynamics,
    dynamics_jacobian,
    rollout,
    rollout_strategy

# ============================================================================
# Exports — Cost Term DSL
# ============================================================================
export
    AbstractCostTerm,
    AbstractTerminalCostTerm,
    CompositeCostTerm,
    CompositeTerminalCostTerm,
    evaluate_cost_term,
    cost_term_gradient,
    cost_term_hessian,
    is_quadratic,
    is_separable_term,
    player_slice,
    minimize

# ============================================================================
# Exports — Core Abstract Types
# ============================================================================
export
    DynamicsSpec,
    TimeHorizon,
    AbstractCost,
    AbstractStageCost,
    AbstractTerminalCost

# ============================================================================
# Exports — Dynamics Types and Accessors
# ============================================================================
export
    SeparableDynamics,
    LinearDynamics,
    CoupledNonlinearDynamics,
    get_A,
    get_B,
    get_B_concatenated,
    is_ltv,
    total_state_dim,
    total_control_dim

# ============================================================================
# Exports — Time Horizon
# ============================================================================
export
    ContinuousTime,
    DiscreteTime

# ============================================================================
# Exports — Cost Types
# ============================================================================
export
    LQStageCost,
    NonlinearStageCost,
    LQTerminalCost,
    NonlinearTerminalCost,
    PlayerObjective,
    get_Q, get_R, get_M, get_q, get_r,
    evaluate_stage_cost,
    evaluate_terminal_cost,
    stage_cost_gradient,
    terminal_cost_gradient,
    stage_cost_hessian,
    terminal_cost_hessian,
    total_cost,
    diagnose_scaling, 
    get_objective

# ============================================================================
# Exports — Constraint Hierarchy (Phase 4)
# ============================================================================
export
    AbstractConstraint,
    AbstractPrivateConstraint,
    AbstractPrivateInequality,
    AbstractPrivateEquality,
    AbstractSharedConstraint,
    AbstractSharedInequality,
    AbstractSharedEquality,
    is_private,
    is_shared,
    is_equality,
    is_inequality,
    get_player,
    get_players,
    evaluate_constraint,
    constraint_jacobian,
    is_active,
    constraint_violation,
    ControlBounds,
    StateBounds,
    PrivateNonlinearInequality,
    PrivateNonlinearEquality,
    PrivateInequality,
    PrivateEquality,
    ProximityConstraint,
    CommunicationConstraint,
    LinearCoupling,
    SharedNonlinearInequality,
    SharedNonlinearEquality,
    SharedInequality,
    SharedEquality,
    control_bounds,
    state_bounds,
    collision_avoidance,
    keep_in_range,
    linear_coupling

# ============================================================================
# Exports — Game Problem
# ============================================================================
export
    GameProblem,
    PlayerSpec,
    CouplingGraph,
    GameMetadata,
    PDGNEProblem,
    LQGameProblem,
    LTVLQGameProblem,
    validate_game_problem

# ============================================================================
# Exports — Property Queries
# ============================================================================
export
    has_separable_dynamics,
    is_lq_game,
    is_potential_game,
    has_shared_constraints,
    is_unconstrained,
    is_pd_gnep,
    is_separable,
    n_steps,
    n_players,
    num_players,
    control_dim

# ============================================================================
# Exports — Player and Game Constructors (Phase 2)
# ============================================================================
export
    Player,
    DifferentialGame,
    remake

# ============================================================================
# Exports — Standard Cost Terms (Phase 2d)
# ============================================================================
export
    QuadraticStateCost,
    QuadraticControlCost,
    ProximityCost,
    CommunicationCost,
    ControlBarrierCost,
    QuadraticTerminalCost,
    ProximityTerminalCost,
    track_goal,
    regularize_input,
    avoid_proximity,
    maintain_proximity,
    terminal_goal

# ============================================================================
# Exports — Solution Hierarchy
# ============================================================================
export
    AbstractSolution,
    GNEPSolution,
    Trajectory,
    get_trajectory,
    get_cost,
    get_costs,
    get_strategy,
    has_strategy,
    has_shared_state,
    is_feedback,
    is_open_loop_solution

# ============================================================================
# Exports — Solver Infrastructure
# ============================================================================
export
    WarmstartData,
    GameSolver,
    solve,
    _solve,
    solver_capabilities,
    required_capabilities

# ============================================================================
# Exports — Trajectory Expansion (Phase 3)
# ============================================================================
export
    DynamicsExpansion,
    CostExpansion,
    TrajectoryExpansion,
    linearize_dynamics,
    quadraticize_costs,
    expand,
    assemble_lq_game,
    reference_trajectory

end