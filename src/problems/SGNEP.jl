"""
AbstractSGNEP{T, Y, J, f, C} <: DifferentialGame

Abstract base type for a Stochastic Generalized Nash Equilibrium Problem.

# Type parameters
- `T` : time horizon type (e.g., `Float64`, `Int`, or `AbstractTimeRange`)
- `Y` : strategy type (e.g., vector of controls, functions)
- `J` : payoff/cost functional type
- `f` : dynamics type (e.g., function `f(t, x, u)`)
- `C` : constraints type (e.g., vector of functions, sets)
- `W` : noise distribution type
"""
abstract type AbstractSGNEP{T, Y, J, f, C, W} <: DifferentialGame end