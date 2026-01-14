"""
AbstractIGNEP{T, Y, J, f, C} <: InverseDifferentialGame

Abstract base type for a Inverse Generalized Nash Equilibrium Problem.

# Type parameters
- `T` : time horizon type (e.g., `Float64`, `Int`, or `AbstractTimeRange`)
- TODO
"""
abstract type AbstractIGNEP{T} <: InverseDifferentialGame end