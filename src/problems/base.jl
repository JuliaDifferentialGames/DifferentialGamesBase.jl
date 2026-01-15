# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractGame{T}

Base type for all game-theoretic problems.

# Type Parameters
- `T` : Numeric type (Float64, Float32, etc.)

# Notes
All games involve:
- Multiple decision-makers (players)
- Strategic interaction (Nash equilibrium concept)
- Coupled or decoupled objectives
- Possible constraints on strategy spaces
"""
abstract type AbstractGame{T} end