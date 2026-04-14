using Test
using LinearAlgebra
using DifferentialGamesBase  

@testset "DifferentialGamesBase" begin
    include("test_phase0.jl")
    include("test_phase1.jl")
    include("test_phase2.jl")
    include("test_phase3.jl")
    include("test_phase4.jl")
end