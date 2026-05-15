using Documenter, DifferentialGamesBase

DocMeta.setdocmeta!(
    DifferentialGamesBase, :DocTestSetup,
    :(using DifferentialGamesBase, LinearAlgebra);
    recursive=true
)

makedocs(
    modules  = [DifferentialGamesBase],
    sitename = "DifferentialGamesBase.jl",
    authors  = "BennetOutland <bennet.outland@pm.me> and contributors",
    format   = Documenter.HTML(
        canonical  = "https://JuliaDifferentialGames.github.io/DifferentialGamesBase.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home"          => "index.md",
        "API Reference" => [
            "Problem Types"     => "api/problems.md",
            "Dynamics"          => "api/dynamics.md",
            "Costs & Objectives" => "api/costs.md",
            "Constraints"       => "api/constraints.md",
            "Solutions"         => "api/solutions.md",
            "Solver Interface"  => "api/solvers.md",
            "Inverse Games"     => "api/inverse.md",
        ],
    ],
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/JuliaDifferentialGames/DifferentialGamesBase.jl",
    devbranch = "main",
)
