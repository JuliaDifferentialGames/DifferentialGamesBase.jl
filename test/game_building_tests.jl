using Test
using LinearAlgebra
using SparseArrays
using DifferentialGamesBase  

@testset "Game Problem Construction" begin
    
    @testset "Unconstrained LQ Game" begin
        # Two-player LQ game with shared state space
        n = 4  # State dimension
        A = [0.0 1.0 0.0 0.0;
             0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 1.0;
             0.0 0.0 0.0 0.0]
        
        B1 = [0.0; 1.0; 0.0; 0.0]
        B2 = [0.0; 0.0; 0.0; 1.0]
        B = [reshape(B1, n, 1), reshape(B2, n, 1)]
        
        Q1 = diagm([1.0, 0.1, 1.0, 0.1])
        Q2 = diagm([1.0, 0.1, 1.0, 0.1])
        Q = [Q1, Q2]
        
        R1 = reshape([0.1], 1, 1)
        R2 = reshape([0.1], 1, 1)
        R = [R1, R2]
        
        Qf1 = diagm([10.0, 1.0, 10.0, 1.0])
        Qf2 = diagm([10.0, 1.0, 10.0, 1.0])
        Qf = [Qf1, Qf2]
        
        x0 = [1.0, 0.0, -1.0, 0.0]
        tf = 5.0
        
        # Test LQGameProblem constructor
        game = LQGameProblem(A, B, Q, R, Qf, x0, tf, dt=0.1)
        
        @test game isa GameProblem{Float64}
        @test num_players(game) == 2
        @test state_dim(game) == 4
        @test control_dim(game) == 2
        @test is_lq_game(game)
        @test is_unconstrained(game)
        @test !is_pd_gnep(game)  # Shared state space
        @test game.dynamics isa LinearDynamics
        
        # Test UnconstrainedLQGame convenience constructor
        game2 = UnconstrainedLQGame(A, B, Q, R, Qf, x0, tf, dt=0.1)
        
        @test game2 isa GameProblem{Float64}
        @test is_lq_game(game2)
        @test is_unconstrained(game2)
    end
    
    @testset "PD-GNEP with Separable Dynamics" begin
        # Two spacecraft with decoupled dynamics, coupled formation cost
        
        # Player 1: 6-DOF spacecraft (position + velocity)
        n1, m1 = 6, 3
        x0_1 = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        function dynamics_1(xi, ui, p, t)
            # ẋ = v, v̇ = u
            return [xi[4:6]; ui]
        end
        
        Q1 = diagm([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # Penalize velocity
        R1 = diagm(0.1 * ones(3))
        stage_cost_1 = LQStageCost(Q1, R1)
        
        Qf1 = diagm([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        terminal_cost_1 = LQTerminalCost(Qf1)
        
        objective_1 = PlayerObjective(1, stage_cost_1, terminal_cost_1)
        
        player1 = PlayerSpec(1, n1, m1, x0_1, dynamics_1, objective_1)
        
        # Player 2: Similar spacecraft
        n2, m2 = 6, 3
        x0_2 = [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        function dynamics_2(xi, ui, p, t)
            return [xi[4:6]; ui]
        end
        
        Q2 = diagm([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        R2 = diagm(0.1 * ones(3))
        stage_cost_2 = LQStageCost(Q2, R2)
        
        Qf2 = diagm([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        terminal_cost_2 = LQTerminalCost(Qf2)
        
        objective_2 = PlayerObjective(2, stage_cost_2, terminal_cost_2)
        
        player2 = PlayerSpec(2, n2, m2, x0_2, dynamics_2, objective_2)
        
        # Collision avoidance constraint (shared)
        d_safe = 5.0
        function collision_constraint(x, u, p, t)
            # Extract positions
            pos1 = x[1:3]
            pos2 = x[7:9]  # Offset by n1
            dist_sq = sum((pos1 - pos2).^2)
            return [d_safe^2 - dist_sq]  # <= 0 means collision-free
        end
        
        collision = NonlinearConstraint(
            collision_constraint,
            1,
            constraint_type = :inequality
        )
        shared_collision = SharedConstraint(collision, [1, 2])
        
        # Build PD-GNEP
        game = PDGNEProblem([player1, player2], [shared_collision], 10.0, 0.1)
        
        @test game isa GameProblem{Float64}
        @test num_players(game) == 2
        @test state_dim(game) == 12  # 6 + 6
        @test control_dim(game) == 6  # 3 + 3
        @test is_pd_gnep(game)
        @test has_separable_dynamics(game)
        @test has_shared_constraints(game)
        @test !is_lq_game(game)  # Nonlinear constraint breaks LQ property
        @test game.dynamics isa SeparableDynamics
        @test length(game.shared_constraints) == 1
    end
    
    @testset "LQ PD-GNEP (Separable LQ)" begin
        # Heterogeneous players with separable linear dynamics and LQ costs
        
        # Player 1: Double integrator (2D)
        n1, m1 = 4, 2
        x0_1 = [1.0, 0.0, 0.0, 0.0]
        
        function dynamics_1(xi, ui, p, t)
            # [x, y, vx, vy]
            return [xi[3]; xi[4]; ui[1]; ui[2]]
        end
        
        Q1 = diagm([1.0, 1.0, 0.1, 0.1])
        R1 = diagm([0.1, 0.1])
        stage_1 = DiagonalLQStageCost([1.0, 1.0, 0.1, 0.1], [0.1, 0.1])
        terminal_1 = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])
        objective_1 = PlayerObjective(1, stage_1, terminal_1)
        
        player1 = PlayerSpec(1, n1, m1, x0_1, dynamics_1, objective_1)
        
        # Player 2: Single integrator (3D)
        n2, m2 = 3, 3
        x0_2 = [0.0, 0.0, 0.0]
        
        function dynamics_2(xi, ui, p, t)
            return ui  # Velocity control
        end
        
        stage_2 = DiagonalLQStageCost([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
        terminal_2 = DiagonalLQTerminalCost([10.0, 10.0, 10.0])
        objective_2 = PlayerObjective(2, stage_2, terminal_2)
        
        player2 = PlayerSpec(2, n2, m2, x0_2, dynamics_2, objective_2)
        
        # Build unconstrained LQ PD-GNEP
        game = PDGNEProblem([player1, player2], 5.0, 0.05)
        
        @test game isa GameProblem{Float64}
        @test is_pd_gnep(game)
        @test is_unconstrained(game)
        @test has_separable_dynamics(game)
        @test state_dim(game) == 7
        @test control_dim(game) == 5
        
        # All costs are LQ (diagonal variant)
        for obj in game.objectives
            @test obj.stage_cost isa DiagonalLQStageCost
            @test obj.terminal_cost isa DiagonalLQTerminalCost
        end
    end
    
    @testset "Coupled Nonlinear Game" begin
        # General nonlinear game with coupled dynamics
        n = 3
        m_total = 4  # Player 1: 2 controls, Player 2: 2 controls
        
        # Coupled nonlinear dynamics
        function coupled_dynamics(x, u, p, t)
            # Nonlinear coupling: x1 affects x2's dynamics
            u1 = u[1:2]
            u2 = u[3:4]
            
            dx1 = x[2] + 0.1 * sin(x[3])
            dx2 = u1[1] - 0.5 * x[1]
            dx3 = u1[2] + u2[1] + 0.2 * x[2]^2
            
            return [dx1; dx2; dx3]
        end
        
        dynamics = CoupledNonlinearDynamics(coupled_dynamics, n, m_total)
        
        # Nonlinear costs for each player
        function cost_1(x, u, p, t)
            u1 = u[1:2]
            return x[1]^2 + 0.5 * x[2]^2 + 0.1 * (u1' * u1)
        end
        
        function cost_2(x, u, p, t)
            u2 = u[3:4]
            return x[3]^2 + 0.1 * (u2' * u2)
        end
        
        stage_1 = NonlinearStageCost(cost_1, is_separable=false)
        terminal_1 = NonlinearTerminalCost(x -> x[1]^2 + x[2]^2)
        objective_1 = PlayerObjective(1, stage_1, terminal_1)
        
        stage_2 = NonlinearStageCost(cost_2, is_separable=false)
        terminal_2 = NonlinearTerminalCost(x -> x[3]^2)
        objective_2 = PlayerObjective(2, stage_2, terminal_2)
        
        objectives = [objective_1, objective_2]
        
        x0 = [1.0, 0.5, -0.5]
        time_horizon = DiscreteTime(3.0, 0.1)
        
        # Build metadata
        state_dims = [n, 0]  # Shared state, player 2 has no separate state
        control_dims = [2, 2]
        state_offsets = [0, 0]
        control_offsets = [0, 2]
        
        cost_coupling = sparse(trues(2, 2))  # Dense coupling
        coupling_graph = CouplingGraph(cost_coupling, Vector{Int}[], nothing)
        
        metadata = GameMetadata(
            state_dims,
            control_dims,
            state_offsets,
            control_offsets,
            coupling_graph,
            false,
            nothing
        )
        
        game = GameProblem{Float64}(
            2,
            objectives,
            dynamics,
            x0,
            PrivateConstraint[],
            SharedConstraint[],
            time_horizon,
            metadata
        )
        
        @test game isa GameProblem{Float64}
        @test !is_pd_gnep(game)  # Coupled dynamics
        @test !is_lq_game(game)  # Nonlinear
        @test is_unconstrained(game)
        @test game.dynamics isa CoupledNonlinearDynamics
    end
    
    @testset "Game with Private Constraints" begin
        # PD-GNEP with control bounds (private constraints)
        
        n1, m1 = 2, 1
        x0_1 = [1.0, 0.0]
        
        dynamics_1 = (xi, ui, p, t) -> [xi[2]; ui[1]]
        
        stage_1 = DiagonalLQStageCost([1.0, 0.1], [0.1])
        terminal_1 = DiagonalLQTerminalCost([10.0, 1.0])
        objective_1 = PlayerObjective(1, stage_1, terminal_1)
        
        # Control bounds: -1 <= u1 <= 1
        u_max = 1.0
        bounds_1 = BoundConstraint([-u_max], [u_max], applies_to=:u)
        private_bounds_1 = PrivateConstraint(bounds_1, 1)
        
        player1 = PlayerSpec(1, n1, m1, x0_1, dynamics_1, objective_1, 
                            [private_bounds_1])
        
        # Player 2: similar
        n2, m2 = 2, 1
        x0_2 = [-1.0, 0.0]
        
        dynamics_2 = (xi, ui, p, t) -> [xi[2]; ui[1]]
        
        stage_2 = DiagonalLQStageCost([1.0, 0.1], [0.1])
        terminal_2 = DiagonalLQTerminalCost([10.0, 1.0])
        objective_2 = PlayerObjective(2, stage_2, terminal_2)
        
        bounds_2 = BoundConstraint([-u_max], [u_max], applies_to=:u)
        private_bounds_2 = PrivateConstraint(bounds_2, 2)
        
        player2 = PlayerSpec(2, n2, m2, x0_2, dynamics_2, objective_2,
                            [private_bounds_2])
        
        game = PDGNEProblem([player1, player2], 5.0, 0.1)
        
        @test game isa GameProblem{Float64}
        @test is_pd_gnep(game)
        @test !is_unconstrained(game)
        @test !has_shared_constraints(game)
        @test length(game.private_constraints) == 2
        
        # Extract player 1's constraints
        p1_constraints = filter(c -> c.player == 1, game.private_constraints)
        @test length(p1_constraints) == 1
        @test p1_constraints[1].constraint isa BoundConstraint
    end
    
    @testset "Three-Player Formation Game" begin
        # Three spacecraft in formation with pairwise collision avoidance
        
        players = PlayerSpec{Float64}[]
        shared_constraints = SharedConstraint[]
        
        # Create three identical players
        for i in 1:3
            n, m = 6, 3
            x0_i = [10.0 * cos(2π * i / 3), 10.0 * sin(2π * i / 3), 0.0,
                    0.0, 0.0, 0.0]
            
            dynamics_i = (xi, ui, p, t) -> [xi[4:6]; ui]
            
            Q = diagm([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            R = diagm(0.1 * ones(3))
            stage = LQStageCost(Q, R)
            
            Qf = diagm([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
            terminal = LQTerminalCost(Qf)
            
            objective_i = PlayerObjective(i, stage, terminal)
            
            player_i = PlayerSpec(i, n, m, x0_i, dynamics_i, objective_i)
            push!(players, player_i)
        end
        
        # Pairwise collision avoidance
        d_safe = 3.0
        for i in 1:3
            for j in (i+1):3
                offset_i = (i - 1) * 6
                offset_j = (j - 1) * 6
                
                collision_ij = NonlinearConstraint(
                    (x, u, p, t) -> begin
                        pos_i = x[offset_i .+ (1:3)]
                        pos_j = x[offset_j .+ (1:3)]
                        [d_safe^2 - sum((pos_i - pos_j).^2)]
                    end,
                    1,
                    constraint_type = :inequality
                )
                
                push!(shared_constraints, SharedConstraint(collision_ij, [i, j]))
            end
        end
        
        game = PDGNEProblem(players, shared_constraints, 8.0, 0.1)
        
        @test num_players(game) == 3
        @test state_dim(game) == 18  # 3 * 6
        @test control_dim(game) == 9  # 3 * 3
        @test length(game.shared_constraints) == 3  # C(3,2) = 3 pairs
        @test is_pd_gnep(game)
        @test has_shared_constraints(game)
    end
    
    @testset "Property Queries" begin
        # Test all property query functions on various game types
        
        # LQ game
        A = diagm([0.0, 0.0])
        B = [reshape([1.0; 0.0], 2, 1), reshape([0.0; 1.0], 2, 1)]
        Q = [diagm([1.0, 1.0]), diagm([1.0, 1.0])]
        R = [reshape([0.1], 1, 1), reshape([0.1], 1, 1)]
        Qf = Q
        
        lq_game = LQGameProblem(A, B, Q, R, Qf, [1.0, 1.0], 5.0)
        
        @test is_lq_game(lq_game)
        @test !is_pd_gnep(lq_game)
        @test is_unconstrained(lq_game)
        @test !is_potential_game(lq_game)
        
        # PD-GNEP
        player = PlayerSpec(
            1, 2, 1, [1.0, 0.0],
            (xi, ui, p, t) -> [xi[2]; ui[1]],
            PlayerObjective(
                1,
                DiagonalLQStageCost([1.0, 0.1], [0.1]),
                DiagonalLQTerminalCost([10.0, 1.0])
            )
        )
        
        pd_game = PDGNEProblem([player], 3.0, 0.1)
        
        @test is_pd_gnep(pd_game)
        @test has_separable_dynamics(pd_game)
        @test is_unconstrained(pd_game)
        @test !is_potential_game(pd_game)
    end
    
    @testset "Metadata and Indexing" begin
        # Test dimension calculations and offsets
        
        players = [
            PlayerSpec(
                1, 4, 2, zeros(4),
                (xi, ui, p, t) -> [xi[3:4]; ui],
                PlayerObjective(
                    1,
                    DiagonalLQStageCost(ones(4), 0.1 * ones(2)),
                    DiagonalLQTerminalCost(10.0 * ones(4))
                )
            ),
            PlayerSpec(
                2, 6, 3, zeros(6),
                (xi, ui, p, t) -> [xi[4:6]; ui],
                PlayerObjective(
                    2,
                    DiagonalLQStageCost(ones(6), 0.1 * ones(3)),
                    DiagonalLQTerminalCost(10.0 * ones(6))
                )
            )
        ]
        
        game = PDGNEProblem(players, 5.0, 0.1)
        
        @test state_dim(game, 1) == 4
        @test state_dim(game, 2) == 6
        @test control_dim(game, 1) == 2
        @test control_dim(game, 2) == 3
        
        @test game.metadata.state_offsets == [0, 4]
        @test game.metadata.control_offsets == [0, 2]
        
        obj1 = get_objective(game, 1)
        @test obj1.player_id == 1
        
        obj2 = get_objective(game, 2)
        @test obj2.player_id == 2
    end
end