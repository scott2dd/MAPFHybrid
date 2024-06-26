"""
    CBSHighLevelNode{S <: MAPFState, A <: MAPFAction, C <: Number, CNR <: MAPFConstraints}

Maintains the information for a high-level search tree node in CBS.

Attributes:
    - `solution::Vector{PlanResult{S,A,C}}` The current full solution encoded in the node
    - `constraints::Vector{CNR}` The set of constraints imposed on the node
    - `cost::C` The cost of the node's full solution
    - `id::Int64` A unique ID for the node in the high-level tree
"""
@with_kw mutable struct CBSHighLevelNode{S <: MAPFState, A <: MAPFAction, C <: Number, CNR <: MAPFConstraints}
    solution::Vector{PlanResult{S,A,C}} = Vector{PlanResult{S,A,C}}(undef, 0)
    constraints::Vector{CNR}            = Vector{CNR}(undef,0)
    cost::C                             = zero(C)
    id::Int64                           = 0
end

Base.isless(hln1::CBSHighLevelNode, hln2::CBSHighLevelNode) = hln1.cost < hln2.cost

# Functions required to be implemented by environment
"""
    set_low_level_context!(env::MAPFEnvironment, agent_idx::Int64, constraints::MAPFConstraints)

Update any contextual information before running the low-level search for an agent
"""
function set_low_level_context! end

"""
    get_first_conflict(env::MAPFEnvironment, solution::Vector{PlanResult})

Analyze the solution vector and return the first path-path conflict in it.
"""
function get_first_conflict end

"""
    create_constraints_from_conflict(env::MAPFEnvironment, conflict::MAPFConflict)

Given the MAPF conflict information, generate the corresponding low-level
search constraints on the individual agents.
"""
function create_constraints_from_conflict end

"""
    overlap_between_constraints(cbase::MAPFConstraints, cother::MAPFConstraints)

Return true if there is any overlap between the two constraint set arguments.
"""
function overlap_between_constraints end

"""
    add_constraint!(cbase::MAPFConstraints, cadd::MAPFConstraints)

Augment the set of constraints cbase in-place with new ones to add.
"""
function add_constraint! end

"""
    low_level_search!(solver::CBSSolver, agent_idx::Int64, s::MAPFState, constraints::MAPFConstraints)

Implement the actual low level search for a single agent on the environment. The search can be any of
the implicit A* variants in https://github.com/Shushman/Graphs.jl/tree/master/src
"""
function low_level_search! end


@with_kw mutable struct CBSSolver{S <: MAPFState, A <: MAPFAction, C <: Number, HC <: HighLevelCost,
                                  F <: MAPFConflict, CNR <: MAPFConstraints, E <: MAPFEnvironment}
    env::E
    hlcost::HC                                                 = HC()
    heap::MutableBinaryMinHeap{CBSHighLevelNode{S,A,C,CNR}}    = MutableBinaryMinHeap{CBSHighLevelNode{S,A,C,CNR}}()
end

"""
search!(solver::CBSSolver{S,A,C,HC,F,CNR,E}, initial_states::Vector{S}) where {S <: MAPFState, A <: MAPFAction, C <: Number, HC <: HighLevelCost,
                                                                               F <: MAPFConflict, CNR <: MAPFConstraints, E <: MAPFEnvironment}

    Calls the CBS Solver on the given problem.
"""
function search!(solver::CBSSolver{S,A,C,HC,F,CNR,E}, initial_states::Vector{S}; time_lim::Float64 = 120.0) where 
        {S <: MAPFState, A <: MAPFAction, C <: Number, HC <: HighLevelCost, 
         F <: MAPFConflict, CNR <: MAPFConstraints, E <: MAPFEnvironment}
    
    times_subroutine = Float64[] 
    times_astar = Float64[]
    time_start = time()
    num_agents = length(initial_states)

    start = CBSHighLevelNode{S,A,C,CNR}(solution = Vector{PlanResult{S,A,C}}(undef, num_agents),
                             constraints = Vector{CNR}(undef, num_agents))
    
    for idx = 1:num_agents

        start.constraints[idx] = get_empty_constraint(CNR)

        # Get constraints from low level context
        set_low_level_context!(solver.env, idx, start.constraints[idx])

        # Calls get_plan_result_from_astar within
        new_solution, tlabel, tastar = low_level_search!(solver, idx, initial_states[idx], start.constraints[idx])
        push!(times_subroutine, tlabel)
        push!(times_astar, tastar)
        # println("new solution for $(idx): ", new_solution)
        # Return empty solution if cannot find
        if isempty(new_solution)
            return start, 1, times_subroutine, times_astar
        end

        start.solution[idx] = new_solution
    end

    start.cost = compute_cost(solver.hlcost, start.solution)
    push!(solver.heap, start)

    id = 1

    while ~(isempty(solver.heap))
        P = pop!(solver.heap)
        conflict = get_first_conflict(solver.env, P.solution)
        
        if conflict == nothing #If no conflict, we are done as this is best node in tree
            return P,id, times_subroutine, times_astar
        elseif time() - time_start > time_lim && id > 1
            P.cost = -1
            return P, id, times_subroutine, times_astar
        end

        # Create additional nodes to resolve conflict (which is not nothing)
        constraints = create_constraints_from_conflict(solver.env, conflict)
        for constraint in constraints
            for (i, c) in constraint
                new_node = deepcopy(P)
                new_node.id = id

                add_constraint!(new_node.constraints[i], c)

                # Redo search with new constraint
                new_node.cost = deaccumulate_cost(solver.hlcost, new_node.cost, new_node.solution[i].cost)

                set_low_level_context!(solver.env, i, new_node.constraints[i])

                new_solution,tlabel, tastar = low_level_search!(solver, i, initial_states[i], new_node.constraints[i])
                push!(times_subroutine, tlabel)
                push!(times_astar, tastar)


                # Only create new node if we found a solution
                if !isempty(new_solution)  #nothing if Zbreak, no path to goal, or Q empty...

                    new_node.solution[i] = new_solution
                    new_node.cost = accumulate_cost(solver.hlcost, new_node.cost, new_solution.cost)
                    push!(solver.heap, new_node)
                    id += 1
                #else (if new_solution == nothing) then just don't add this node to tree (is dead)
                end
            end
        end
    end

    #If no nodes left to expand, then return an empty solution
    blanknode = CBSHighLevelNode{S,A,C,CNR}()
    blanknode.cost = -2 #flag to indicate no solution found
    return blanknode, id, times_subroutine, times_astar

end
