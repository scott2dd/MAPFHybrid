function add_constraint!(cbase::HybridConstraints, cadd::HybridConstraints)

    union!(cbase.vertex_constraints, cadd.vertex_constraints)
    union!(cbase.edge_constraints, cadd.edge_constraints)

end

function overlap_between_constraints(cbase::HybridConstraints, cother::HybridConstraints)

    vertex_intersect = intersect(cbase.vertex_constraints, cother.vertex_constraints)
    edge_intersect = intersect(cbase.edge_constraints, cother.edge_constraints)

    return ~(isempty(vertex_intersect)) || ~(isempty(edge_intersect))

end



function get_mapf_state_from_idx(env::HybridEnvironment, idx::Int64)
    return env.state_graph.vertices[idx]
end

function get_mapf_action(env::HybridEnvironment, source::Int64, target::Int64)

    state1 = env.state_graph.vertices[source]
    state2 = env.state_graph.vertices[target]

    if equal_except_time(state1, state2)
        return HybridAction(Wait::Action, 0)
    else ##NEED TO: switch these to ONLY wait and move (move needs target)
        return Grid2DAction(Move::Action,target)
    end
    return nothing
end


function set_low_level_context!(env::HybridEnvironment, agent_idx::Int64, constraints::HybridConstraints)

    env.last_goal_constraint = -1
    env.agent_idx = agent_idx

    for vc in constraints.vertex_constraints
        if (vc.nodeIdx == env.goals[agent_idx])
            env.last_goal_constraint = max(env.last_goal_constraint, vc.time)
        end
    end
end

function admissible_heuristic_hybrid(env::HybridEnvironment, s::HybridState)
    return abs(s.x - env.goals[env.agent_idx].x) + abs(s.y - env.goals[env.agent_idx].y)
    return norm
end

function is_solution(env::HybridEnvironment, s::HybridState)
    return s.x == env.goals[env.agent_idx].x && s.y == env.goals[env.agent_idx].y &&
            s.time > env.last_goal_constraint
end


function get_state(agent_idx::Int64, solution::Vector{PlanResult{HybridState,HybridAction,Int64}}, rel_time::Int64)

    @assert agent_idx <= length(solution)

    # t must be relative time index from start of solution
    if rel_time < length(solution[agent_idx].states)
        return solution[agent_idx].states[rel_time+1][1]
    end

    # Just return the last state after verifying it is not empty
    @assert ~(isempty(solution[agent_idx].states))
    return solution[agent_idx].states[end][1]

end

function state_valid(env::HybridEnvironment, constraints::HybridConstraints, s::HybridState)

    con = constraints.vertex_constraints

    return s.nodeIdx >=1 && ~(HybridLocation(s.nodeIdx) in env.obstacles) &&
            ~(VertexConstraint(time=s.time, nodeIdx=s.nodeIdx) in con)
end

function transition_valid(env::HybridEnvironment, constraints::HybridConstraints, s1::HybridState, s2::HybridState)

    con = constraints.edge_constraints

    return ~(EdgeConstraint(time=s1.time, nodeIdx1=s1.nodeIdx, nodeIdx2 = s2.nodeIdx) in con)
end

function get_env_state_idx!(env::HybridEnvironment, s::HybridState)
    return s.nodeIdx
    idx = get(env.state_to_idx, s, 0)

    if idx == 0
        add_vertex!(env.state_graph, s)
        nbr_idx = num_vertices(env.state_graph)
        env.state_to_idx[s] = nbr_idx
        idx = nbr_idx
    end

    return idx
end


function get_first_conflict(env::HybridEnvironment, solution::Vector{PlanResult{HybridState,HybridAction,Int64}})

    max_rel_time = 0
    for sol in solution
        max_rel_time = max(max_rel_time, length(sol.states) - 1)
    end

    for rel_time = 0:max_rel_time-1
        # Drive-drive vertex collisions
        for i = 1:length(solution)-1
            state1 = get_state(i, solution, rel_time)
            for j = i+1:length(solution)
                state2 = get_state(j, solution, rel_time)

                if equal_except_time(state1, state2)
                    return HybridConflict(time=rel_time, agent_1=i, agent_2=j,
                                    type=Vertex::ConflictType, nodeIdx1 = state1.nodeIdx, nodeIdx2 = -99)
                end
            end
        end

        # Drive-drive edge
        for i = 1:length(solution)-1
            state1a = get_state(i, solution, rel_time)
            state1b = get_state(i, solution, rel_time + 1)

            for j = i+1:length(solution)
                state2a = get_state(j, solution, rel_time)
                state2b = get_state(j, solution, rel_time + 1)

                if equal_except_time(state1a, state2b) &&
                    equal_except_time(state1b, state2a)

                    return HybridConflict(time=rel_time, agent_1=i, agent_2=j, type=Edge::ConflictType, nodeIdx1 = state1a.nodeIdx, nodeIdx2 = state1b.nodeIdx)
                end
            end
        end
    end

    # No conflict
    return nothing
end

# Return a Dict{Int64, Grid2DConstraints}
function create_constraints_from_conflict(env::HybridEnvironment, conflict::HybridConflict)

    res_constraints = Dict{Int64, HybridConstraints}()

    if conflict.type == Vertex::ConflictType

        c1 = HybridConstraints()
        push!(c1.vertex_constraints, VertexConstraint(time=conflict.time, nodeIdx = conflict.nodeIdx1))
        res_constraints[conflict.agent_1] = c1
        res_constraints[conflict.agent_2] = c1

    elseif conflict.type == Edge::ConflictType

        c1 = HybridConstraints()
        push!(c1.edge_constraints, EdgeConstraint(time=conflict.time, 
        nodeIdx1 = conflict.nodeIdx1, nodeIdx2 = conflict.nodeIdx2))
        
        res_constraints[conflict.agent_1] = c1

        c2 = HybridConstraints()
        push!(c2.edge_constraints, EdgeConstraint(time=conflict.time, nodeIdx1 = conflict.nodeIdx2, nodeIdx2 = conflict.nodeIdx1))
        res_constraints[conflict.agent_2] = c2
    end

    return [res_constraints]
end


function low_level_search!(solver::CBSSolver, agent_idx::Int64, s::HybridState, constraints::HybridConstraints)
    env = solver.env

    # Reset env index
    env.curr_goal_idx = 0

    # Retrieve vertex index for state
    idx = get_env_state_idx!(env, s)

    # Set all edge weights to one
    ##NEED TO: check if we need this??? symmetric costs will make this harder for our labeling algo?
    edge_wt_fn(u, v) = 1

    # Set the heuristic
    heuristic(v) = admissible_heuristic_hybrid(env, v)

    # Run the search
    # @info "RUNNING SEARCH!"
    # vis = CBSGoalVisitorImplicit(env, constraints)
    vis = nothing
    #NEED TO:
    plan_result =  a_star_implicit_shortest_path!(env.orig_graph, env, s, agent_idx, constraints)
    # plan_result = astar_get_plan(env, pathout, state_seq, cost, fmin, cum_cost, ind_cost)

    # Return empty solution
    if plan_result == nothing
        return PlanResult{HybridState,HybridAction,Int64}()
    end

    return plan_result
end

# mutable struct CBSGoalVisitorImplicit <: AbstractDijkstraVisitor
#     env::HybridEnvironment
#     constraints::HybridConstraints
# end

function include_vertex!(env::HybridEnvironment, u::HybridState, v::HybridState, d::N, nbrs::Vector{Int64}) where {N <: Number}

    if is_solution(env, v)
        # @info "Found low-level solution!"
        env.curr_goal_idx = get_env_state_idx!(env, v)
        return false
    end

    # Need to generate neighbors of v using Actions
    empty!(nbrs)

    new_time = v.time + 1

    # Action = Wait
    temp_state = HybridState(time=new_time, nodeIdx=v.nodeIdx)
    if state_valid(vis.env, vis.constraints, temp_state) &&
        transition_valid(vis.env, vis.constraints, v, temp_state)
        nbr_idx = get_env_state_idx!(vis.env, temp_state)
        push!(nbrs, nbr_idx)
    end

    #for each neighbor... Do action move...
    for j in neighbors(vis.env.state_graph, v.nodeIdx)
        temp_state = HybridState(time=new_time, nodeIdx=j)
        if state_valid(vis.env, vis.constraints, temp_state) &&
            transition_valid(vis.env, vis.constraints, v, temp_state)
            nbr_idx = get_env_state_idx!(vis.env, temp_state)
            push!(nbrs, nbr_idx)
        end
    end
    # @show nbrs
    return true
end


