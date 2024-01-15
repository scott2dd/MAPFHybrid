# Only need to define heuristic stuff
function focal_state_heuristic_hybrid(env::HybridEnvironment, solution::Vector{PlanResult{HybridState,HybridAction,Int64}},
                                      s::HybridState)

    num_conflicts = 0

    for (i, agent_soln) in enumerate(solution)
        if i != env.agent_idx && ~(isempty(agent_soln))
            state2 = get_state(i, solution, s.time)
            if equal_except_time(s, state2)
                num_conflicts += 1
            end
        end
    end

    return num_conflicts
end


function focal_transition_heuristic_hybrid(env::HybridEnvironment,  solution::Vector{PlanResult{HybridState,HybridAction,Int64}},
                                           s1a::HybridState, s1b::HybridState)

    num_conflicts = 0

    for (i, agent_soln) in enumerate(solution)
        if i != env.agent_idx && ~(isempty(agent_soln))

            s2a = get_state(i, solution, s1a.time)
            s2b = get_state(i, solution, s1b.time)

            if equal_except_time(s1a, s2b) && equal_except_time(s1b, s2a)
                num_conflicts += 1
            end
        end
    end

    return num_conflicts
end

function focal_heuristic(env::HybridEnvironment, solution::Vector{PlanResult{HybridState,HybridAction,Int64}})

    num_conflicts = 0
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
                    num_conflicts += 1
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

                    num_conflicts += 1
                end
            end
        end
    end

    return num_conflicts
end

function low_level_search!(solver::ECBSSolver, agent_idx::Int64, statei::HybridState,
                           constraints::HybridConstraints, solution::Vector{PlanResult{HybridState,HybridAction,Int64}})

    env = solver.env

    env.curr_goal_idx = 0

    idx = get_env_state_idx!(env, statei)

    # Set the focal state heuristics
    focal_state_heuristic(s) = focal_state_heuristic_hybrid(solver.env, solution, s)
    focal_transition_heuristic(s1a, s1b) = focal_transition_heuristic_hybrid(solver.env, solution, s1a, s1b)

    # Run the search 
    goali = env.goals[agent_idx]
    plan_result = label_temporal_focal(env, constraints, agent_idx, statei, goali, solver.weight, focal_state_heuristic, focal_transition_heuristic)  
    
    if plan_result == nothing
        return PlanResult{HybridState,HybridAction,Int64}(), 0.0, 0.0
    end

    return plan_result, 0.0, 0.0
end
