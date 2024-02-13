function label_temporal(env::HybridEnvironment, constraints::HybridConstraints, agent_idx::Int64, initstate::HybridState, goal::Int64)
    #proc EucgraphInst
    def = env.euc_inst
    Alist, F, C, Z = def.Alist, def.F, def.C, def.Z
    Cmin = minimum(nonzeros(C))
    Bstart = Int(floor(5 * mean(nonzeros(C))))
    Qstart = 9999
    Bmax = Bstart
    SC = 0
    locs = def.locs

    start, starttime = initstate.nodeIdx, initstate.time
    # println("agent $agent_idx || start: $start || goal: $goal")
    N = length(Alist)
    graph = env.orig_graph
    Fvec = fill(2^63 - 1, N)
    heur_astar = HybridUAVPlanning.get_heur_astar(goal, locs, Fvec)
    heur_astar(start)
    heur_label! = HybridUAVPlanning.get_heur_label(Fvec, graph, C, goal, heur_astar)

    if heur_label!(start) == 2^63 - 1
        printstyled("  subroutine: no path to goal!", color=:light_red)
        return nothing
    end

    #init state_to_idx and idx_to_state
    #a state is (node, time) -> (Int, Int)
    #we can reach a state (node, time) with differing fuel/energy levels
    state_to_idx = Dict{Tuple{Int64,Int64},Int64}()
    idx_to_state = Dict{Int64,Tuple{Int64,Int64}}()
    #add init state here...
    state_to_idx[(start, starttime)] = 1
    idx_to_state[1] = (start, 0)

    #init open and closed list
    # Q = MutableBinaryMinHeap([(heur_label!(start) + 0,
        # [0, Bstart, Qstart, 1, 1, 1, 1, 1, heur_label!(start) + 0])]
    #g, B, g_f, state, priorstate, came_from_idx, pathlength, gentrack_idx, h
    # )
    label_id=label_id_counter = 1
    label_init = MyLabelImmut(
        batt_state=Bstart,
        gen_state=Qstart,
        gcost=0,
        fcost=heur_label!(start),
        hcost=heur_label!(start),
        focal_heuristic=0,
        state_idx=1,
        prior_state_idx=0,
        node_idx=start,
        prior_node_idx=0,
        _hold_came_from_prior=0,
        came_from_idx=1,
        pathlength=1,
        _hold_gen_track_prior=0,
        gentrack_idx=1,
        gen_bool=0,
        label_id=label_id_counter
    )
    label_id_counter += 1
    Q = MutableBinaryMinHeap{MyLabelImmut}()
    push!(Q, label_init)

    
    P = Dict{Int64, Set{AbbreviatedLabel}}()
    
    came_from = [Tuple{Int64,Int64}[] for _ in 1:N]
    push!(came_from[start], (0, 9999)) # so we just add a dummy entry to [start]
    gen_track = [Tuple{Int64,Int64}[] for _ in 1:N]  #data struct to track generator patterns 
    #same paths will have same time, so we can have entries per node (rather than per state)

    fmin = Inf
    z = 0
    while true #loop until get to end node, or Q is empty
        isempty(Q) && (printstyled("   Q empty, Z  = $z... \n", color=:light_cyan); break)
        #pull minimum cost label....
        labelN = pop!(Q)
        !haskey(P, labelN.state_idx) && (P[labelN.state_idx] = Set{AbbreviatedLabel}())
        push!(P[labelN.state_idx], abbreviated_label(labelN))
        fmin = min(fmin, labelN.fcost)
        
        statei_idx = labelN.state_idx
        statei = idx_to_state[statei_idx]
        nodei = labelN.node_idx
        pathi = get_path(labelN, came_from, start)
        if nodei == goal
            opt_cost = labelN.gcost
            opt_path = get_path(labelN, came_from, start)
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = get_gen(labelN, gen_track)
            plan = PlanResult(states=state_seq, actions=actions, cost=Int(floor(opt_cost)), fmin=Int(floor(fmin)), gen=gen)
            return plan
        end

        
        for nodej in Alist[nodei]
            nodej == nodei && continue
            nodej ∈ pathi && continue

            hj = heur_label!(nodej)
            Fbin = F[nodei, nodej] # if we can glide, then ALWAYS glide

            #new state and add to both dicts (if we havent seen before)
            newstate = (nodej, statei[2] + 1)

            #now check if j @ time is a constraints
            if VertexConstraint(newstate[2], newstate[1]) in constraints.vertex_constraints
                # println("vertex constraint hit!!")
                continue
            elseif EdgeConstraint(newstate[2] - 1, nodei, nodej) in constraints.edge_constraints || EdgeConstraint(newstate[2] - 1, nodej, nodej) in constraints.edge_constraints
                # println("edge constraint hit!!")
                continue
            end

            if !haskey(state_to_idx, newstate)
                state_to_idx[newstate] = length(state_to_idx) + 1
                newstate_idx = state_to_idx[newstate]
                idx_to_state[newstate_idx] = newstate
            end
            newstate_idx = state_to_idx[newstate]

            #GEN ON
            new_batt_state = labelN.batt_state - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej]
            new_gen_state = labelN.gen_state - Z[nodei, nodej]
            if def.GFlipped[nodei, nodej] == 0 && new_batt_state >= 0 && new_gen_state >= 0 && new_batt_state <= Bmax
                temp_new_label = MyLabelImmut(
                    gcost = labelN.gcost + C[nodei, nodej],
                    fcost = labelN.gcost + C[nodei, nodej] + hj,
                    hcost = hj,
                    focal_heuristic = 0,
                    node_idx = nodej,
                    state_idx = newstate_idx,
                    prior_state_idx = labelN.state_idx,
                    prior_node_idx = labelN.node_idx,
                    _hold_came_from_prior = labelN.came_from_idx,
                    came_from_idx = -1, #fill in later! (in update func)
                    pathlength = labelN.pathlength + 1,
                    _hold_gen_track_prior = labelN.gentrack_idx,
                    gentrack_idx = -1,  #fill in later!
                    gen_bool = 1,
                    batt_state = new_batt_state,
                    gen_state = new_gen_state,
                    label_id = label_id_counter
                )
                label_id_counter += 1

                if EFF_heap(Q, temp_new_label) && (!haskey(P, newstate_idx) || EFF_P(P[newstate_idx], abbreviated_label(temp_new_label))) #if no key, will not call EFF_P (short circuit) so we don't need to worry about P[newstate_idx] being empty! as it won't be called in that case
                    new_label = update_path_and_gen!(temp_new_label, came_from, gen_track)
                    push!(Q, new_label)
                end
            end
            
            #GEN OFF
            new_batt_state = labelN.batt_state - C[nodei, nodej]*(1 - Fbin)            
            if new_batt_state >= 0
                temp_new_label = MyLabelImmut(
                    gcost = labelN.gcost + C[nodei, nodej],
                    fcost = labelN.gcost + C[nodei, nodej] + hj,
                    hcost = hj,
                    focal_heuristic = 0,
                    node_idx = nodej,
                    state_idx = newstate_idx,
                    prior_state_idx = labelN.state_idx,
                    prior_node_idx = labelN.node_idx,
                    _hold_came_from_prior = labelN.came_from_idx,
                    came_from_idx = -1, #fill in later!
                    pathlength = labelN.pathlength + 1,
                    _hold_gen_track_prior = labelN.gentrack_idx,
                    gentrack_idx = -1, #fill in later!
                    gen_bool = 0,
                    batt_state = labelN.batt_state - C[nodei, nodej] * (1 - Fbin),
                    gen_state = labelN.gen_state,
                    label_id = label_id_counter
                )
                if EFF_heap(Q, temp_new_label) && (!haskey(P, newstate_idx) || EFF_P(P[newstate_idx], abbreviated_label(temp_new_label))) #see above comment about short circuit
                    new_label = update_path_and_gen!(temp_new_label, came_from, gen_track)
                    push!(Q, new_label)
                end
            end
        end
        z += 1
        z == 1_000_000 && (printstyled("ZBREAK", color=:light_red); break)
        # z%500 == 0 && ProgressMeter.next!(prog)
    end
    return nothing
end
