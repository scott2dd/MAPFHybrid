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
        fmin = min(fmin, labelN.fcost)
        
        statei_idx = labelN.state_idx
        statei = idx_to_state[statei_idx]
        nodei = labelN.node_idx
        # print(nodei, raw" ")
        # P[statei] = vcat(P[nodei], reshape(label_treated,(1,9)))
        if nodei == goal
            opt_cost = labelN.gcost
            opt_path = get_path(labelN, came_from, start)
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = get_gen(labelN, gen_track)
            plan = PlanResult(states=state_seq, actions=actions, cost=Int(floor(opt_cost)), fmin=Int(floor(fmin)), gen=gen)
            return plan
            # label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
            # node = idx_to_state[label_copy[4]][1]
            # prior_node = idx_to_state[label_copy[5]][1]
            # label_copy[4], label_copy[5] = node, prior_node
            # opt_path = HybridUAVPlanning.get_path(label_copy, came_from, start)
            # state_seq, actions = get_state_path(opt_path, initstate, def.C)
            # gen = HybridUAVPlanning.get_gen(label_treated, gen_track)
        end

        # label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
        # prior_state = idx_to_state[label_copy[5]]
        # prior_node = prior_state[1]
        # label_copy[4], label_copy[5] = nodei, prior_node #switch to node, rather than state, just for path getting
        # pathi = HybridUAVPlanning.get_path(label_copy, came_from, start)
        pathi = get_path(labelN, came_from, start)
        nodei = labelN.node_idx
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
            # if def.GFlipped[nodei, nodej] == 0 && label_treated[2] - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej] >= 0 && label_treated[3] - Z[nodei, nodej] >= 0 && label_treated[2] - C[nodei, nodej] + Z[nodei, nodej] <= Bmax
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
                    batt_state = labelN.batt_state - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej],
                    gen_state = labelN.gen_state - Z[nodei, nodej],
                    label_id = label_id_counter
                )
                label_id_counter += 1

                if EFF_heap(Q, temp_new_label)
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
                if EFF_heap(Q, temp_new_label)
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

function label_temporal_high_IQ(env::HybridEnvironment, constraints::HybridConstraints, agent_idx::Int64, initstate::HybridState, goal::Int64)
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
    label_id = label_id_counter = 1
    label_init = MyLabel(
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
    Q = MutableBinaryMinHeap{MyLabel}()
    push!(Q, label_init)


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
        fmin = min(fmin, labelN.fcost)

        statei_idx = labelN.state_idx
        statei = idx_to_state[statei_idx]
        nodei = labelN.node_idx
        # P[statei] = vcat(P[nodei], reshape(label_treated,(1,9)))
        if nodei == goal
            opt_cost = labelN.gcost
            opt_path = get_path(labelN, came_from, start)
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = get_gen(labelN, gen_track)
            plan = PlanResult(states=state_seq, actions=actions, cost=Int(floor(opt_cost)), fmin=Int(floor(fmin)), gen=gen)
            return plan
            # label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
            # node = idx_to_state[label_copy[4]][1]
            # prior_node = idx_to_state[label_copy[5]][1]
            # label_copy[4], label_copy[5] = node, prior_node
            # opt_path = HybridUAVPlanning.get_path(label_copy, came_from, start)
            # state_seq, actions = get_state_path(opt_path, initstate, def.C)
            # gen = HybridUAVPlanning.get_gen(label_treated, gen_track)
        end

        # label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
        # prior_state = idx_to_state[label_copy[5]]
        # prior_node = prior_state[1]
        # label_copy[4], label_copy[5] = nodei, prior_node #switch to node, rather than state, just for path getting
        # pathi = HybridUAVPlanning.get_path(label_copy, came_from, start)
        pathi = get_path(labelN, came_from, start)
        nodei = labelN.node_idx
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
            # if def.GFlipped[nodei, nodej] == 0 && label_treated[2] - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej] >= 0 && label_treated[3] - Z[nodei, nodej] >= 0 && label_treated[2] - C[nodei, nodej] + Z[nodei, nodej] <= Bmax
            if def.GFlipped[nodei, nodej] == 0 && new_batt_state >= 0 && new_gen_state >= 0 && new_batt_state <= Bmax
                new_label = MyLabel(
                    gcost=labelN.gcost + C[nodei, nodej],
                    fcost=labelN.gcost + C[nodei, nodej] + hj,
                    hcost=hj,
                    focal_heuristic=0,
                    node_idx=nodej,
                    state_idx=newstate_idx,
                    prior_state_idx=labelN.state_idx,
                    prior_node_idx=labelN.node_idx,
                    _hold_came_from_prior=labelN.came_from_idx,
                    came_from_idx=-1, #fill in later! (in update func)
                    pathlength=labelN.pathlength + 1,
                    _hold_gen_track_prior=labelN.gentrack_idx,
                    gentrack_idx=-1,  #fill in later!
                    gen_bool=1,
                    batt_state=labelN.batt_state - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej],
                    gen_state=labelN.gen_state - Z[nodei, nodej],
                    label_id=label_id_counter
                )
                label_id_counter += 1

                if EFF_heap(Q, new_label)
                    update_path_and_gen!(new_label, came_from, gen_track)
                    push!(Q, new_label)
                end
            end

            #GEN OFF
            new_batt_state = labelN.batt_state - C[nodei, nodej] * (1 - Fbin)
            if new_batt_state >= 0
                new_label = MyLabel(
                    gcost=labelN.gcost + C[nodei, nodej],
                    fcost=labelN.gcost + C[nodei, nodej] + hj,
                    hcost=hj,
                    focal_heuristic=0,
                    node_idx=nodej,
                    state_idx=newstate_idx,
                    prior_state_idx=labelN.state_idx,
                    prior_node_idx=labelN.node_idx,
                    _hold_came_from_prior=labelN.came_from_idx,
                    came_from_idx=-1, #fill in later!
                    pathlength=labelN.pathlength + 1,
                    _hold_gen_track_prior=labelN.gentrack_idx,
                    gentrack_idx=-1, #fill in later!
                    gen_bool=0,
                    batt_state=labelN.batt_state - C[nodei, nodej] * (1 - Fbin),
                    gen_state=labelN.gen_state,
                    label_id=label_id_counter
                )
                if EFF_heap(Q, new_label)
                    update_path_and_gen!(new_label, came_from, gen_track)
                    push!(Q, new_label)
                end
            end
        end
        z += 1
        z == 200_000 && (printstyled("  ZBREAK@$(z)", color=:light_red); break)
        # z%500 == 0 && ProgressMeter.next!(prog)
    end
    return nothing
end


"""
    take the path and zip it into PlanResult format. Need cumulative cost and action cost (wij) along the path
"""
function get_state_path(pathL::Vector{Int64}, init_state::HybridState, C::SparseMatrixCSC{Int64, Int64})
    #for ease, assume all movements take 1 time step!
    pathLength = length(pathL)
        
    stateseq = Tuple{HybridState, Int64}[] #state plus cumulative cost to get there
    actions = Tuple{HybridAction, Int64}[] #action (which node we move to) and cost of that action
    state = deepcopy(init_state)
    running_cost = 0
    push!(stateseq, (state, running_cost))
    prior = state.nodeIdx
    for i in 2:length(pathL) #for each node in path, get the state
        node = pathL[i]
        state = HybridState(nodeIdx = node, time = state.time + 1, b=0, g=0)

        wij = C[prior, node]

        running_cost += wij
        push!(stateseq, (state, running_cost)) #new state plus cost
        #now get action to get to this ^^ state and the cost of that action (wij)
        if wij == 0
            push!(actions, (HybridAction(Wait, 0), 0))
        else
            push!(actions, (HybridAction(Move, node), wij))
        end
        prior = node
    end
    return stateseq, actions
end


"""
    input EucGraphInt instance
    constraints - vector of hybrid constraints for the problem (we only pick out our agent's constraints)
    agent_idx 
    removed heur input, assume we are always doing astar (for MAPF we definitely want this!)
"""
function label_temporal_low_IQ(env::HybridEnvironment, constraints::HybridConstraints, agent_idx::Int64, initstate::HybridState, goal::Int64) 
    #proc EucgraphInst
    def = env.euc_inst
    Alist, F, C, Z = def.Alist, def.F, def.C, def.Z
    Cmin = minimum(nonzeros(C)) 
    Bstart = Int(floor(5*mean(nonzeros(C))))
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
    state_to_idx = Dict{Tuple{Int64, Int64}, Int64}()
    idx_to_state = Dict{Int64, Tuple{Int64, Int64}}()
    #add init state here...
    state_to_idx[(start,starttime)] = 1
    idx_to_state[1] = (start,0)

    #init open and closed list
    Q = MutableBinaryMinHeap([   (heur_label!(start) + 0, 
        [0, Bstart, Qstart, 1 ,1,    1,1,1,   heur_label!(start) + 0])] 
        #g, B, g_f, state, priorstate, came_from_idx, pathlength, gentrack_idx, h
        ) 
    #label is: [partialcost, b, g, thisstate, priorstate, IDX in came_from[priorstate], pathlength, pathtrack_idx, pathtrack_idx, h]
    P = [zeros(Int, 0,9) for k = 1:size(C,1)] #set of treated labels | new labels compare to this AND Q

    came_from = [Vector{Int64}[] for _ in 1:N]
    push!(came_from[start], [0, 9999]) # so we just add a dummy entry to [start]
    gen_track = [Vector{Int64}[] for _ in 1:N]  #data struct to track generator patterns 
    #same paths will have same time, so we can have entries per node (rather than per state)

    # came_from = Dict{Int64, Vector{Vector{Int64}}}(0 => [[0,999999]])  #state_idx not node
    # gen_track = Dict{Int64, Vector{Vector{Int64}}}() #don't need to init... 
    #when adding to above, check if state exists (if it does, then should exist in the above)
    fmin = Inf
    z=0 
    while true #loop until get to end node, or Q is empty
        isempty(Q) && (printstyled("   Q empty, Z  = $z... \n", color=:light_cyan); break)

        #pull minimum cost label....
        next = pop!(Q)
        fmin = min(fmin, next[1])
        label_treated = next[2]
        statei_idx = label_treated[4]
        statei = idx_to_state[statei_idx]
        nodei = statei[1]
        # print(nodei, " ")
        # P[statei] = vcat(P[nodei], reshape(label_treated,(1,9)))
        if nodei == goal
            opt_cost =  label_treated[1]
            label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
            node = idx_to_state[label_copy[4]][1]
            prior_node = idx_to_state[label_copy[5]][1]
            label_copy[4], label_copy[5] = node, prior_node
            opt_path = HybridUAVPlanning.get_path(label_copy, came_from, start) 
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = HybridUAVPlanning.get_gen(label_treated, gen_track)
            plan = PlanResult(states = state_seq, actions=actions, cost=Int(floor(opt_cost)), fmin=Int(floor(fmin)), gen = gen)
            
            return plan
        end

        label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
        prior_state = idx_to_state[label_copy[5]]
        prior_node = prior_state[1]
        label_copy[4], label_copy[5] = nodei, prior_node #switch to node, rather than state, just for path getting
        
        pathi = HybridUAVPlanning.get_path(label_copy, came_from, start)

        for nodej in Alist[nodei]
            nodej==nodei && continue
            nodej∈pathi && continue
            
            hj =  heur_label!(nodej) 
            Fbin = F[nodei,nodej] # if we can glide, then ALWAYS glide
            
            #new state and add to both dicts (if we havent seen before)
            newstate = (nodej, statei[2]+1)

            #now check if j @ time is a constraints
            if VertexConstraint(newstate[2], newstate[1]) in constraints.vertex_constraints
                # println("vertex constraint hit!!")
                continue
            elseif EdgeConstraint(newstate[2]-1, nodei, nodej) in constraints.edge_constraints || EdgeConstraint(newstate[2]-1, nodej, nodej) in constraints.edge_constraints 
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
            if def.GFlipped[nodei,nodej] == 0 && label_treated[2]-C[nodei,nodej]*(1-Fbin) + Z[nodei,nodej] >= 0 && label_treated[3]-Z[nodei,nodej] >= 0 && label_treated[2]-C[nodei,nodej] + Z[nodei,nodej] <= Bmax
                label = Vector{Int64}(undef, 9)  
                label[1] = label_treated[1] + C[nodei, nodej]
                label[2] = label_treated[2] - C[nodei, nodej]*(1-Fbin) + Z[nodei, nodej] 
                label[3] = label_treated[3] -Z[nodei, nodej]
                label[4] = newstate_idx
                label[5] = label_treated[4] #prior label's NODE idx...
                label[6] = label_treated[6] #hold this for searching in came_from[prior]
                label[7] = label_treated[7] + 1 #path length
                label[8] = label_treated[8] #correct later
                label[9] = 1 #gen bool, correct later
                
                
                if HybridUAVPlanning.EFF_heap(Q, label) # && EFF_P(P, label)
                    #correct path...
                    pnode = idx_to_state[label[5]][1]
                    path_pointer = findall(x-> x == [pnode, label[6]], came_from[nodej])
                    if isempty(path_pointer) #if no other label has used this same path...
                        push!(came_from[nodej], [pnode, label[6]])   #[prior is i, in idx _k_ in came_from[i]]
                        label[6] = length(came_from[nodej])
                    else #if another label already used this same path, then we use that same pointer
                        pointer_idx = path_pointer[1]
                        label[6] = pointer_idx #label now has index for path in came_from 
                        priorstate = label[5]
                    end
                    #correct gen....
                    PL = label[7]
                    gen_pointer = findall(x -> x == [label[9], label[8]], gen_track[PL])  #label[9] is holding gen on/off
                    if isempty(gen_pointer)
                        push!(gen_track[PL], [label[9], label[8]])
                        label[8] = length(gen_track[PL])
                    else
                        pointer_idx = gen_pointer[1]
                        label[8] = pointer_idx #label now has index for generator pattern in gen_track
                    end

                    #correct hj
                    label[9] = label[1] + hj
                    #add to Q..
                    push!(Q, (label[1]+hj,label))

                end
            end
            #GEN OFF
            if label_treated[2]-C[nodei, nodej]*(1-Fbin) >= 0 
                # label =  [gc, label_treated[2]-C[nodei, nodej]*(1-Fbin), label_treated[3],  j, Int(length(X)+1), gc+h]
                label = Vector{Int64}(undef, 9)  
                label[1] = label_treated[1] + C[nodei, nodej]
                label[2] = label_treated[2] - C[nodei, nodej]*(1-Fbin)
                label[3] = label_treated[3] + 0
                label[4] = newstate_idx
                label[5] = label_treated[4] #prior label's state idx...
                label[6] = label_treated[6] #correct later
                label[7] = label_treated[7] + 1 
                label[8] = label_treated[8] #correct later
                label[9] = 0 #gen bool, correct later


                if HybridUAVPlanning.EFF_heap(Q, label) # && EFF_P(P, label)
                    #correct path...
                    pnode = idx_to_state[label[5]][1]
                    path_pointer = findall(x-> x == [pnode, label[6]], came_from[nodej])
                    if isempty(path_pointer) #if no other label has used this same path...
                        push!(came_from[nodej], [pnode, label[6]])   #[prior is i, in idx _k_ in came_from[i]]
                        label[6] = length(came_from[nodej])
                    else #if another label already used this same path, then we use that same pointer
                        pointer_idx = path_pointer[1]
                        label[6] = pointer_idx #label now has index for path in came_from 
                        priorstate = label[5]
                    end
                    #correct gen....
                    PL = label[7]
                    gen_pointer = findall(x -> x == [label[9], label[8]], gen_track[PL])  #label[9] is holding gen on/off
                    if isempty(gen_pointer)
                        push!(gen_track[PL], [label[9], label[8]])
                        label[8] = length(gen_track[PL])
                    else
                        pointer_idx = gen_pointer[1]
                        label[8] = pointer_idx #label now has index for generator pattern in gen_track
                    end

                    #correct hj
                    label[9] = label[1] + hj
                    push!(Q, (label[9],label))
                end
            end
        end
        z+=1
        z == 200_000 && (printstyled("  ZBREAK@$(z)", color=:light_red); break)
        # z%500 == 0 && ProgressMeter.next!(prog)
    end
    return nothing
end