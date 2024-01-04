function hybrid_label_temporal_focal(env::HybridEnvironment, constraints::HybridConstraints, agent_idx::Int64, initstate::HybridState, goal::Int64, , eps::Float64 = 1.0)
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

    #init open list 
    open_heap = MutableBinaryMinHeap([(heur_label!(start) + 0,
        [0, Bstart, Qstart, 1, 1, 1, 1, 1, heur_label!(start) + 0])])
    
        #init focal list
    focal_heap = MutableBinaryMinHeap(top(open_heap))

    P = [zeros(Int, 0, 9) for k = 1:size(C, 1)] #set of treated labels | new labels compare to this AND open_heap

    #now init path and gen tracking data
    came_from = [Vector{Int64}[] for _ in 1:N]
    push!(came_from[start], [0, 9999]) 
    gen_track = [Vector{Int64}[] for _ in 1:N]  
    

    fmin = Inf
    z = 0
    while true #loop until get to end node, or open_heap is empty
        isempty(open_heap) && (printstyled("open set empty, Z  = $z... \n", color=:light_cyan); break)

        #pull minimum cost label....
        next = pop!(open_heap)
        fmin = min(fmin, next[1])
        label_treated = next[2]
        statei_idx = label_treated[4]
        statei = idx_to_state[statei_idx]
        nodei = statei[1]
        # P[statei] = vcat(P[nodei], reshape(label_treated,(1,9)))
        if nodei == goal
            opt_cost = label_treated[1]
            label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
            node = idx_to_state[label_copy[4]][1]
            prior_node = idx_to_state[label_copy[5]][1]
            label_copy[4], label_copy[5] = node, prior_node
            opt_path = get_path(label_copy, came_from, start)
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = get_gen(label_treated, gen_track)
            plan = PlanResult(states=state_seq, actions=actions, cost=Int(floor(opt_cost)), fmin=Int(floor(fmin)), gen=gen)

            return plan
        end

        label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
        prior_state = idx_to_state[label_copy[5]]
        prior_node = prior_state[1]
        label_copy[4], label_copy[5] = nodei, prior_node #switch to node, rather than state, just for path getting

        pathi = get_path(label_copy, came_from, start)

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
            if def.GFlipped[nodei, nodej] == 0 && label_treated[2] - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej] >= 0 && label_treated[3] - Z[nodei, nodej] >= 0 && label_treated[2] - C[nodei, nodej] + Z[nodei, nodej] <= Bmax
                label = Vector{Int64}(undef, 9)
                label[1] = label_treated[1] + C[nodei, nodej]
                label[2] = label_treated[2] - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej]
                label[3] = label_treated[3] - Z[nodei, nodej]
                label[4] = newstate_idx
                label[5] = label_treated[4] #prior label's NODE idx...
                label[6] = label_treated[6] #hold this for searching in came_from[prior]
                label[7] = label_treated[7] + 1 #path length
                label[8] = label_treated[8] #correct later
                label[9] = 1 #gen bool, correct later


                if HybridUAVPlanning.EFF_heap(open_heap, label) # && EFF_P(P, label)
                    #correct path...
                    pnode = idx_to_state[label[5]][1]
                    path_pointer = findall(x -> x == [pnode, label[6]], came_from[nodej])
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
                    #add to open_heap..
                    push!(open_heap, (label[1] + hj, label))

                end
            end
            #GEN OFF
            if label_treated[2] - C[nodei, nodej] * (1 - Fbin) >= 0
                # label =  [gc, label_treated[2]-C[nodei, nodej]*(1-Fbin), label_treated[3],  j, Int(length(X)+1), gc+h]
                label = Vector{Int64}(undef, 9)
                label[1] = label_treated[1] + C[nodei, nodej]
                label[2] = label_treated[2] - C[nodei, nodej] * (1 - Fbin)
                label[3] = label_treated[3] + 0
                label[4] = newstate_idx
                label[5] = label_treated[4] #prior label's state idx...
                label[6] = label_treated[6] #correct later
                label[7] = label_treated[7] + 1
                label[8] = label_treated[8] #correct later
                label[9] = 0 #gen bool, correct later


                if HybridUAVPlanning.EFF_heap(open_heap, label) # && EFF_P(P, label)
                    #correct path...
                    pnode = idx_to_state[label[5]][1]
                    path_pointer = findall(x -> x == [pnode, label[6]], came_from[nodej])
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
                    push!(open_heap, (label[9], label))
                end
            end
        end
        z += 1
        z == 200_000 && (printstyled("  ZBREAK@$(z)", color=:light_red); break)
        # z%500 == 0 && ProgressMeter.next!(prog)
    end
    return nothing
end