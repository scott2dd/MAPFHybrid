"""
    input EucGraphInt instance
    constraints - vector of hybrid constraints for the problem (we only pick out our agent's constraints)
    agent_idx 
    removed heur input, assume we are always doing astar (for MAPF we definitely want this!)
"""
function hybrid_label_temporal(def::EucGraphInt, constraints::HybridConstraints, agent_idx::Int64, start::Int64, goal::Int64, Bstart::Int64 = 0, Gstart::Int64 = 0) 
    #proc EucgraphInst
    Alist, F, C, Z = def.Alist, def.F, def.C, def.Z
    Cmin = minimum(nonzeros(C)) 
    Bstart = Int(floor(5*mean(nonzeros(C))))
    Qstart = 9999
    Bmax = Bstart
    SC = 0
    locs = def.locs
    
    N = length(Alist)
    graph = make_graph(def)
    Fvec = fill(2^63 - 1, N)
    heur_astar = get_heur_astar(goal, locs, Fvec)
    heur_astar(start)
    heur_label! = get_heur_label(Fvec, graph, C, goal, heur_astar)
    
    if heur_label!(start) == 2^63 - 1     
        printstyled("  subroutine: no path to goal!", color=:light_red)
        return -1,[0], Bool.([0]) #infeasible!
    end
    
    #init state_to_idx and idx_to_state
    #a state is (node, time) -> (Int, Int)
    #we can reach a state (node, time) with differing fuel/energy levels
    state_to_idx = Dict{Tuple{Int64, Int64}, Int64}()
    idx_to_state = Dict{Int64, Tuple{Int64, Int64}}()
    #add init state here...
    state_to_idx[(start,0)] = 1
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
    z=0 
    while true #loop until get to end node, or Q is empty
        isempty(Q) && (printstyled("Q empty, Z  = $z... \n", color=:light_cyan); break)

        #pull minimum cost label....
        next = pop!(Q)
        label_treated = next[2]
        statei_idx = label_treated[4]
        statei = idx_to_state[statei_idx]
        nodei = statei[1]
        # P[statei] = vcat(P[nodei], reshape(label_treated,(1,9)))
        if nodei == goal
            opt_cost =  label_treated[1]
            label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
            node = idx_to_state[label_copy[4]][1]
            prior_node = idx_to_state[label_copy[5]][1]
            label_copy[4], label_copy[5] = node, prior_node
            opt_path = get_path(label_copy, came_from, start) 
            # opt_gen = get_gen(label_treated, gen_track)
            return opt_cost, opt_path
        end

        label_copy = deepcopy(label_treated) #make copy with nodes instead of state idxs to backtrack path and state
        prior_state = idx_to_state[label_copy[5]]
        prior_node = prior_state[1]
        label_copy[4], label_copy[5] = nodei, prior_node #switch to node, rather than state, just for path getting
        
        pathi = get_path(label_copy, came_from, start) 

        for nodej in Alist[nodei]
            nodej==nodei && continue
            nodej∈pathi && continue
            
            hj =  heur_label!(nodej) 
            Fbin = F[nodei,nodej] # if we can glide, then ALWAYS glide
            
            #new state and add to both dicts (if we havent seen before)
            newstate = (nodej, statei[2]+1)

            #now check if j @ time is a constraints
            if VertexConstraint(newstate[2], newstate[1]) in constraints.vertex_constraints
                println("vertex constraint hit!!")
                continue
            elseif EdgeConstraint(newstate[2]-1, nodei, nodej) in constraints.edge_constraints || EdgeConstraint(newstate[2]-1, nodej, nodej) in constraints.edge_constraints 
                println("edge constraint hit!!")
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
                
                
                if EFF_heap(Q, label) # && EFF_P(P, label)
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


                if EFF_heap(Q, label) # && EFF_P(P, label)
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
        z == 200_000 && (printstyled("ZBREAK@$(z)", color=:light_red); break)
        # z%500 == 0 && ProgressMeter.next!(prog)
    end
    return 0,[0], Bool.([0])
end