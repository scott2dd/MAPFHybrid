

#type this!!!!!!!!!!
# ********************************
function update_focal!(old_bound, new_bound, open_list, focal_list, focal_map)
    for heap_node in open_list.nodes
        label = heap_node.value #get label
        if label.fcost > old_bound && label.fcost <= new_bound
            focal_map[label.label_id] = focal_list = push!(focal_list, label)
        end
    end
end



# MAIN LABELING FUNCTION
function label_temporal_focal(env::HybridEnvironment, constraints::HybridConstraints, 
            agent_idx::Int64, initstate::HybridState, goal::Int64, eps::Float64, 
            focal_state_heuristic::Function, focal_transition_heuristic::Function)
    def = env.euc_inst
    Alist, F, C, Z = def.Alist, def.F, def.C, def.Z
    Bstart, Qstart = Int(floor(5 * mean(nonzeros(C)))), 9999
    Bmax = Bstart
    start, starttime = initstate.nodeIdx, initstate.time
    
    N = length(Alist)
    graph = env.orig_graph

    Fvec = fill(2^63 - 1, N)
    heur_astar = HybridUAVPlanning.get_heur_astar(goal, def.locs, Fvec)
    heur_astar(start)
    heur_label! = HybridUAVPlanning.get_heur_label(Fvec, graph, C, goal, heur_astar)
    if heur_label!(start) == 2^63 - 1
        printstyled("  subroutine: no path to goal!", color=:light_red)
        return nothing
    end

    #init state_to_idx and idx_to_state
    state_to_idx = Dict{Tuple{Int64,Int64},Int64}()
    idx_to_state = Dict{Int64,Tuple{Int64,Int64}}()
    state_to_idx[(start, starttime)] = 1 #add init state
    idx_to_state[1] = (start, 0)
    
    #now init path and gen tracking data
    came_from = [Vector{Int64}[] for _ in 1:N]
    push!(came_from[start], [0, 9999])
    gen_track = [Vector{Int64}[] for _ in 1:N]

    #init open list 
    open_list = MutableBinaryMinHeap{MyLabel}()
    focal_list = MutableBinaryHeap{MyLabel, MyLabelFocalCompare}()

    open_map, focal_map = Dict{Int64,Int64}(), Dict{Int64,Int64}()

    label_id_counter = 1
    init_label = MyLabel(
        batt_state=Bstart, 
        gen_state=Qstart, 
        gcost=0, 
        fcost = heur_label!(start), 
        hcost = heur_label!(start), 
        focal_heuristic=0, 
        state_idx=1, 
        prior_state_idx = 0, 
        node_idx = start,
        prior_node_idx = 0, 
        _hold_came_from_prior=0, 
        came_from_idx=1, 
        pathlength=1, 
        _hold_gen_track_prior=0, 
        gentrack_idx=1, 
        gen_bool=0, 
        label_id=label_id_counter)
    label_id_counter += 1
    
    open_map[init_label.label_id] = push!(open_list, init_label)
    focal_map[init_label.label_id] = push!(focal_list, init_label)

    fmin = Inf
    z = 0
    while true #loop until get to end node, or open_list is empty
        isempty(open_list) && (printstyled("open set empty, Z  = $z... \n", color=:light_cyan); break)
        fmin = top(open_list).fcost #keep to return for high level FOCAL
        
        labelN = top(focal_list) #top of focal_list
        open_handle = open_map[labelN.label_id] #grab handle from open_map

        pop!(focal_list) #remove from focal_list
        delete!(focal_map, labelN.label_id) #remove from focal_map
        delete!(open_list, open_handle) #emove from open_list
        delete!(open_map, labelN.label_id) #remove from open_map

        if labelN.node_idx == goal
            opt_cost = labelN.gcost
            opt_path = get_path(labelN, came_from, start)
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = get_gen(labelN, gen_track)

            plan = PlanResult(states=state_seq, actions=actions, cost=Int64(floor(opt_cost)), fmin=Int64(floor(fmin)), gen=gen)
            return plan
        end

        pathi = get_path(labelN, came_from, start)
        nodei = labelN.node_idx
        for nodej in Alist[nodei]
            nodej == nodei && continue
            nodej ∈  pathi && continue

            hj = heur_label!(nodej)
            Fbin = F[nodei, nodej] 

            #new state and add to both dicts (if we havent seen before)
            statei = idx_to_state[labelN.state_idx]
            newstate = (nodej, statei[2] + 1)

            #now check if j @ time is a constraints
            if VertexConstraint(newstate[2], newstate[1]) in constraints.vertex_constraints
                continue
            elseif EdgeConstraint(newstate[2] - 1, nodei, nodej) in constraints.edge_constraints || EdgeConstraint(newstate[2] - 1, nodej, nodej) in constraints.edge_constraints
                continue
            end

            if !haskey(state_to_idx, newstate) #if we not seen this state prior, add to dict
                state_to_idx[newstate] = length(state_to_idx) + 1#state id is order we see them
                newstate_idx = state_to_idx[newstate] #get idx for new state
                idx_to_state[newstate_idx] = newstate #now add that to opp dict
            end
            newstate_idx = state_to_idx[newstate]
            
            newhybridstate = HybridState(nodeIdx=newstate[1], time=newstate[2], b = 0, g = 0)
            oldhybridstate = HybridState(nodeIdx=nodei, time=statei[2],b = 0, g = 0)
            old_fh = labelN.focal_heuristic
            new_focal_heuristic = old_fh + focal_state_heuristic(newhybridstate) + focal_transition_heuristic(oldhybridstate, newhybridstate)

            #GEN ON
            new_batt_state = labelN.batt_state - C[nodei, nodej]*(1 - Fbin) + Z[nodei, nodej]
            new_gen_state = labelN.gen_state - Z[nodei, nodej]
            if def.GFlipped[nodei, nodej] == 0 && new_batt_state >= 0 && new_gen_state >= 0 && new_batt_state <= Bmax
                new_label = MyLabel(
                    gcost = labelN.gcost + C[nodei, nodej],
                    fcost = labelN.gcost + C[nodei, nodej] + hj,
                    hcost = hj,
                    focal_heuristic = new_focal_heuristic,
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
                if EFF_heap(open_list, new_label)
                    update_path_and_gen!(new_label, came_from, gen_track)
                    open_map[new_label.label_id] = push!(open_list, new_label)
                    if new_label.fcost <= fmin*eps #if in range then add to focal
                        focal_map[new_label.label_id] = push!(focal_list, new_label)
                    end
                end
            end
            
            #GEN OFF
            new_batt_state = labelN.batt_state - C[nodei, nodej]*(1 - Fbin)            
            if new_batt_state >= 0
                new_label = MyLabel(
                    gcost = labelN.gcost + C[nodei, nodej],
                    fcost = labelN.gcost + C[nodei, nodej] + hj,
                    hcost = hj,
                    focal_heuristic = new_focal_heuristic,
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
                label_id_counter += 1
                if EFF_heap(open_list, new_label)
                    update_path_and_gen!(new_label, came_from, gen_track)
                    open_map[new_label.label_id] = push!(open_list, new_label)
                    if new_label.fcost <= fmin * eps #if in range then add to focal
                        focal_map[new_label.label_id] = push!(focal_list, new_label)
                    end
                end
            end
        end

        if !isempty(open_list) && fmin * eps < top(open_list).fcost
            update_focal!(eps*fmin, eps*top(open_list).fcost, open_list, focal_list)
        end
        z += 1
        z == 200_000 && (printstyled("  ZBREAK@$(z)", color=:light_red); break)
        # z%1000 == 0 && ProgressMeter.next!(prog)
    end
    return nothing
end





