mutable struct MyLabel
    gcost::Float64
    fcost::Float64
    hcost::Float64
    focal_heuristic::Float64
    state_idx::Int64
    node_idx::Int64
    prior_state_idx::Int64
    prior_node_idx::Int64
    _hold_came_from_prior::Int64
    came_from_idx::Int64
    pathlength::Int64
    _hold_gen_track_prior::Int64
    gentrack_idx::Int64
    gen_bool::Int64 #on/off to get to this state (prior edge)
    batt_state::Float64
    gen_state::Float64
    label_id::Int64
end

#now define isless for MyLabels, for ordering in a heap
Base.isless(l1::MyLabel, l2::MyLabel) = l1.fcost < l2.fcost

#now define lexico ordering for MyLabels for focal heap (used 2 param ordering)
struct MyLabelFocalCompare <: Base.Order.Ordering end
Base.lt(o::MyLabelFocalCompare, n1::MyLabel, n2::MyLabel) =
    (n1.focal_heurisitc, n1.fcost) < (n2.focal_heuristic_value, n2.fcost)


function get_path(label::MyLabel, came_from::Vector{Vector{Int64}}, start::Int64)
    path = Int64[]
    here = label.nodeidx
    cf_idx_here = label.came_from_idx

    here == start && return [start]
    push!(path, here)
    while here != start
        next = came_from[here][cf_idx_here][1]
        cf_idx_next = came_from[here][cf_idx_here][2]
        
        push!(path, next)
        here = copy(next)
        cf_idx_here = copy(cf_idx_next)
    end
    return reverse(path)
end
function get_gen(label::MyLabel, gen_track::Vector{Vector{Int64}})
    #get generator pattern from recursive data struct
    genOut = Bool[]
    PL = label.pathlength #path length of tracked label (in # of nodes) ... so if PL=1 then no edges, 
    gt_idx = label.gentrack_idx #index for gen_track
    while PL > 1
        gen_now = gen_track[PL][gt_idx][1]
        push!(genOut, gen_now)
        gt_idx = gen_track[PL][gt_idx][2]
        PL -= 1
    end
    return reverse(genOut)
end

function update_path_and_gen!(new_label::MyLabel, came_from::Vector{Vector{Int64}}, gen_track::Vector{Vector{Int64}})
    #correct path...
    pnode = new_label.prior_node_idx
    p_came_from_idx = new_label._hold_came_from_prior
    path_pointer = findall(x -> x == [pnode, p_came_from_idx], came_from[nodej])
    if isempty(path_pointer) #if no other label has used this same path...
        push!(came_from[nodej], [pnode, p_came_from_idx])
        new_label.came_from_idx = length(came_from[nodej])
    else #if path exists prior, then we use the (nonempty) pointer
        pointer_idx = path_pointer[1]
        new_label.came_from_idx = pointer_idx #label now has index for came_from 
    end

    #correct gen....
    gen_pointer = findall(x -> x == [new_label.gen_bool, new_label._hold_gen_track_prior], gen_track[new_label.pathlength])
    if isempty(gen_pointer)
        push!(gen_track[PL], [new_label.gen_bool, new_label._hold_gen_track_prior])
        new_label.gentrack_idx = length(gen_track[PL])
    else
        pointer_idx = gen_pointer[1]
        new_label.gentrack_idx = pointer_idx #label now has index for gen_track
    end

end

function update_focal!(old_bound, new_bound, open_list, focal_list)
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
            focal_state_heuristic, focal_transition_heuristic)
    def = env.euc_inst
    Alist, F, C, Z = def.Alist, def.F, def.C, def.Z
    Bstart, Qstart = Int(floor(5 * mean(nonzeros(C)))), 9999
    Bmax = Bstart
    
    
    # println("agent $agent_idx || start: $start || goal: $goal")
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
    start, starttime = initstate.nodeIdx, initstate.time
    state_to_idx[(start, starttime)] = 1 #add init state
    idx_to_state[1] = (start, 0)
    
    #now init path and gen tracking data
    came_from = [Vector{Int64}[] for _ in 1:N]
    push!(came_from[start], [0, 9999])
    gen_track = [Vector{Int64}[] for _ in 1:N]

    #init open list 
    open_list = MutableBinaryMinHeap{MyLabel}()
    focal_list = MutableBinaryMinHeap{MyLabel, MyLabelFocalCompare}()
    open_map, focal_map = Dict{Int64,Int64}(), Dict{Int64,Int64}()

    label_id_counter = 1
    init_label = MyLabel(batt_state=Bstart, gen_state=Qstart, gcost=0, fcost = heur_label!(start), hcost = heur_label!(start), focal_heuristic=0, state_idx=1, prior_state_idx = 1, prior_node_idx = 1, _hold_came_from_prior=0, came_from_idx=1, pathlength=1, _hold_gen_track_prior=0, gentrack_idx=1, gen_bool=0, label_id=label_id_counter)
    label_id_counter += 1
    
    open_map[start] = push!(open_list, init_label)
    focal_map[start] = push!(focal_list, init_label)

    fmin = 0
    z = 0
    while true #loop until get to end node, or open_heap is empty
        isempty(open_heap) && (printstyled("open set empty, Z  = $z... \n", color=:light_cyan); break)

        fmin = top(open_list).fcost #keep to return for high level FOCAL
        labelN = pop!(open_heap)
        delete!(open_list, open_map[labelN.label_id]) #remove from open_list

        if labelN.node_idx == goal
            opt_cost = labelN.gcost
            opt_path = get_path(labelN, came_from, start)
            state_seq, actions = get_state_path(opt_path, initstate, def.C)
            gen = get_gen(labelN, gen_track)
            plan = PlanResult(states=state_seq, actions=actions, cost=Int(floor(opt_cost)), fmin=fmin, gen=gen)
            return plan
        end

        pathi = get_path(labelN, came_from, start)

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

            #Make label with GEN ON
            new_batt_state = labelN.batt_state - C[nodei, nodej]*(1 - Fbin) + Z[nodei, nodej]
            new_gen_state = labelN.gen_state - Z[nodei, nodej]
            if def.GFlipped[nodei, nodej] == 0 && new_batt_state >= 0 && new_gen_state >= 0 && new_batt_state <= Bmax
                
                new_label = MyLabel(
                    gcost = labelN.gcost + C[nodei, nodej],
                    fcost = labelN.gcost + C[nodei, nodej] + hj,
                    hcost = hj,
                    focal_heuristic = 0
                    state_idx = newstate_idx,
                    prior_state_idx = labelN.state_idx,
                    prior_node_idx = labelN.node_idx,
                    _hold_came_from_prior = labelN.came_from_idx,
                    came_from_idx = NaN, #fill in later!
                    pathlength = labelN.pathlength + 1,
                    _hold_gen_track_prior = labelN.gentrack_idx,
                    gentrack_idx = NaN,  #fill in later!
                    gen_bool = 1,
                    batt_state = labelN.batt_state - C[nodei, nodej] * (1 - Fbin) + Z[nodei, nodej],
                    gen_state = labelN.gen_state - Z[nodei, nodej],
                    label_id = label_id_counter
                )
                label_id_counter += 1
                if HybridUAVPlanning.EFF_heap(open_heap, new_label)
                    update_path_and_gen!(new_label, came_from, gen_track)
                    open_map[new_label.label_id] = push!(open_list, new_label)
                    if new_label.fcost < fmin*eps #if in range then add to focal
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
                    focal_heuristic = 0
                    state_idx = newstate_idx,
                    prior_state_idx = labelN.state_idx,
                    prior_node_idx = labelN.node_idx,
                    _hold_came_from_prior = labelN.came_from_idx,
                    came_from_idx = NaN, #fill in later!
                    pathlength = labelN.pathlength + 1,
                    _hold_gen_track_prior = labelN.gentrack_idx,
                    gentrack_idx = NaN, #fill in later!
                    gen_bool = 0,
                    batt_state = labelN.batt_state - C[nodei, nodej] * (1 - Fbin),
                    gen_state = labelN.gen_state,
                    label_id = label_id_counter
                )
                label_id_counter += 1
                if HybridUAVPlanning.EFF_heap(open_heap, new_label)
                    update_path_and_gen!(new_label, came_from, gen_track)
                    open_map[new_label.label_id] = push!(open_list, new_label)
                    if new_label.fcost < fmin * eps #if in range then add to focal
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





