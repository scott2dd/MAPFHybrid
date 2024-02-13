@kwdef struct MyLabelImmut <: Label
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

#isless is defined in HybridUAVPlanning.struct_defs... but we can rewrite here for our own specific ordering!
Base.isless(l1::Label, l2::Label) = (l1.fcost, l1.gcost) < (l2.fcost, l2.gcost)

#now define lexico ordering for MyLabels for focal heap (used 2 param ordering)
struct MyLabelFocalCompare <: Base.Order.Ordering end
Base.lt(o::MyLabelFocalCompare, n1::Label, n2::Label) =
    (n1.focal_heuristic, n1.fcost, n1.gcost) < (n2.focal_heuristic, n2.fcost, n2.gcost)


# We must redefine EFF_heap -> becuase we need to look at state_idx for comparisons rather than node_idx.....
function EFF_heap(Q::MutableBinaryMinHeap{L}, label_new::L) where L<:Label 
    isempty(Q) && (return true)
    node_map_copy = Q.node_map
    for k in 1:length(node_map_copy)
        node_map_copy[k] == 0 && continue
        Q[k].state_idx != label_new.state_idx && continue #if they are different STATES, then skip...

        (Q[k].gcost <= label_new.gcost && Q[k].batt_state >= label_new.batt_state && Q[k].gen_state >= label_new.gen_state) && (return false)
    end
    return true #if all this passes, then return true (is efficient)
end

#remove these later.... just commeting out before deleting in case we need them.....
# function get_path(label::Label, came_from::Vector{Vector{Tuple{Int64,Int64}}}, start::Int64)
#     path = Int64[]
#     here = label.node_idx
#     cf_idx_here = label.came_from_idx

#     here == start && return [start]
#     push!(path, here)
#     while here != start
#         next = came_from[here][cf_idx_here][1]
#         cf_idx_next = came_from[here][cf_idx_here][2]

#         push!(path, next)
#         here = copy(next)
#         cf_idx_here = copy(cf_idx_next)
#     end
#     return reverse(path)
# end
# function get_gen(label::Label, gen_track::Vector{Vector{Tuple{Int64,Int64}}})
#     #get generator pattern from recursive data struct
#     genOut = Bool[]
#     PL = label.pathlength #path length of tracked label (in # of nodes) ... so if PL=1 then no edges, 
#     gt_idx = label.gentrack_idx #index for gen_track
#     while PL > 1
#         gen_now = gen_track[PL][gt_idx][1]
#         push!(genOut, gen_now)
#         gt_idx = gen_track[PL][gt_idx][2]
#         PL -= 1
#     end
#     return reverse(genOut)
# end


function update_path_and_gen!(new_label::MyLabelImmut, came_from::Vector{Vector{Tuple{Int64,Int64}}}, gen_track::Vector{Vector{Tuple{Int64,Int64}}})
    #correct path...
    pnode = new_label.prior_node_idx
    nextnode = new_label.node_idx
    p_came_from_idx = new_label._hold_came_from_prior
    path_pointer = findall(x -> x == [pnode, p_came_from_idx], came_from[nextnode])
    if isempty(path_pointer) #if no other label has used this same path...
        push!(came_from[nextnode], (pnode, p_came_from_idx))
        came_from_idx = length(came_from[nextnode])
    else #if path exists prior, then we use the (nonempty) pointer
        pointer_idx = path_pointer[1]
        came_from_idx = pointer_idx #label now has index for came_from 
    end

    #correct gen....
    PL = new_label.pathlength
    gen_pointer = findall(x -> x == [new_label.gen_bool, new_label._hold_gen_track_prior], gen_track[new_label.pathlength])
    if isempty(gen_pointer)
        push!(gen_track[PL], (new_label.gen_bool, new_label._hold_gen_track_prior))
        gentrack_idx = length(gen_track[PL])
    else
        pointer_idx = gen_pointer[1]
        gentrack_idx = pointer_idx #label now has index for gen_track
    end

    label_updated = MyLabelImmut(
        gcost=new_label.gcost,
        fcost=new_label.fcost,
        hcost=new_label.hcost,
        focal_heuristic=new_label.focal_heuristic,
        state_idx=new_label.state_idx,
        node_idx=new_label.node_idx,
        prior_state_idx=new_label.prior_state_idx,
        prior_node_idx=new_label.prior_node_idx,
        _hold_came_from_prior=new_label._hold_came_from_prior,
        came_from_idx=came_from_idx,
        pathlength=new_label.pathlength,
        _hold_gen_track_prior=new_label._hold_gen_track_prior,
        gentrack_idx=gentrack_idx,
        gen_bool=new_label.gen_bool,
        batt_state=new_label.batt_state,
        gen_state=new_label.gen_state,
        label_id=new_label.label_id
    )
    return label_updated
end


@kwdef mutable struct MyLabel <: Label
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
function update_path_and_gen!(new_label::MyLabel, came_from::Vector{Vector{Tuple{Int64,Int64}}}, gen_track::Vector{Vector{Tuple{Int64,Int64}}})
    #correct path...
    pnode = new_label.prior_node_idx
    nextnode = new_label.node_idx
    p_came_from_idx = new_label._hold_came_from_prior
    path_pointer = findall(x -> x == [pnode, p_came_from_idx], came_from[nextnode])
    if isempty(path_pointer) #if no other label has used this same path...
        push!(came_from[nextnode], (pnode, p_came_from_idx))
        new_label.came_from_idx = length(came_from[nextnode])
    else #if path exists prior, then we use the (nonempty) pointer
        pointer_idx = path_pointer[1]
        new_label.came_from_idx = pointer_idx #label now has index for came_from 
    end

    #correct gen....
    PL = new_label.pathlength
    gen_pointer = findall(x -> x == [new_label.gen_bool, new_label._hold_gen_track_prior], gen_track[new_label.pathlength])
    if isempty(gen_pointer)
        push!(gen_track[PL], (new_label.gen_bool, new_label._hold_gen_track_prior))
        new_label.gentrack_idx = length(gen_track[PL])
    else
        pointer_idx = gen_pointer[1]
        new_label.gentrack_idx = pointer_idx #label now has index for gen_track
    end

end


## END OF MY ADDED CODE !



"""
    PlanResult{S <: MAPFState, A <: MAPFAction, C <: Number}

Stores the result of an individual agent's search.

Attributes:
    - `states::Vector{Tuple{S,C}}` The list of (state, cost-to-go) on the path
    - `actions::Vector{Tuple{A,C}}` The list of (actions, cost) on the path.
    - `cost::C` The cumulative cost of the path
    - `fmin::C` The minimum f-value expanded during search
"""
@with_kw struct PlanResult{S <: MAPFState, A <: MAPFAction, C <: Number}
    states::Vector{Tuple{S,C}}  = Vector{Tuple{S,C}}(undef, 0)
    actions::Vector{Tuple{A,C}} = Vector{Tuple{A,C}}(undef, 0)
    gen::Vector{Int64}          = Vector{Int64}(undef, 0)
    cost::C                     = zero(C)
    fmin::C                     = zero(C)
end

Base.isempty(pln::PlanResult) = isempty(pln.states)

"""
    get_mapf_state_from_idx(env::E, idx::Int64) where {E <: MAPFEnvironment}

My A* search implementation returns a list of graph indices; this function maps an index
to the corresponding state for that environment.
"""
function get_mapf_state_from_idx end

"""
    get_mapf_action(env::E, source::Int64, target::Int64) where {E <: MAPFEnvironment}

Returns the MAPFAction while going from source index to target index.
"""
function get_mapf_action end

"""
    get_plan_result_from_astar(env::E, a_star_dists::Dict, a_star_parent_indices::Dict,
                               start_idx::Int64, goal_idx::Int64,
                               best_fvalue::D) where {E <: MAPFEnvironment, D <: Number}

Converts the result of an A*-search (from my Graphs.jl implementation) to a corresponding PlanResult instance.
"""
function get_plan_result_from_astar(env::E, a_star_dists::Dict, a_star_parent_indices::Dict,
                                    start_idx::Int64, goal_idx::Int64,
                                    best_fvalue::D) where {E <: MAPFEnvironment, D <: Number}

    # First ensure the goal is reachable
    if ~(haskey(a_star_dists, goal_idx))
        return nothing
    end

    # Set both cost and fmin
    cost = a_star_dists[goal_idx]
    if best_fvalue == zero(D)
        fmin = a_star_dists[goal_idx]
    else
        fmin = best_fvalue
    end

    # Insert last elements to states and actions arrays
    goal_state = get_mapf_state_from_idx(env, goal_idx)
    states = [(goal_state, cost)]

    curr_idx = a_star_parent_indices[goal_idx]

    goal_action = get_mapf_action(env, curr_idx, goal_idx)
    action_cost = a_star_dists[goal_idx] - a_star_dists[curr_idx]
    actions = [(goal_action, action_cost)]

    # Now walk back to start
    while curr_idx != start_idx
        prev_idx = a_star_parent_indices[curr_idx]

        # Update states
        pushfirst!(states, (get_mapf_state_from_idx(env, curr_idx), a_star_dists[curr_idx]))

        # Update actions
        pushfirst!(actions, (get_mapf_action(env, prev_idx, curr_idx),
                        a_star_dists[curr_idx] - a_star_dists[prev_idx]))

        curr_idx = prev_idx
    end

    pushfirst!(states, (get_mapf_state_from_idx(env, start_idx), 0))

    return PlanResult(states=states, actions=actions, cost=cost, fmin=fmin)
end