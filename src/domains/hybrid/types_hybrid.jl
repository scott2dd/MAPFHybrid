#was grid 2D.. we are now switching to our EucInt type to run on our problems...
using Parameters
import  LinearAlgebra.norm
const HybridLocation = NamedTuple{(:nodeIdx,),Tuple{Int64}}

@with_kw struct HybridState <: MAPFState
    time::Int64
    nodeIdx::Int64
    b::Int64
    g::Int64

end
Base.isequal(s1::HybridState, s2::HybridState) = (s1.time, s1.nodeIdx) == (s2.time, s2.nodeIdx)

function equal_except_time(s1::HybridState, s2::HybridState)
    return (s1.nodeIdx == s2.nodeIdx) 
end


@with_kw mutable struct HybridEnvironment <: MAPFEnvironment
    nNodes::Int64
    obstacles::Vector{HybridLocation}
    locs::Matrix{Float64}
    goals::Vector{Int64} #vector of node Idxs
    orig_graph::SimpleWeightedDiGraph{Int64,Float64}    = SimpleWeightedDiGraph()
    state_graph::SimpleWeightedDiGraph{Int64, Float64}    = SimpleWeightedDiGraph()
    state_to_idx::Dict{HybridState,Int64} = Dict{HybridState,Int64}()
    idx_to_state::Dict{Int64,HybridState} = Dict{Int64,HybridState}() #for reverse lookup
    last_goal_constraint::Int64             = -1
    agent_idx::Int64                        = 0
    curr_goal_idx::Int64                    = 0
    euc_inst::EucGraphInt = EucGraphInt(0, 0, [[]], sparse(zeros(Int64, 1,1)), sparse(zeros(Int64, 1,1)), sparse(zeros(Int64, 1,1)), sparse(zeros(Int64, 1,1)), 0, 0, 0,0, sparse(zeros(Int64, 1)), [1. 2. ;3. 4.])  #euc graph.  Just pass whole thing, by reference this is fine.  
end

@enum Action Move=1 Wait=2 
### NEED TO: How to adapt to generic graphs?

struct HybridAction <: MAPFAction
    action::Action
    target::Int64 #target,if wait then set to 0, no 0 node!
end

@enum ConflictType Vertex=1 Edge=2

@with_kw struct HybridConflict <: MAPFConflict
    time::Int64
    agent_1::Int64
    agent_2::Int64
    type::ConflictType
    nodeIdx1::Int64
    nodeIdx2::Int64
end


@with_kw struct VertexConstraint
    time::Int64
    nodeIdx::Int64
end

Base.isless(vc1::VertexConstraint, vc2::VertexConstraint) = (vc1.time, vc1.nodeIdx) < (vc2.time, vc2.nodeIdx)
Base.isequal(vc1::VertexConstraint, vc2::VertexConstraint) = (vc1.time, vc1.nodeIdx) == (vc2.time, vc2.nodeIdx)

@with_kw struct EdgeConstraint
    time::Int64
    nodeIdx1::Int64 
    nodeIdx2::Int64
end

Base.isless(ec1::EdgeConstraint, ec2::EdgeConstraint) = (ec1.time, ec1.nodeIdx1, ec1.nodeIdx2) < (ec2.time, ec2.nodeIdx1, ec2.nodeIdx2)
Base.isequal(ec1::EdgeConstraint, ec2::EdgeConstraint) = (ec1.time, ec1.nodeIdx1, ec1.nodeIdx2) == (ec2.time, ec2.nodeIdx1,ec2.nodeIdx2)

@with_kw struct HybridConstraints <: MAPFConstraints
    vertex_constraints::Set{VertexConstraint}   = Set{VertexConstraint}()
    edge_constraints::Set{EdgeConstraint}       = Set{EdgeConstraint}()
end

get_empty_constraint(::Type{HybridConstraints}) = HybridConstraints()

Base.isempty(constraints::HybridConstraints) = (isempty(constraints.vertex_constraints) && isempty(constraints.edge_constraints))


function reconstruct_path(parents::Dict{Int64,Int64}, curr_idx::Int64, stateidx_to_nodeidx::Dict{Int64,Int64}, idx_to_state::Dict{Int64,HybridState}, graph::SimpleWeightedDiGraph{Int64, Float64}, totalcost::Int64)
    #parents is stateidx to stateidx
    running_cost = totalcost + 0
    curr_node_idx = stateidx_to_nodeidx[curr_idx]
    # path = [curr_node_idx]
    stateseq = Tuple{HybridState, Int64}[]
    pushfirst!(stateseq, (idx_to_state[curr_idx], running_cost))
    
    actions = Tuple{HybridAction, Int64}[]
    next_node = stateidx_to_nodeidx[parents[curr_idx]]
    wij = get_weight(graph, next_node, curr_node_idx)
    pushfirst!(actions, (HybridAction(Move, curr_idx), wij))
    running_cost -= wij
    while haskey(parents, curr_idx)
        curr_idx = parents[curr_idx]
        curr_node_idx = stateidx_to_nodeidx[curr_idx]
        state = idx_to_state[curr_idx]
        pushfirst!(stateseq, (state, running_cost))

        if haskey(parents, curr_idx)
            next_node = stateidx_to_nodeidx[parents[curr_idx]]
            wij = get_weight(graph, next_node, curr_node_idx)
            running_cost -= wij
            if wij == 0 #if it was a wait action
                pushfirst!(actions, (HybridAction(Wait, 0), wij))
            else
                pushfirst!(actions, (HybridAction(Move, curr_idx), wij))
            end
            running_cost -= wij
        end
    end

    
    return stateseq, actions
end
function a_star_implicit_shortest_path!(graph::SimpleWeightedDiGraph{Int64}, env::HybridEnvironment, state::HybridState, uavi::Int64, constraints::HybridConstraints)
    goal = env.goals[uavi]
    state_to_idx = deepcopy(env.state_to_idx)
    idx_to_state = deepcopy(env.idx_to_state)
    parents = Dict{Int64,Int64}() #state_idx to state_idx
    stateidx_to_nodeidx = Dict{Int64,Int64}() #state_idx to node_idx
    gscores = Dict{Int64,Float64}() #keep a list of gscores
    fmin = Inf
    if haskey(state_to_idx, state)
        curr_idx = state_to_idx[state]
        
    else
        state_to_idx[state] = length(state_to_idx) + 1
        curr_idx = state_to_idx[state]
        stateidx_to_nodeidx[curr_idx] = state.nodeIdx
        idx_to_state[curr_idx] = state
    end
    node_idx = state.nodeIdx

    heur(i) = norm(env.locs[i,:] - env.locs[goal,:]) # must take node_idx, for orig graph!!!
     
    openlist = MutableBinaryMinHeap([(0.0 + heur(node_idx), curr_idx, state, 0.0)])
    closedlist = Set{Int64}()


    while !isempty(openlist) #pick and expand, add to open list...
        curr = pop!(openlist) #curr gives us state_graph idx
        curr_idx, state, gval = curr[2], curr[3], curr[4]
        node_idx = state.nodeIdx
        f = curr[1]
        fmin = min(f, fmin)
        # println(node_idx)
        if node_idx == goal
            cost = gscores[curr_idx]
            state_seq, actions = reconstruct_path(parents, curr_idx, stateidx_to_nodeidx, idx_to_state, graph,Int(cost))
            plan = PlanResult(states=state_seq, actions=actions, cost=Int(floor(cost)), fmin=Int(floor(fmin)))
            return plan
        end
        
        #otherwise, expand this node and add each to open list
        for raw_nbr in neighbors(graph, node_idx) #; node_idx]
            bold, gold = state.b, state.g
            bij, gij = euc_inst.C[node_idx, raw_nbr], euc_inst.Z[node_idx, raw_nbr]
            newstate = HybridState(time=state.time+1, nodeIdx=raw_nbr, b = bold+ bij, g = gold + gij)
            nbr_node_idx = newstate.nodeIdx
            # println("neighbor: ", raw_nbr)
            #check if we are on a vertex constraint
            if VertexConstraint(time=newstate.time, nodeIdx=newstate.nodeIdx) in constraints.vertex_constraints
                continue
            elseif EdgeConstraint(time=newstate.time-1, nodeIdx1=node_idx, nodeIdx2=nbr_node_idx) in constraints.edge_constraints || EdgeConstraint(time=newstate.time-1, nodeIdx1=nbr_node_idx, nodeIdx2= node_idx) in constraints.edge_constraints #if we are  on a edge constraint...
                continue
            end

            if haskey(state_to_idx, newstate)
                nbr_idx = state_to_idx[newstate]
            else
                state_to_idx[newstate] = length(state_to_idx) + 1
                nbr_idx = state_to_idx[newstate]
                stateidx_to_nodeidx[nbr_idx] = nbr_node_idx
                idx_to_state[nbr_idx] = newstate
            end
            
            wij = get_weight(graph, node_idx, nbr_node_idx)
            if wij == 0
                wij = 0.0001
            end
            nbr_gval = gval + wij
            fval = nbr_gval + heur(nbr_node_idx)

            if haskey(gscores, nbr_idx) && nbr_gval >= gscores[nbr_idx]
                    continue #don't add if we already have seen a better/equal path to here
            end
            #otherwise, add to the list
            push!(openlist, (fval, nbr_idx, newstate, nbr_gval))
            parents[nbr_idx] = curr_idx
            gscores[nbr_idx] = nbr_gval
        end

        #now also do neighbor!
    end
    return nothing  #if open list empty, then no path found...
end

