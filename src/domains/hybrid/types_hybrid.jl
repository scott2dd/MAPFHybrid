#was grid 2D.. we are now switching to our EucInt type to run on our problems...
using Parameters

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
    obstacles::Set{HybridLocation}
    goals::Vector{Int64} #vector of node Idxs
    state_graph::SimpleWeightedDiGraph{Int64}    = SimpleWeightedDiGraph{Int64}()
    state_to_idx::Dict{HybridState,Int64}         = Dict{HybridState,Int64}()
    last_goal_constraint::Int64             = -1
    agent_idx::Int64                        = 0
    curr_goal_idx::Int64                    = 0
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


