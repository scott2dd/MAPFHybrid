module MAPFHybrid

# stdlib
using Random
using DataStructures

# others
using Parameters
using Graphs
using SimpleWeightedGraphs
using HybridUAVPlanning
using SparseArrays
import Statistics.mean


# Abstract types
""" MAPFState

Abstract base type for tracking the state of an agent in a MAPF problem
"""
abstract type MAPFState end

""" MAPFAction

Abstract base type for the action that an agent can take in a MAPF problem.
"""
abstract type MAPFAction end

""" MAPFConflict

Abstract base type for a path-path conflict in a MAPF problem.
"""
abstract type MAPFConflict end

""" MAPFConstraints

Abstract base type for the constraints on the low-level path search induced by a conflict in a MAPF problem.
"""
abstract type MAPFConstraints end

""" MAPFEnvironment

Abstract base type for the environment in a MAPF problem.
"""
abstract type MAPFEnvironment end


include("utils.jl")
include("high_level_cost.jl")
include("cbs.jl")
include("ecbs.jl")
# include("domains/grid2d/types.jl")
# include("domains/grid2d/cbs_grid2d.jl")
# include("domains/grid2d/ecbs_grid2d.jl")
# include("domains/grid2d/a_star_epsilon_grid2d.jl")
include("domains/hybrid/types_hybrid.jl")
include("domains/hybrid/label_temporal.jl")
include("domains/hybrid/label_temporal_focal.jl")
include("domains/hybrid/cbs_hybrid.jl")
include("domains/hybrid/ecbs_hybrid.jl")
# Types
export
    MAPFState,
    MAPFAction,
    MAPFConflict,
    MAPFConstraints,
    MAPFEnvironment

# Utils
export
    PlanResult,
    get_mapf_state_from_idx,
    get_mapf_action,
    get_plan_result_from_astar

# High Level Cost elements
export
    HighLevelCost,
    compute_cost,
    accumulate_cost,
    deaccumulate_cost,
    SumOfCosts,
    Makespan

# CBS elements
export
    get_empty_constraint,
    set_low_level_context!,
    get_first_conflict,
    create_constraints_from_conflict,
    overlap_between_constraints,
    add_constraint,
    low_level_search!,
    CBSSolver,
    ECBSSolver,
    search!,
    focal_heuristic,
    a_star_implicit_shortest_path!

# Hybrid Types
export
    HybridState,
    HybridAction,
    HybridConflict,
    HybridConstraints,
    HybridEnvironment,
    HybridLocation

# Grid 2D Types
# export
    # Grid2DState,
    # Grid2DAction,
    # Grid2DConflict,
    # Grid2DConstraints,
    # Grid2DEnvironment,
    # Grid2DLocation,
    # AStarGrid2DEnvironment



end # module
