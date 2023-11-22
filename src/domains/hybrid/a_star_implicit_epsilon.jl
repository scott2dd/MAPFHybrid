struct AStarEpsilonHEntry{D <: Number}
    a_star_entry::AStarHEntry{D}
    focal_heuristic::D
end

struct CompareFocalHeap
end
compare(comp::CompareFocalHeap, e1::AStarEpsilonHEntry, e2::AStarEpsilonHEntry) = (e1.focal_heuristic, e1.a_star_entry.fvalue, -e1.a_star_entry.gvalue) < (e2.focal_heuristic, e2.a_star_entry.fvalue, -e2.a_star_entry.gvalue)



@with_kw mutable struct AStarEpsilonStates{D<:Number}
    a_star_states::AStarStates{D}                                           = AStarStates{D}()
    # Subset of heap
    focal_heap::MutableBinaryHeap{AStarEpsilonHEntry{D},CompareFocalHeap}   = MutableBinaryHeap{AStarEpsilonHEntry{D},CompareFocalHeap}()
    focal_hmap::Dict{Int64,Int64}                                           = Dict{Int64,Int64}()
    best_fvalue::D                                                          = zero(D)
    focal_heur_value::Dict{Int64,D}                                         = Dict{Int64,D}()
end


"""
Execute expand operation on the chosen node, while implicitly generating its neighbors based on a
visitor method, and populating `neighbors` with the neighboring vertices.
"""
function process_neighbors_implicit!(
    state::AStarEpsilonStates{D},
    graph::AbstractGraph{V},
    edge_wt_fn::Function,
    neighbors::Vector{Int64},
    parent_entry::AStarEpsilonHEntry{D},
    visitor::AbstractDijkstraVisitor,
    eps_weight::Float64,
    admissible_heuristic::Function,
    focal_state_heuristic::Function,
    focal_transition_heuristic::Function) where {V, D <: Number}

    dv = zero(D)
    u = parent_entry.a_star_entry.v_idx
    du = parent_entry.a_star_entry.gvalue

    for iv in neighbors

        # Default color 0
        v_color = get(state.a_star_states.colormap, iv, 0)

        if v_color == 0

            # Inserting for the first time
            state.a_star_states.dists[iv] = dv = du + edge_wt_fn(graph.vertices[u], graph.vertices[iv])
            state.a_star_states.parent_indices[iv] = u
            state.a_star_states.colormap[iv] = 1
            Graphs.discover_vertex!(visitor, graph.vertices[u], graph.vertices[iv], dv)

            new_fvalue = dv + admissible_heuristic(graph.vertices[iv])
            new_focal_heur = parent_entry.focal_heuristic + focal_state_heuristic(graph.vertices[iv]) +
                                                            focal_transition_heuristic(graph.vertices[u], graph.vertices[iv])

            new_entry = AStarEpsilonHEntry(AStarHEntry(iv, dv, new_fvalue), new_focal_heur)

            state.a_star_states.hmap[iv] = push!(state.a_star_states.heap, new_entry.a_star_entry)
            state.focal_heur_value[iv] = new_focal_heur

            # Only insert in focal list if condition satisfied
            if new_fvalue <= eps_weight * state.best_fvalue
                state.focal_hmap[iv] = push!(state.focal_heap, new_entry)
            end

        elseif v_color == 1

            dv = du + edge_wt_fn(graph.vertices[u], graph.vertices[iv])

            # Update cost-to-come if cheaper from current parent
            if dv < state.a_star_states.dists[iv]

                state.a_star_states.dists[iv] = dv
                state.a_star_states.parent_indices[iv] = u

                Graphs.update_vertex!(visitor, graph.vertices[u], graph.vertices[iv], dv)

                updated_fvalue = dv + admissible_heuristic(graph.vertices[iv])
                updated_entry = AStarEpsilonHEntry(AStarHEntry(iv, dv, updated_fvalue), parent_entry.focal_heuristic)

                old_fvalue = state.a_star_states.heap.nodes[state.a_star_states.heap.node_map[state.a_star_states.hmap[iv]]].value.a_star_entry.fvalue
                # @show old_fvalue

                # @show updated_entry
                update!(state.a_star_states.heap, state.a_star_states.hmap[iv], updated_entry.a_star_entry)

                # Enter into focal list if new fvalue is good enough
                # but it was not before
                if updated_fvalue <= eps_weight * state.best_fvalue &&
                    old_fvalue > eps_weight * state.best_fvalue
                    state.focal_hmap[iv] = push!(state.focal_heap, updated_entry)
                end

            end
        end
    end
end


"""
Runs A*-epsilon or focal search algorithm on the given list-of-indices graph.
As with all A* versions (except a_star_spath), the graph is implicit and the graph visitor must generate the neighbors of the expanded
vertex just-in-time.

Arguments:
    - `graph::AbstractGraph{V} where {V}`
    - `edge_wt_fn::F1 where {F1 <: Function}` Maps (u,v) to the edge weight of the u -> v edge
    - `source::Int64`
    - `visitor::AbstractDijkstraVisitor` The graph visitor that implements include_vertex!, which is called when
                                         a vertex is expanded
    - `eps_weight::Float64` w sub-optimality factor for focal search
    - `admissible_heuristic::F2 where {F2 <: Function}` Maps u to h(u); an admissible heuristic for the cost-to-go
    - `focal_state_heuristic::F3 where {F3 <: Function}`
    - `focal_transition_heuristic::F4 where {F4 <: Function}`
    - `::Type{D} = Float64` The type of the edge weight value

Returns:
    - `::AStarEpsilonStates` The result of the search
"""
function a_star_implicit_epsilon_path!(
    graph::AbstractGraph{V},                # the graph
    edge_wt_fn::F1, # distances associated with edges
    source::Int64,             # the source index
    visitor::AbstractDijkstraVisitor,# visitor object\
    eps_weight::Float64,
    admissible_heuristic::F2,      # Heuristic function for vertices
    focal_state_heuristic::F3,
    focal_transition_heuristic::F4,
    ::Type{D} = Float64) where {V, D <: Number, F1 <: Function, F2 <: Function, F3 <: Function, F4 <: Function}

    state = AStarEpsilonStates{D}()
    set_source!(state.a_star_states, source)
    source_heur = admissible_heuristic(graph.vertices[source])
    state.best_fvalue = source_heur

    # Initialize heap
    source_entry = AStarEpsilonHEntry(AStarHEntry(source, zero(D), source_heur), focal_state_heuristic(graph.vertices[source]))
    source_handle = push!(state.a_star_states.heap, source_entry.a_star_entry)
    state.a_star_states.hmap[source] = source_handle

    # Initialize focal heap
    focal_handle = push!(state.focal_heap, source_entry)
    state.focal_hmap[focal_handle] = source_handle

    while ~(isempty(state.a_star_states.heap))

        # Enter open-set items that are now valid into focal list
        old_best_fvalue = state.best_fvalue
        state.best_fvalue = top(state.a_star_states.heap).fvalue

        if state.best_fvalue > old_best_fvalue

            # Iterate over open set  in increasing order of fvalue and insert in focal list if valid
            for node in sort(state.a_star_states.heap.nodes, by = x->x.value.fvalue)
                fvalue = node.value.fvalue

                if fvalue > eps_weight * old_best_fvalue && fvalue <= eps_weight * state.best_fvalue
                    new_focal_entry = AStarEpsilonHEntry(node.value, state.focal_heur_value[node.value.v_idx])
                    state.focal_hmap[node.value.v_idx] = push!(state.focal_heap, new_focal_entry)
                end

                if fvalue > eps_weight * state.best_fvalue
                    break
                end
            end
        end

        # pick next vertex to include
        focal_entry, focal_handle = top_with_handle(state.focal_heap)
        heap_handle = state.a_star_states.hmap[focal_entry.a_star_entry.v_idx]

        ui = focal_entry.a_star_entry.v_idx
        du = focal_entry.a_star_entry.gvalue
        state.a_star_states.colormap[ui] = 2

        # Will be populated by include_vertex!
        nbrs = Vector{Int64}(undef, 0)

        if Graphs.include_vertex!(visitor, graph.vertices[state.a_star_states.parent_indices[ui]], graph.vertices[ui], du, nbrs) == false
            return state
        end
        # @show [graph.vertices[n] for n in nbrs]

        # Delete from open list before considering neighbors
        # TODO: Check this!!!
        pop!(state.focal_heap)
        # delete!(state.focal_heap, focal_handle)
        delete!(state.focal_hmap, focal_entry.a_star_entry.v_idx)
        delete!(state.a_star_states.heap, heap_handle)
        delete!(state.a_star_states.hmap, focal_entry.a_star_entry.v_idx)

        # process u's neighbors
        process_neighbors_implicit!(state, graph, edge_wt_fn, nbrs, focal_entry, visitor,
                                    eps_weight, admissible_heuristic, focal_state_heuristic, focal_transition_heuristic)
        Graphs.close_vertex!(visitor, graph.vertices[ui])
    end

    return state
end