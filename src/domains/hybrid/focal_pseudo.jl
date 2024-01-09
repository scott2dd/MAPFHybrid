#now define focal search.  Let's do pseudo-pseudocode
function focal_label_search(graph, start, goal ; eps = 1.3)
    open_list = MutableBinaryMinHeap{MyLabel}()
    focal_list = MutableBinaryMinHeap{MyLabel, myLabelFocalCompare}()
    
    closed_list = Dict{Int64,MyLabel}()

    id_track = 1 #count labels (ids for labels)
    z = 0 #init z 

    hS = h(start)
    
    init_label = MyLabel(batt_state=0, gen_state=0, gcost = 0, fcost=0 + hS, hcost=hS, focal_heuristic=0, state_idx=1, prior_state_idx=0, _hold_came_from_prior=0, came_from_idx=0, pathlength=0, _hold_gen_track_prior=0, gentrack_idx=0, gen_bool=0, label_id=id_track)

    id_track += 1

    open_map, focal_map = Dict{Int64,Int64}(), Dict{Int64,Int64}()
    open_map[start] = push!(open_list, init_label)
    focal_map[start] = push!(focal_list, init_label)

    while !isempty(focal_list)
        fmin = top(open_list).fcost
        n = pop!(focal_list) #pull and delete
        delete!(open_list, open_map[n.label_id]) #remove from open_list

        if n.state_idx == goal
            return n
        end

        for j in succ(n)
            new_label = new_labels(n, graph)
            if true #eff, no conflicts, etc...
                open_map[new_label.label_id] = push!(open_list, new_label)
                if new_labels.fcost < fmin*eps #then add to focal
                    focal_map[new_label.label_id] = push!(focal_list, new_label)
                end
            end
        if !isempty(open_list) && fmin*eps < top(open_list).fcost
            update_focal!(eps*fmin, eps*top(open_list).fcost, open_list, focal_list)
        end
        z += 1
    end
    return nothing
end


function new_labels(label, graph)
    return MyLabel(gcost=0.0, fcost=0.0, hcost=0.0, focal_heuristic=0.0, state_idx=0, prior_state_idx=0, _hold_came_from_prior=0, came_from_idx=0, pathlength=0, _hold_gen_track_prior=0, gentrack_idx=0, gen_bool=0, batt_state=0.0, gen_state=0.0)
end


