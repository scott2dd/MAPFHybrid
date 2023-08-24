import JSON
using HybridUAVPlanning
using Graphs
using MAPFHybrid

function CBS_hybrid_wrapper(euc_inst::EucGraphInt, starts::Vector{Int64}, goals::Vector{Int64}, hlcost::String = "SOC")

    initial_states = Vector{HybridState}(undef, 0)
    
    graph = make_graph(euc_inst)
    
    env = HybridEnvironment(nNodes = nv(graph), goals=goals, obstacles=HybridLocation[], state_graph = graph)

    ## TODO: Definitely a better way to do this....
    hlcost = "SOC"
    if hlcost == "SOC"
        solver = CBSSolver{HybridState,HybridAction,Int64,SumOfCosts,HybridConflict,HybridConstraints,HybridEnvironment}(env=env)
    elseif hlcost == "MS"
        solver = CBSSolver{HybridState,HybridAction,Int64,Makespan,HybridConflict,HybridConstraints,HybridEnvironment}(env=env)
    else
        println("Enter either SOC or MS for high level cost")
    end

    @time solution = search!(solver, initial_states)

    if isempty(solution)
        println("Solution Empty from CBS")
    else
        cost = 0
        makespan = 0
        for s in solution
            cost += s.cost
            makespan = max(s.cost, makespan)
        end

        printstyled("CBS Statistics:\n", bold=true, color=:lightblue)
        println("Cost: ", cost)
        println("Makespan: ", makespan)

        stats_dict = Dict("cost"=>cost, "makespan"=>makespan)

        schedule_dict = Dict()

        for (a, soln) in enumerate(solution)
            soln_dict = Vector{Dict}(undef,0)
            for (state, t) in soln.states
                push!(soln_dict, Dict("x"=>state.x, "y"=>state.y, "t"=>t))
            end

            schedule_dict[string("agent",a-1)] = soln_dict
        end
    end

    result_dict = Dict("statistics"=>stats_dict, "schedule"=>schedule_dict)

    open(outfile, "w") do f
        JSON.print(f, result_dict, 2)
    end
end

using JLD2
using HybridUAVPlanning
using Random
@load "Problems/euc_probs_disc/50_4conn_1"
nNodes = size(euc_inst.C,1)
starts, goals = randperm(nNodes)[1:3], randperm(nNodes)[1:3]

CBS_hybrid_wrapper(euc_inst, starts, goals)