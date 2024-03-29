using Flux
using TriMeshGame
using MeshPlotter
using ProximalPolicyOptimization
using Distributions: Categorical
using BSON
using Printf

TM = TriMeshGame
PPO = ProximalPolicyOptimization
MP = MeshPlotter

const HALF_EDGES_PER_ELEMENT = 3
const ACTIONS_PER_EDGE = 2
const NO_ACTION_REWARD = 0


#####################################################################################################################
# GENERATING AND MANIPULATING ENVIRONMENT STATE
struct StateData
    vertex_score::Any
    action_mask
end

function Base.show(io::IO, s::StateData)
    na = size(s.vertex_score, 2)
    println(io, "StateData")
    println(io, "\tNum Actions : $na")
end

function pad_vertex_scores(vertex_scores_vector)
    num_half_edges = [size(vs, 2) for vs in vertex_scores_vector]
    max_num_half_edges = maximum(num_half_edges)
    num_new_cols = max_num_half_edges .- num_half_edges
    padded_vertex_scores = [TM.zero_pad(vs, nc) for (vs, nc) in zip(vertex_scores_vector, num_new_cols)]
    return padded_vertex_scores
end

function pad_action_mask(action_mask_vector)
    num_actions = length.(action_mask_vector)
    max_num_actions = maximum(num_actions)
    num_new_actions = max_num_actions .- num_actions
    padded_action_mask = [TM.pad(am, nr, -Inf32) for (am, nr) in zip(action_mask_vector, num_new_actions)]
    return padded_action_mask
end

function PPO.prepare_state_data_for_batching!(state_data_vector)
    vertex_score = [s.vertex_score for s in state_data_vector]
    action_mask = [s.action_mask for s in state_data_vector]

    padded_vertex_scores = pad_vertex_scores(vertex_score)
    padded_action_mask = pad_action_mask(action_mask)

    state_data_vector .= [StateData(vs, am) for (vs, am) in zip(padded_vertex_scores, padded_action_mask)]
end

function PPO.batch_state(state_data_vector)
    vs = [s.vertex_score for s in state_data_vector]
    am = [s.action_mask for s in state_data_vector]

    batch_vertex_score = cat(vs..., dims=3)
    batch_action_mask = cat(am..., dims=2)
    return StateData(batch_vertex_score, batch_action_mask)
end

function val_or_missing(vector, template, missing_val)
    return [t == 0 ? missing_val : vector[t] for t in template]
end

function get_action_mask(active_triangle)
    actions_per_triangle = HALF_EDGES_PER_ELEMENT * ACTIONS_PER_EDGE

    requires_mask = repeat(.!active_triangle', inner=(actions_per_triangle, 1))
    mask = vec([r ? -Inf32 : 0.0f0 for r in requires_mask])
    return mask
end

function PPO.state(wrapper)
    env = wrapper.env
    template = TM.make_level4_template(env.mesh)

    vs = val_or_missing(env.vertex_score, template, 0)
    vd = val_or_missing(env.mesh.degrees, template, 0)
    vdist = val_or_missing(wrapper.distance_weights, template, 0)
    vdist = vdist .- vdist[1,:]'

    matrix = vcat(vs, vd, vdist)

    am = get_action_mask(env.mesh.active_triangle)

    s = StateData(matrix, am)

    return s
end
#####################################################################################################################


#####################################################################################################################
# CHECKING VALID STATE
function all_active_vertices(mesh)
    for vertex in mesh.connectivity
        if !((vertex == 0) || (TM.is_active_vertex(mesh, vertex)))
            return false
        end
    end
    return true
end

function all_active_triangle_or_boundary(mesh)
    for triangle in mesh.t2t
        if !(TM.is_active_triangle_or_boundary(mesh, triangle))
            return false
        end
    end
    return true
end

function no_triangle_self_reference(mesh)
    for triangle in 1:TM.triangle_buffer(mesh)
        if TM.is_active_triangle(mesh, triangle)
            nbrs = mesh.t2t[:, triangle]
            if any(triangle .== nbrs)
                return false
            end
        end
    end
    return true
end

function has_initial_optimal_score(wrapper)
    return wrapper.opt_score == optimum_score(wrapper.env.vertex_score)
end

function check_valid_state(wrapper)
    mesh = wrapper.env.mesh
    flag = all_active_vertices(mesh) && 
           no_triangle_self_reference(mesh) && 
           all_active_triangle_or_boundary(mesh) &&
           all_unique_neighbors(mesh) &&
           has_initial_optimal_score(wrapper)

    return flag
end

function has_unique_neighbors(neighbors)
    @assert length(neighbors) == 3
    a, b, c = neighbors

    flag = (a != b || a == 0) &&
           (a != c || a == 0) &&
           (b != c || b == 0)
    return flag
end

function all_unique_neighbors(mesh)
    for triangle in 1:TM.triangle_buffer(mesh)
        if TM.is_active_triangle(mesh, triangle)
            neighbors = mesh.t2t[:, triangle]
            if !has_unique_neighbors(neighbors)
                return false
            end
        end
    end
    return true
end
#####################################################################################################################





#####################################################################################################################
# EVALUATING POLICY
function PPO.action_probabilities(policy, state)
    vertex_score, action_mask = state.vertex_score, state.action_mask

    logits = vec(policy(vertex_score)) + action_mask
    p = softmax(logits)
    return p
end

function PPO.batch_action_probabilities(policy, state)
    vertex_score, action_mask = state.vertex_score, state.action_mask
    nf, nq, nb = size(vertex_score)
    logits = reshape(policy(vertex_score), :, nb) + action_mask
    probs = softmax(logits, dims=1)
    return probs
end
#####################################################################################################################





#####################################################################################################################
# STEPPING THE ENVIRONMENT
function PPO.reward(wrapper)
    return wrapper.reward
end

function PPO.is_terminal(wrapper)
    return wrapper.is_terminated
end

function index_to_action(index)
    actions_per_triangle = HALF_EDGES_PER_ELEMENT * ACTIONS_PER_EDGE

    triangle_index = div(index - 1, actions_per_triangle) + 1

    triangle_action_idx = rem(index - 1, actions_per_triangle)
    half_edge_index = div(triangle_action_idx, ACTIONS_PER_EDGE) + 1
    action_type = rem(triangle_action_idx, ACTIONS_PER_EDGE) + 1

    return triangle_index, half_edge_index, action_type
end


function step_wrapper!(wrapper, triangle_index, half_edge_index, action_type)
    env = wrapper.env
    previous_score = wrapper.current_score
    success = false

    @assert TM.is_active_triangle(env.mesh, triangle_index) "Attempting to act on inactive triangle $triangle_index with action ($triangle_index, $half_edge_index, $action_type)"
    @assert action_type in 1:ACTIONS_PER_EDGE "Expected action type in 1:$ACTIONS_PER_EDGE, got type = $action_type"
    @assert half_edge_index in 1:HALF_EDGES_PER_ELEMENT "Expected edge in 1:$HALF_EDGES_PER_ELEMENT, got edge = $half_edge_index"
    @assert check_valid_state(wrapper) "Invalid state encountered, check the environment"

    if action_type == 1
        success = TM.step_flip!(env, triangle_index, half_edge_index)
    elseif action_type == 2
        success = TM.step_split!(env, triangle_index, half_edge_index)
    # elseif action_type == 3
    #     success = TM.step_collapse!(env, triangle_index, half_edge_index)
    else
        error("Unexpected action type $action_type")
    end

    if success
        wrapper.distance_weights = compute_distance_weights(env.mesh)
        wrapper.current_score = global_score(env.vertex_score, wrapper.distance_weights)
        wrapper.reward = previous_score - wrapper.current_score
    else
        wrapper.reward = NO_ACTION_REWARD
    end
    
    wrapper.num_actions += 1
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score, 
        wrapper.num_actions, wrapper.max_actions)

end

function action_space_size(env)
    nt = TM.triangle_buffer(env.mesh)
    return nt * HALF_EDGES_PER_ELEMENT * ACTIONS_PER_EDGE
end

function PPO.step!(wrapper, linear_action_index)
    @assert 1 <= linear_action_index <= action_space_size(wrapper.env)
    triangle_index, half_edge_index, action_type = index_to_action(linear_action_index)

    step_wrapper!(wrapper, triangle_index, half_edge_index, action_type)
end
#####################################################################################################################





#####################################################################################################################
# PLOTTING STUFF
function smooth_wrapper!(wrapper, niter = 1)
    for iteration in 1:niter
        TM.averagesmoothing!(wrapper.env.mesh)
    end
end

function plot_env_score!(ax, score, opt_score; coords = (0.8, 0.8), fontsize = 50)
    tpars = Dict(
        :color => "black",
        :horizontalalignment => "center",
        :verticalalignment => "center",
        :fontsize => fontsize,
        :fontweight => "bold",
    )

    text = string(score) * "/" * string(opt_score)
    ax.text(coords[1], coords[2], text; tpars...)
end

function plot_env(_env, current_score, opt_score, number_elements, internal_order)
    env = deepcopy(_env)


    TM.reindex!(env)
    mesh = env.mesh
    fig, ax = MP.plot_mesh(TM.active_vertex_coordinates(mesh), TM.active_triangle_connectivity(mesh),
        vertex_score=TM.active_vertex_score(env), vertex_size=20, fontsize=15,
        number_elements = number_elements, internal_order = internal_order)

    plot_env_score!(ax, current_score, opt_score)

    return fig
end

function plot_wrapper(wrapper, filename = "", smooth_iterations = 5; number_elements = false)
    smooth_wrapper!(wrapper, smooth_iterations)
    
    internal_order = number_elements
    element_numbers = number_elements ? findall(wrapper.env.mesh.active_triangle) : false
    fig = plot_env(wrapper.env, wrapper.current_score, wrapper.opt_score, element_numbers, internal_order)

    if length(filename) > 0
        fig.tight_layout()
        fig.savefig(filename)
    end

    return fig
end

function plot_trajectory(policy, wrapper, root_directory)
    if !isdir(root_directory)
        mkpath(root_directory)
    end

    fig_name = "figure-" * lpad(0, 3, "0") * ".png"
    filename = joinpath(root_directory, fig_name)
    plot_wrapper(wrapper, filename)

    fig_index = 1
    done = PPO.is_terminal(wrapper)
    while !done 
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        
        fig_name = "figure-" * lpad(fig_index, 3, "0") * ".png"
        filename = joinpath(root_directory, fig_name)
        plot_wrapper(wrapper, filename)
        fig_index += 1

        done = PPO.is_terminal(wrapper)
    end
end
#####################################################################################################################



#####################################################################################################################
# EVALUATING PERFORMANCE

mutable struct SaveBestModel
    file_path
    num_trajectories
    best_return
    mean_returns
    std_returns
    function SaveBestModel(root_dir, num_trajectories, filename = "best_model.bson")
        if !isdir(root_dir)
            mkpath(root_dir)
        end

        file_path = joinpath(root_dir, filename)
        mean_returns = []
        std_returns = []
        new(file_path, num_trajectories, -Inf, mean_returns, std_returns)
    end
end

function save_model(s::SaveBestModel, policy)
    d = Dict("evaluator" => s, "policy" => policy)
    BSON.@save s.file_path d
end

function (s::SaveBestModel)(policy, wrapper)
    ret, dev = average_normalized_returns(policy, wrapper, s.num_trajectories)
    if ret > s.best_return
        s.best_return = ret
        @printf "\nNEW BEST RETURN : %1.4f\n" ret
        println("SAVING MODEL AT : " * s.file_path * "\n\n")
        save_model(s, policy)
    end

    @printf "RET = %1.4f\tDEV = %1.4f\n" ret dev
    push!(s.mean_returns, ret)
    push!(s.std_returns, dev)
end

function single_trajectory_normalized_return(policy, wrapper)
    maxreturn = wrapper.current_score - wrapper.opt_score
    if maxreturn == 0
        return 1.0
    else
        ret = PPO.single_trajectory_return(policy, wrapper)
        return ret / maxreturn
    end
end

function average_normalized_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = single_trajectory_normalized_return(policy, wrapper)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_single_trajectory_return(policy, wrapper)
    env = wrapper.env

    done = PPO.is_terminal(wrapper)

    initial_score = env.current_score
    minscore = initial_score
    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)

        minscore = min(minscore, env.current_score)
        done = PPO.is_terminal(wrapper)
    end
    return initial_score - minscore
end

function average_best_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_single_trajectory_return(policy, wrapper)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_normalized_single_trajectory_return(policy, wrapper)
    max_return = wrapper.env.current_score - wrapper.env.opt_score
    if max_return == 0
        return 1.0
    else
        ret = best_single_trajectory_return(policy, wrapper)
        return ret/max_return
    end
end

function average_normalized_best_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_normalized_single_trajectory_return(policy, wrapper)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_state_in_rollout(wrapper, policy)
    best_wrapper = deepcopy(wrapper)

    minscore = wrapper.env.current_score
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)

        if wrapper.env.current_score < minscore 
            minscore = wrapper.env.current_score
            best_wrapper = deepcopy(wrapper)
        end
    end

    return best_wrapper
end
#####################################################################################################################





#####################################################################################################################
# DEBUGGING : SEARCHING FOR INVALID STATE

function trajectory_to_invalid_state(policy, wrapper)   
    done = PPO.is_terminal(wrapper)
    history = Dict("envs" => [deepcopy(wrapper)], "actions" => [0])

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))
        PPO.step!(wrapper, action)
        
        push!(history["envs"], deepcopy(wrapper))
        push!(history["actions"], action)

        if !check_valid_state(wrapper)
            println("\n\nINVALID STATE ENCOUNTERED\n\n")
            return history
        end

        done = PPO.is_terminal(wrapper)
    end
end

function search_invalid_action(policy, wrapper, num_iter)
    for iteration in 1:num_iter
        println("SEARCH ITERATION : $iteration")
        PPO.reset!(wrapper)
        out = trajectory_to_invalid_state(policy, wrapper)
        if !isnothing(out)
            return out
        end
    end
end

#####################################################################################################################