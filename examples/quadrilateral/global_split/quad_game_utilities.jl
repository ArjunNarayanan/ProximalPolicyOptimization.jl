using Flux
using PlotQuadMesh
using RandomQuadMesh
using QuadMeshGame
using ProximalPolicyOptimization
using Distributions: Categorical
using BSON
using Printf

RQ = RandomQuadMesh
QM = QuadMeshGame
PPO = ProximalPolicyOptimization
PQ = PlotQuadMesh

include("../policy.jl")

const NUM_ACTIONS_PER_EDGE = 5
const NO_ACTION_REWARD = 0

#####################################################################################################################
# GENERATING AND MANIPULATING ENVIRONMENT STATE
struct StateData
    vertex_score::Any
    action_mask
end

function Base.show(io::IO, s::StateData)
    println(io, "StateData")
end

function pad_vertex_scores(vertex_scores_vector)
    num_half_edges = [size(vs, 2) for vs in vertex_scores_vector]
    max_num_half_edges = maximum(num_half_edges)
    num_new_cols = max_num_half_edges .- num_half_edges
    padded_vertex_scores = [QM.zero_pad_matrix_cols(vs, nc) for (vs, nc) in zip(vertex_scores_vector, num_new_cols)]
    return padded_vertex_scores
end

function pad_action_mask(action_mask_vector)
    num_actions = length.(action_mask_vector)
    max_num_actions = maximum(num_actions)
    num_new_actions = max_num_actions .- num_actions
    padded_action_mask = [QM.pad_vector(am, nr, -Inf32) for (am, nr) in zip(action_mask_vector, num_new_actions)]
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

function action_mask(template, actions_per_edge)
    requires_mask = mapslices(x -> all(x .== 0), template, dims=1)
    requires_mask = repeat(requires_mask, inner=(actions_per_edge, 1))
    requires_mask = vec(requires_mask)
    mask = vec([r ? -Inf32 : 0.0f0 for r in requires_mask])
    return mask
end

function PPO.state(wrapper)
    env = wrapper.env
    template = QM.make_level4_template(env.mesh)

    vs = val_or_missing(env.vertex_score, template, 0)
    vd = val_or_missing(env.mesh.degree, template, 0)
    
    # vdist = val_or_missing(wrapper.distance_to_boundary, template, 0)
    # vdist = vdist .- vdist[1,:]'
    
    matrix = vcat(vs, vd)
    am = action_mask(template, NUM_ACTIONS_PER_EDGE)

    s = StateData(matrix, am)

    return s
end
#####################################################################################################################


#####################################################################################################################
# EVALUATING POLICY
function PPO.action_probabilities(policy, state)
    @assert policy.num_output_channels == NUM_ACTIONS_PER_EDGE
    
    vertex_score, action_mask = state.vertex_score, state.action_mask
    logits = vec(policy(vertex_score)) + action_mask
    p = softmax(logits)
    return p
end

function PPO.batch_action_probabilities(policy, state)
    @assert policy.num_output_channels == NUM_ACTIONS_PER_EDGE

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
    actions_per_quad = 4 * NUM_ACTIONS_PER_EDGE

    quad = div(index - 1, actions_per_quad) + 1

    quad_action_idx = rem(index - 1, actions_per_quad)
    edge = div(quad_action_idx, NUM_ACTIONS_PER_EDGE) + 1
    action = rem(quad_action_idx, NUM_ACTIONS_PER_EDGE) + 1

    return quad, edge, action
end

function action_space_size(env)
    nq = QM.quad_buffer(env.mesh)
    return nq * 4 * NUM_ACTIONS_PER_EDGE
end

function assert_valid_mesh(mesh)
    @assert QM.all_active_vertices(mesh) "Found inactive vertices in mesh connectivity"
    @assert QM.no_quad_self_reference(mesh) "Found self-referencing quads in mesh q2q"
    @assert QM.all_active_quad_or_boundary(mesh) "Found inactive quads in mesh q2q"
end

function PPO.step!(wrapper, quad, edge, type)
    env = wrapper.env
    previous_score = wrapper.current_score
    success = false

    @assert QM.is_active_quad(env.mesh, quad) "Attempting to act on inactive quad $quad with action ($quad, $edge, $type)"
    @assert type in 1:NUM_ACTIONS_PER_EDGE "Expected action type in {1,2,3,4} got type = $type"
    @assert edge in (1, 2, 3, 4) "Expected edge in {1,2,3,4} got edge = $edge"
    assert_valid_mesh(env.mesh)

    # println("\tStepping : ", quad, edge, type)

    if type == 1
        success = QM.step_left_flip!(env, quad, edge)
    elseif type == 2
        success = QM.step_right_flip!(env, quad, edge)
    elseif type == 3
        success = QM.step_split!(env, quad, edge)
    elseif type == 4
        success = QM.step_collapse!(env, quad, edge)
    elseif type == 5
        success = QM.step_global_split_without_loops!(env, quad, edge, 50)
    else
        error("Unexpected action type $type")
    end

    if success
        wrapper.current_score = global_score(wrapper.env.vertex_score)
        wrapper.reward = previous_score - wrapper.current_score
    else
        wrapper.reward = NO_ACTION_REWARD
    end
    
    wrapper.num_actions += 1
    wrapper.is_terminated = check_terminated(
        wrapper.current_score,
        wrapper.opt_score,
        wrapper.num_actions,
        wrapper.max_actions
    )
end

function PPO.step!(wrapper, action_index)
    env = wrapper.env
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !wrapper.is_terminated "Attempting to step in terminated environment with action $action_index"

    quad, edge, type = index_to_action(action_index)
    PPO.step!(wrapper, quad, edge, type)
end
#####################################################################################################################


#####################################################################################################################
# PLOTTING STUFF
function plot_mesh(mesh)
    mesh = deepcopy(mesh)
    QM.reindex_quads!(mesh)
    QM.reindex_vertices!(mesh)
    fig, ax = PQ.plot_mesh(QM.active_vertex_coordinates(mesh),
        QM.active_quad_connectivity(mesh),
        elem_numbers=elem_numbers,
        internal_order=internal_order,
        node_numbers=node_numbers)
    return fig
end

function plot_env_score!(ax, score; coords = (0.8, 0.8), fontsize = 50)
    tpars = Dict(
        :color => "black",
        :horizontalalignment => "center",
        :verticalalignment => "center",
        :fontsize => fontsize,
        :fontweight => "bold",
    )

    ax.text(coords[1], coords[2], score; tpars...)
end

function plot_env(env, score)
    env = deepcopy(env)

    QM.reindex_game_env!(env)
    mesh = env.mesh
    vs = QM.active_vertex_score(env)

    fig, ax = PQ.plot_mesh(
        QM.active_vertex_coordinates(mesh),
        QM.active_quad_connectivity(mesh),
        vertex_score=vs,
        vertex_size = 30
    )
    
    plot_env_score!(ax, score)

    return fig, ax
end

function plot_wrapper(wrapper, filename = ""; smooth_iterations = 5)
    smooth_wrapper!(wrapper, smooth_iterations)

    text = string(wrapper.current_score) * " / " * string(wrapper.opt_score)
    fig, ax = plot_env(wrapper.env, text)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    if length(filename) > 0
        fig.tight_layout()
        fig.savefig(filename)
    end

    return fig
end

function smooth_wrapper!(wrapper, num_iterations = 1)
    for iteration in 1:num_iterations
        QM.averagesmoothing!(wrapper.env.mesh)
    end
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

function plot_returns(ret, lower_fill, upper_fill)
    fig, ax = subplots()
    ax.plot(ret)
    ax.fill_between(1:length(ret), lower_fill, upper_fill, alpha = 0.2, facecolor = "blue")
    ax.grid()
    ax.set_xlabel("PPO Iterations")
    ax.set_ylabel("Normalized returns")
    return fig, ax
end

function plot_normalized_returns(ret, dev)
    lower = ret - dev
    upper = ret + dev
    upper[upper .> 1.0] .= 1.0
    return plot_returns(ret, lower, upper)
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
    done = PPO.is_terminal(wrapper)

    initial_score = wrapper.current_score
    minscore = wrapper.current_score

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)

        minscore = min(minscore, wrapper.current_score)
        done = PPO.is_terminal(wrapper)
    end
    return initial_score - minscore
end

function average_best_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_single_trajectory_return(wrapper, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_normalized_single_trajectory_return(policy, wrapper)
    max_return = wrapper.current_score - wrapper.opt_score
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
    minscore = wrapper.current_score
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)

        if wrapper.current_score < minscore 
            minscore = wrapper.current_score
            best_wrapper = deepcopy(wrapper)
        end
    end

    return best_wrapper
end
#####################################################################################################################