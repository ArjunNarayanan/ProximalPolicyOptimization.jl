using Flux
using PlotQuadMesh
using RandomQuadMesh
using QuadMeshGame
using ProximalPolicyOptimization
using Distributions: Categorical

RQ = RandomQuadMesh
QM = QuadMeshGame
PPO = ProximalPolicyOptimization
PQ = PlotQuadMesh

include("policy.jl")

#####################################################################################################################
# GENERATING AND MANIPULATING ENVIRONMENT STATE
struct StateData
    vertex_score::Any
    action_mask
end

function Base.show(io::IO, s::StateData)
    println(io, "StateData")
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

function action_mask(active_quad; actions_per_edge=4)
    actions_per_quad = 4 * actions_per_edge
    requires_mask = repeat(.!active_quad', inner=(actions_per_quad, 1))
    mask = vec([r ? -Inf32 : 0.0f0 for r in requires_mask])
    return mask
end

function PPO.state(wrapper)
    env = wrapper.env
    template = QM.make_level4_template(env.mesh)

    vs = val_or_missing(env.vertex_score, template, 0)
    vd = val_or_missing(env.mesh.degree, template, 0)
    matrix = vcat(vs, vd)

    am = action_mask(env.mesh.active_quad)

    s = StateData(matrix, am)

    return s
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
    env = wrapper.env
    return env.reward
end

function PPO.is_terminal(wrapper)
    env = wrapper.env
    return env.is_terminated
end

function index_to_action(index; actions_per_edge=4)
    actions_per_quad = 4 * actions_per_edge

    quad = div(index - 1, actions_per_quad) + 1

    quad_action_idx = rem(index - 1, actions_per_quad)
    edge = div(quad_action_idx, actions_per_edge) + 1
    action = rem(quad_action_idx, actions_per_edge) + 1

    return quad, edge, action
end

function action_space_size(env; actions_per_edge=4)
    nq = QM.quad_buffer(env.mesh)
    return nq * 4 * actions_per_edge
end

function all_active_vertices(mesh)
    flag = true
    for vertex in mesh.connectivity
        if !((vertex == 0) || (QM.is_active_vertex(mesh, vertex)))
            flag = false
        end
    end
    return flag
end

function no_quad_self_reference(mesh)
    flag = true
    for quad in 1:QM.quad_buffer(mesh)
        if QM.is_active_quad(mesh, quad)
            nbrs = mesh.q2q[:, quad]
            if any(quad .== nbrs)
                flag = false
            end
        end
    end
    return flag
end

function all_active_quad_or_boundary(mesh)
    flag = true
    for quad in mesh.q2q
        if !(QM.is_active_quad_or_boundary(mesh, quad))
            flag = false
        end
    end
    return flag
end

function check_valid_state(wrapper)
    mesh = wrapper.env.mesh
    flag = all_active_vertices(mesh) && no_quad_self_reference(mesh) && all_active_quad_or_boundary(mesh)
    return flag
end

function PPO.step!(wrapper, quad, edge, type, no_action_reward=-4)
    env = wrapper.env

    @assert QM.is_active_quad(env.mesh, quad) "Attempting to act on inactive quad $quad with action ($quad, $edge, $type)"
    @assert type in (1, 2, 3, 4) "Expected action type in {1,2,3,4,5} got type = $type"
    @assert edge in (1, 2, 3, 4) "Expected edge in {1,2,3,4} got edge = $edge"
    # @assert check_valid_state(wrapper) "Invalid state encountered, check the environment"

    if type == 1
        QM.step_left_flip!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 2
        QM.step_right_flip!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 3
        QM.step_split!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 4
        QM.step_collapse!(env, quad, edge, no_action_reward=no_action_reward)
        # elseif type == 5
        #     QM.step_nothing!(env)
    else
        error("Unexpected action type $type")
    end

end

function PPO.step!(wrapper, action_index; no_action_reward=-4)
    env = wrapper.env
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !env.is_terminated "Attempting to step in terminated environment with action $action_index"

    quad, edge, type = index_to_action(action_index)
    PPO.step!(wrapper, quad, edge, type, no_action_reward)
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

    text = string(score)
    ax.text(coords[1], coords[2], score; tpars...)
end

function plot_env(env)
    env = deepcopy(env)

    QM.reindex_game_env!(env)
    mesh = env.mesh
    vs = QM.active_vertex_score(env)

    fig, ax = PQ.plot_mesh(
        QM.active_vertex_coordinates(mesh),
        QM.active_quad_connectivity(mesh),
        vertex_score=vs,
    )
    plot_env_score!(ax, env.current_score)

    return fig
end

function plot_wrapper(wrapper, filename = ""; smooth_iterations = 5)
    smooth_wrapper!(wrapper, smooth_iterations)

    fig = plot_env(wrapper.env)

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
#####################################################################################################################


#####################################################################################################################
# EVALUATING POLICY
function best_single_trajectory_return(wrapper, policy)
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

function average_best_returns(wrapper, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_single_trajectory_return(wrapper, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function single_trajectory_scores(wrapper, policy)
    initial_score = wrapper.env.initial_score
    scores = []
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        push!(scores, initial_score - wrapper.env.current_score)
        done = PPO.is_terminal(wrapper)
    end
    return scores
end

function single_trajectory_return(wrapper, policy)
    env = wrapper.env
    ret = 0
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)
        ret += PPO.reward(wrapper)
    end
    return ret
end

function best_state_in_rollout(wrapper, policy)
    wrappers = [deepcopy(wrapper)]
    scores = [wrapper.env.current_score - wrapper.env.opt_score]
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)

        push!(wrappers, deepcopy(wrapper))
        push!(scores, wrapper.env.current_score - wrapper.env.opt_score)
    end
    idx = argmin(scores)
    return wrappers[idx]
end

function average_returns(wrapper, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = single_trajectory_return(wrapper, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function single_trajectory_normalized_return(wrapper, policy)
    env = wrapper.env
    maxreturn = env.current_score - env.opt_score
    if maxreturn == 0
        return 1.0
    else
        ret = best_single_trajectory_return(wrapper, policy)
        return ret / maxreturn
    end
end

function average_normalized_returns(wrapper, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = single_trajectory_normalized_return(wrapper, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function moving_average(vector, window_size=5)
    half_window = div(window_size, 2)
    smoothed = similar(vector)
    N = length(vector)
    for (idx, val) in enumerate(vector)
        start = max(1, idx - half_window)
        start = min(N - window_size + 1, start)
        stop = start + window_size - 1
        smoothed[idx] = sum(vector[start:stop]) / window_size
    end
    return smoothed
end
#####################################################################################################################