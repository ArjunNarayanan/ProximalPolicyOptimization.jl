using Flux
using PlotQuadMesh
using QuadMeshGame
using ProximalPolicyOptimization
using Distributions: Categorical

QM = QuadMeshGame
PPO = ProximalPolicyOptimization
PQ = PlotQuadMesh

include("policy.jl")

mutable struct GameEnvWrapper
    mesh0::Any
    desired_degree
    max_actions::Any
    env::Any
    function GameEnvWrapper(mesh0, desired_degree, max_actions)
        mesh = deepcopy(mesh0)
        d0 = deepcopy(desired_degree)
        env = QM.GameEnv(mesh, d0, max_actions)
        new(mesh0, d0, max_actions, env)
    end
end

function Base.show(io::IO, wrapper::GameEnvWrapper)
    println(io, "GameEnvWrapper")
    show(io, wrapper.env)
end

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

function action_mask(active_quad)
    requires_mask = repeat(.!active_quad', inner=(20,1))
    mask = vec([r ? -Inf32 : 0.0f0 for r in requires_mask])
    return mask
end

function PPO.state(wrapper)
    env = wrapper.env

    vs = val_or_missing(env.vertex_score, env.template, 0)
    am = action_mask(env.mesh.active_quad)
    s = StateData(vs, am)

    return s
end

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

function PPO.reward(wrapper)
    env = wrapper.env
    return env.reward
end

function PPO.is_terminal(wrapper)
    env = wrapper.env
    return env.is_terminated
end

function PPO.reset!(wrapper)
    mesh = deepcopy(wrapper.mesh0)
    d0 = deepcopy(wrapper.desired_degree)
    wrapper.env = QM.GameEnv(mesh, d0, wrapper.max_actions)
end

function index_to_action(index)
    quad = div(index-1, 20) + 1

    quad_action_idx = rem(index-1,20)
    edge = div(quad_action_idx, 5) + 1
    action = rem(quad_action_idx, 5) + 1

    return quad, edge, action
end

function action_space_size(env; actions_per_edge = 5)
    nq = QM.quad_buffer(env.mesh)
    return nq*4*actions_per_edge
end

function PPO.step!(wrapper, quad, edge, type, no_action_reward = -4)
    env = wrapper.env

    @assert QM.is_active_quad(env.mesh, quad) "Attempting to act on inactive quad $quad with action $action_index"
    @assert type in (1,2,3,4,5) "Expected action type in {1,2,3,4,5} got type = $type"
    @assert edge in (1,2,3,4) "Expected edge in {1,2,3,4} got edge = $edge"

    if type == 1
        QM.step_left_flip!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 2
        QM.step_right_flip!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 3
        QM.step_split!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 4
        QM.step_collapse!(env, quad, edge, no_action_reward=no_action_reward)
    elseif type == 5
        QM.step_nothing!(env)
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

function plot_env(wrapper; elem_numbers = false, internal_order=false)
    env = wrapper.env
    QM.reindex_game_env!(env)
    
    mesh = env.mesh
    vs = QM.active_vertex_score(env)
    fig, ax = PQ.plot_mesh(
        QM.active_vertex_coordinates(mesh),
        QM.active_quad_connectivity(mesh),
        vertex_score = vs,
        elem_numbers=elem_numbers,
        internal_order=internal_order
    )
    return fig
end

function smooth_mesh!(wrapper)
    QM.averagesmoothing!(wrapper.env.mesh)
end

function single_trajectory_return(wrapper, policy)
    env = wrapper.env

    done = PPO.is_terminal(wrapper)
    if done
        return 0.0
    else
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
end

function single_trajectory_normalized_return(wrapper, policy)
    env = wrapper.env
    maxreturn = env.current_score - env.opt_score
    if maxreturn == 0
        return 1.0
    else
        ret = single_trajectory_return(wrapper, policy)
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