using Revise
using PlotQuadMesh
using QuadMeshGame
using ProximalPolicyOptimization

QM = QuadMeshGame
PPO = ProximalPolicyOptimization

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

function val_or_missing(vector, template, missing_val)
    return [t == 0 ? missing_val : vector[t] for t in template]
end

function PPO.state(wrapper)
    env = wrapper.env
    vs = val_or_missing(env.vertex_score, env.template, 0)
    return vs
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
    nq = QM.number_of_quads(env.mesh)
    return nq*4*actions_per_edge
end

function PPO.step!(wrapper, action_index; no_action_reward=-4)
    env = wrapper.env
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !env.is_terminated "Attempting to step in terminated environment with action $action_index"
    
    quad, edge, type = index_to_action(action_index)
    @assert QM.is_active_quad(env.mesh, quad) "Attempting to act on inactive quad $quad with action $action_index"

    @assert type in (1,2,3,4,5)

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

function plot_env(wrapper)
    env = wrapper.env
    QM.reindex_vertices!(env.mesh)
    QM.reindex_quads!(env.mesh)
end