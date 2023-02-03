include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")

using BenchmarkTools

function collect_rollouts(env, policy, episodes_per_iteration)
    rollouts = PPO.EpisodeData()
    PPO.collect_rollouts!(rollouts, env, policy, episodes_per_iteration)
end

function batch_state(rollouts, batch_size)
    state = PPO.batch_state(rollouts.state_data[1:batch_size])
end

function get_gradient(policy, state, selected_actions, old_action_probabilities, advantage, epsilon)
    weights = Flux.params(policy)
    grad = Flux.gradient(weights) do
        loss = PPO.ppo_loss(policy, state, selected_actions, old_action_probabilities, advantage, epsilon)
    end
    return grad
end


output_dir = "examples/triangle/random_polygon/output/level4"
polygon_degree = 20
hmax = 0.25
max_actions = 50

epsilon = 0.10
episodes_per_iteration = 20
minibatch_size = 32
num_ppo_iterations = 100
num_evaluation_trajectories = 100


policy = Policy(96, 32, 2, ACTIONS_PER_EDGE)
wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)
optimizer = Adam(1e-4)


@benchmark collect_rollouts($wrapper, $policy, $episodes_per_iteration)

rollouts = PPO.EpisodeData()
PPO.collect_rollouts!(rollouts, wrapper, policy, episodes_per_iteration)

PPO.prepare_state_data_for_batching!(rollouts.state_data);
@time PPO.prepare_state_data_for_batching!(rollouts.state_data);
@time PPO.prepare_state_data_for_batching!(rollouts.state_data);

@benchmark PPO.shuffle!($rollouts)

@benchmark batch_state($rollouts, 32)

start = 1
stop = minibatch_size

state = PPO.batch_state(rollouts.state_data[start:stop])
ap = PPO.batch_action_probabilities(policy, state)

@benchmark PPO.batch_action_probabilities($policy, $state)

current_action_probabilities = rollouts.selected_action_probabilities[start:stop]
advantage = rollouts.rewards[start:stop]
selected_actions = rollouts.selected_actions[start:stop]

get_gradient(policy, state, selected_actions, current_action_probabilities, advantage, epsilon)

@benchmark get_gradient($policy, $state, $selected_actions, $current_action_probabilities, $advantage, $epsilon)

weights = Flux.params(policy)
grad = get_gradient(policy, state, selected_actions, current_action_probabilities, advantage, epsilon)
Flux.update!(optimizer, weights, grad)

@benchmark Flux.update!($optimizer, $weights, $grad)


@benchmark PPO.step_epoch!($policy, $optimizer, $rollouts, $epsilon, $minibatch_size)