include("quad_game_utilities.jl")
include("../random_polygon_environment.jl")


discount = 1.0
epsilon = 0.05
minibatch_size = 32
episodes_per_iteration = 20
num_epochs = 10
num_iter = 500
quad_alg = "catmull-clark"
root_dir = "/Users/arjun/.julia/dev/ProximalPolicyOptimization/examples/quadrilateral/global_split/"

num_evaluation_trajectories = 100
output_dir = joinpath(root_dir, "output")
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

poly_degree = 10
max_actions = 20
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)

policy = SimplePolicy.Policy(216, 128, 1, 5)
optimizer = ADAM(1e-4)

# PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_iter, evaluator)


data = BSON.load("examples/quadrilateral/global_split/output/best_model.bson")[:d]
policy = data["policy"]
evaluator = data["evaluator"]


# poly_degree = 10
# max_actions = 100
# wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)
# PPO.reset!(wrapper)
# plot_wrapper(wrapper, number_elements = false)

# probs = PPO.action_probabilities(policy, PPO.state(wrapper))
# action = rand(Categorical(probs))
# q, e, t = index_to_action(action)
# PPO.step!(wrapper, action)
# plot_wrapper(wrapper, number_elements = false)


# m, s, a = average_normalized_returns_and_action_stats(policy, wrapper, 100)

# plot_wrapper(wrapper, number_elements = true)
# step_wrapper!(wrapper, 29, 2, 5)

# PPO.reset!(wrapper)
# fig_output_dir = joinpath(output_dir, "rollout-3")
# plot_trajectory(policy, wrapper, fig_output_dir)
