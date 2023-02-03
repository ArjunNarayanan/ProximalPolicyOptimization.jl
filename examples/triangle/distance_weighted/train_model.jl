include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")

polygon_degree = 20
hmax = 0.25
max_actions = 100
discount = 1.0
epsilon = 0.05
episodes_per_iteration = 20
minibatch_size = 32
num_ppo_iterations = 500
num_evaluation_trajectories = 100

root_dir = "/Users/arjun/.julia/dev/ProximalPolicyOptimization/examples/triangle/distance_weighted/"
output_dir = joinpath(root_dir, "output")


policy = Policy(144, 128, 1, ACTIONS_PER_EDGE)
wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)

optimizer = Adam(1e-4)
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)


PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_ppo_iterations, evaluator,
    epsilon=epsilon)



# plot_wrapper(wrapper)
# ret, dev = average_normalized_best_returns(policy, wrapper, 100)

# using PyPlot
# fig, ax = subplots()
# ax.plot(evaluator.mean_returns)
# fig

# PPO.reset!(wrapper)
# plot_trajectory(policy, wrapper, "examples/triangle/random_polygon/output/level2/figures/rollout-5")