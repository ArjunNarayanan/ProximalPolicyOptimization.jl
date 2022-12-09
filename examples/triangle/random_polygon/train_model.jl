include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")

using BSON
using BSON: @save, @load

output_dir = "examples/triangle/random_polygon/output/level4"
polygon_degree = 20
hmax = 0.25
max_actions = 100

epsilon = 0.10
episodes_per_iteration = 20
minibatch_size = 32
num_ppo_iterations = 100
num_evaluation_trajectories = 100


@load "examples/triangle/random_polygon/output/level2/best_model.bson" policy
# policy = BSON.load("examples/triangle/random_polygon/output/level4/best_model.bson")[:d]["policy"]

# policy = Policy(96, 32, 2, ACTIONS_PER_EDGE)
wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)

# plot_wrapper(wrapper)

# ret, dev = average_normalized_best_returns(policy, wrapper, 100)

# optimizer = Adam(1e-4)
# evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

# PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_ppo_iterations, evaluator,
#     epsilon=epsilon)

# using PyPlot
# fig, ax = subplots()
# ax.plot(evaluator.mean_returns)
# fig

PPO.reset!(wrapper)
plot_trajectory(policy, wrapper, "examples/triangle/random_polygon/output/level2/figures/rollout-5")