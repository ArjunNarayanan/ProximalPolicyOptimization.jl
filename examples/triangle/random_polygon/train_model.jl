include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")


output_dir = "examples/triangle/random_polygon/output"
polygon_degree = 10
hmax = 0.20
max_actions = 50

epsilon = 0.10
episodes_per_iteration = 10
minibatch_size = 32
epochs_per_iteration = 20
num_ppo_iterations = 1000
num_evaluation_trajectories = 100

wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)
policy = Policy(24, 32, 2, ACTIONS_PER_EDGE)
optimizer = Adam(1e-4)
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_ppo_iterations, evaluator,
epsilon = epsilon)

# ret, dev = average_best_returns(policy, wrapper, 100)

# PPO.reset!(wrapper)
# plot_trajectory(policy, wrapper, "examples/triangle/random_polygon/output/figures/rollout-8")




# using BSON: @save
# @save "examples/triangle/random_polygon/output/best_model.bson" policy 

# using PyPlot
# fig, ax = subplots()
# ax.plot(ret)
# fig