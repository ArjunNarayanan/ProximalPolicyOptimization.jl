include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")

polygon_degree = 20
hmax = 0.25
max_actions = 20
discount = 1.0
epsilon = 0.05
episodes_per_iteration = 20
minibatch_size = 32
num_ppo_iterations = 500
num_evaluation_trajectories = 100

root_dir = "/Users/arjun/.julia/dev/ProximalPolicyOptimization/examples/triangle/distance_weighted/"
output_dir = joinpath(root_dir, "output")

wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)

policy = Policy(144, 128, 1, ACTIONS_PER_EDGE)
optimizer = Adam(1e-4)
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)
PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_ppo_iterations, evaluator,
    epsilon=epsilon)

# data = BSON.load("examples/triangle/distance_weighted/output/best_model.bson")[:d]
# policy = data["policy"]
# evaluator = data["evaluator"]


# polygon_degree = 20
# wrapper = RandPolyWrapper(polygon_degree, 0.2, 100)
# PPO.reset!(wrapper)
# output_dir = joinpath(root_dir, "output", "figures", "rollout-2")
# plot_trajectory(policy, wrapper, output_dir)