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
# evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

poly_degree = 10
max_actions = 20
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)

# policy = SimplePolicy.Policy(216, 128, 1, 5)
# optimizer = ADAM(1e-4)

checkpoint = BSON.load(joinpath(root_dir, "output", "best_model.bson"))[:d]
policy = checkpoint["policy"]


# PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_iter, evaluator)


max_actions = 20
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)
# ret, dev = average_normalized_best_returns(policy, wrapper, 100)

# using PyPlot
# fig, ax = plot_normalized_returns(evaluator.mean_returns, evaluator.std_returns)
# ax.set_title("Average returns vs training iterations for 4-level template")
# fig.savefig("examples/quadrilateral/test_global_split/output/level4/figures/learning_curve.png")

PPO.reset!(wrapper)
fig_output_dir = joinpath(output_dir, "rollout-2")
plot_trajectory(policy, wrapper, fig_output_dir)
