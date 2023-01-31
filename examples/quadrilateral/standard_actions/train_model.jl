include("quad_game_utilities.jl")
include("random_polygon_environment.jl")


discount = 1.0
epsilon = 0.05
minibatch_size = 32
episodes_per_iteration = 20
num_epochs = 10
num_iter = 500
quad_alg = "catmull-clark"

num_evaluation_trajectories = 100
output_dir = "examples/quadrilateral/template_action_mask/level4"
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

poly_degree = 10
max_actions = 20
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)

PPO.reset!(wrapper)
plot_wrapper(wrapper)

policy = SimplePolicy.Policy(216, 128, 2, 4)
optimizer = ADAM(1e-4)

PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_iter, evaluator)


# using PyPlot
# fig, ax = plot_normalized_returns(evaluator.mean_returns, evaluator.std_returns)
# ax.set_title("Average returns vs training iterations for 4-level template")
# fig.savefig("examples/quadrilateral/test_global_split/output/level4/figures/learning_curve.png")
# ret, dev = average_normalized_best_returns(policy, wrapper, 100)

# PPO.reset!(wrapper)
# output_dir = "examples/quadrilateral/test_global_split/output/level4/figures/rollout-5/"
# plot_trajectory(policy, wrapper, output_dir)
