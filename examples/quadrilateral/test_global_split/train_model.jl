include("quad_game_utilities.jl")
include("rand_poly_env.jl")


discount = 1.0
epsilon = 0.05
minibatch_size = 32
episodes_per_iteration = 20
num_epochs = 10
num_iter = 200
quad_alg = "catmull-clark"

num_evaluation_trajectories = 100
output_dir = "examples/quadrilateral/test_global_split/output/level4"
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

poly_degree = 10
max_actions = 30
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)

policy = SimplePolicy.Policy(108, 128, 2, 4)

optimizer = ADAM(1e-4)
ret, dev = PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_iter, evaluator)


# using PyPlot
# fig, ax = subplots()
# ax.plot(ret)
# ax.grid()
# ax.set_xlabel("PPO Iterations")
# ax.set_ylabel("Average returns")
# fig.savefig("test/output/figures/catmull-clark-training-returns.png")


# using BSON: @save
# @save "test/output/poly-30-policy.bson" policy
# ret, dev = average_normalized_returns(wrapper, policy, 100)
# PPO.reset!(wrapper)
# plot_trajectory(policy, wrapper, "test/output/figures/rollouts/rollout-4/")
