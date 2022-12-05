include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")


polygon_degree = 10
hmax = 0.30
max_actions = 50

episodes_per_iteration = 10
minibatch_size = 32
epochs_per_iteration = 10
num_ppo_iterations = 200

wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)
# policy = Policy(24, 32, 2, ACTIONS_PER_EDGE)
# optimizer = Adam(1e-4)

# ret, dev = PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, minibatch_size, num_ppo_iterations)

ret, dev = average_best_returns(policy, wrapper, 100)

PPO.reset!(wrapper)
plot_trajectory(policy, wrapper, "examples/triangle/random_polygon/output/figures/rollout-8")




# using BSON: @save
# @save "examples/triangle/random_polygon/output/best_model.bson" policy 

# using PyPlot
# fig, ax = subplots()
# ax.plot(ret)
# fig