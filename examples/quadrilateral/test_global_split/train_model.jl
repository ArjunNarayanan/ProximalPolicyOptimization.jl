include("quad_game_utilities.jl")
include("rand_poly_env.jl")

function evaluator(policy, wrapper; num_trajectories = 50)
    ret, dev = average_best_returns(wrapper, policy, num_trajectories)
    return ret, dev
end



mesh = QM.square_mesh(3)
vertex_score = zeros(Int, 16)
vertex_score[6] = 1
vertex_score[10] = -1
vertex_score[5] = 1

update_vertex_score_for_global_split!(vertex_score, mesh)

PQ.plot_mesh(QM.active_vertex_coordinates(mesh), QM.active_quad_connectivity(mesh),
number_vertices = true, number_elements = true)[1]


# discount = 1.0
# epsilon = 0.05
# batch_size = 10
# episodes_per_iteration = 20
# num_epochs = 10
# num_iter = 1000
# quad_alg = "catmull-clark"

# poly_degree = 10
# max_actions = 30
# wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg = quad_alg)

# using BSON
# d = BSON.load("test/output/catmull-clark-policy-l4.bson")
# policy = d[:policy]

# ret, dev = average_normalized_returns(wrapper, policy, 100)

# PPO.reset!(wrapper)
# plot_trajectory(policy, wrapper, "test/output/figures/rollouts/rollout-4/")

# policy = SimplePolicy.Policy(216, 128, 2, 4)
# optimizer = ADAM(1e-4)

# ret, dev = PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, 
# discount, epsilon, batch_size, num_epochs, num_iter, evaluator)


# using PyPlot
# fig, ax = subplots()
# ax.plot(ret)
# ax.grid()
# ax.set_xlabel("PPO Iterations")
# ax.set_ylabel("Average returns")
# fig.savefig("test/output/figures/catmull-clark-training-returns.png")


# using BSON: @save
# @save "test/output/poly-30-policy.bson" policy