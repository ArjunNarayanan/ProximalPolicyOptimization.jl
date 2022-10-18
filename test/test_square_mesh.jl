include("quad_game_utilities.jl")
include("square_mesh_env.jl")

function evaluator(policy, wrapper; num_trajectories=100)
    ret, dev = average_returns(wrapper, policy, num_trajectories)
    return ret, dev
end

mesh_size = 3
num_rand_actions = 5
max_actions = 10
discount = 1.0
epsilon = 0.05
batch_size = 10
epochs_per_iteration = 5
episodes_per_iteration = 20
num_iter = 1000

wrapper = SquareMeshWrapper(mesh_size, num_rand_actions, max_actions)
policy = SimplePolicy.Policy(72, 128, 2, 4)
optimizer = ADAM(1e-4)

ret, dev = PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, discount, 
epsilon, batch_size, epochs_per_iteration, num_iter, evaluator)

ret, dev = average_normalized_returns(wrapper, policy, 100)

PPO.reset!(wrapper)


using PyPlot
fig, ax = subplots()
ax.plot(ret)
ax.grid()
ax.set_xlabel("PPO Iterations")
ax.set_ylabel("Average returns")
ax.set_title("Average returns vs PPO iterations for SquareMesh environment")
fig.tight_layout()
fig.savefig("output/square-mesh-returns.png")