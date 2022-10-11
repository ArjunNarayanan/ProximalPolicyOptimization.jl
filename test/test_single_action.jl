include("quad_game_utilities.jl")

function evaluator(policy, wrapper; num_trajectories = 100)
    ret, dev = average_normalized_returns(wrapper, policy, num_trajectories)
    return ret, dev
end

discount = 0.9
epsilon = 0.05
batch_size = 5
episodes_per_iteration = 20
num_epochs = 10
num_iter = 10

mesh0 = QM.square_mesh(2)
d0 = deepcopy(mesh0.degree)
QM.left_flip!(mesh0, 1, 3)
wrapper = GameEnvWrapper(mesh0, d0, 4)

policy = SimplePolicy.Policy(36, 64, 5)
optimizer = ADAM(1e-3)

PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, discount, epsilon, 
batch_size, num_epochs, num_iter, evaluator)
