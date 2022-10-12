include("test_quad_game_utilities.jl")

function evaluator(policy, wrapper; num_trajectories = 100)
    ret, dev = average_normalized_returns(wrapper, policy, num_trajectories)
    return ret, dev
end

discount = 0.9
epsilon = 0.05
batch_size = 5
episodes_per_iteration = 20
num_epochs = 10
num_iter = 100

mesh0 = QM.square_mesh(2)
action_list = [(1,2), (1,3), (2,1), (2,2), (3,3), (3,4), (4,1), (4,4)]
wrapper = GameEnvWrapper(mesh0, action_list, 5)


policy = SimplePolicy.Policy(36, 64, 5)
optimizer = ADAM(1e-3)


PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, discount, epsilon, 
batch_size, num_epochs, num_iter, evaluator)

wrapper = GameEnvWrapper(mesh0, action_list, 1)
ret, dev = evaluator(policy, wrapper)

# mesh0 = QM.square_mesh(2)
# d0 = deepcopy(mesh0.degree)
# QM.right_flip!(mesh0, 1, 3)
# wrapper = GameEnvWrapper(mesh0, d0, 2)
