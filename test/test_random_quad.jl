include("quad_game_utilities.jl")
include("rand_poly_env.jl")

function evaluator(policy, wrapper; num_trajectories = 100)
    ret, dev = average_returns(wrapper, policy, num_trajectories)
    return ret, dev
end

function count_non_flips(wrapper, policy)
    env = wrapper.env
    counter = 0
    
    done = PPO.is_terminal(wrapper)
    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))
        q, e, t = index_to_action(action)
        if t > 2
            counter += 1
        end

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)
    end
    
    return counter
end

function rollouts(wrapper, policy; numtrials = 1000)
    nonflips = 0
    for trial in 1:numtrials
        PPO.reset!(wrapper)
        nonflips += count_non_flips(wrapper, policy)
    end
    return nonflips/numtrials
end

poly_degree = 5
max_actions = 10

discount = 1.0
epsilon = 0.02
batch_size = 10
episodes_per_iteration = 20
num_epochs = 5
num_iter = 1000

wrapper = RandPolyEnv(poly_degree, max_actions)
# policy = SimplePolicy.Policy(72, 128, 2, 4)
# optimizer = ADAM(1e-4)

PPO.reset!(wrapper)
fig = plot_wrapper(wrapper)

# ret, dev = PPO.ppo_iterate!(policy, wrapper, optimizer, episodes_per_iteration, 
# discount, epsilon, batch_size, num_epochs, num_iter, evaluator)


# using PyPlot
# fig, ax = subplots()
# ax.plot(ret)
# ax.grid()
# fig


# using BSON: @save
# @save "test/output/poly-30-policy.bson" policy