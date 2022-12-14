include("quad_game_utilities.jl")
include("rand_poly_env.jl")

function evaluator(policy, wrapper; num_trajectories = 50)
    ret, dev = average_best_returns(wrapper, policy, num_trajectories)
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



discount = 1.0
epsilon = 0.05
batch_size = 10
episodes_per_iteration = 20
num_epochs = 10
num_iter = 1000
quad_alg = "catmull-clark"

poly_degree = 10
max_actions = 30
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg = quad_alg)

using BSON
d = BSON.load("test/output/catmull-clark-policy-l4.bson")
policy = d[:policy]

# ret, dev = average_normalized_returns(wrapper, policy, 100)

PPO.reset!(wrapper)
plot_trajectory(policy, wrapper, "test/output/figures/rollouts/rollout-4/")

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