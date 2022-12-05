function single_trajectory_return(policy, env)
    ret = 0
    done = is_terminal(env)

    while !done
        probs = action_probabilities(policy, state(env))
        action = rand(Categorical(probs))

        step!(env, action)

        done = is_terminal(env)
        ret += reward(env)
    end
    return ret
end

function average_returns(policy, env, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        reset!(env)
        ret[idx] = single_trajectory_return(policy, env)
    end
    return Flux.mean(ret), Flux.std(ret)
end
