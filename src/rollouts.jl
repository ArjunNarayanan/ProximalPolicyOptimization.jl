function collect_step_data!(buffer, env, policy)
    cpu_state = state(env)
    gpu_state = cpu_state |> gpu

    ap = action_probabilities(policy, gpu_state) |> cpu
    a = rand(Categorical(ap))
    @assert ap[a] > 0.0

    step!(env, a)

    r = reward(env)
    t = is_terminal(env)

    update!(buffer, cpu_state, ap[a], a, r, t)
end

function collect_episode_data!(episode_data, env, policy)
    terminal = is_terminal(env)

    while !terminal
        collect_step_data!(episode_data, env, policy)
        terminal = is_terminal(env)
    end
end

function compute_returns(rewards, terminal, discount)
    ne = length(rewards)
    
    T = eltype(rewards)
    values = zeros(T, ne)
    v = zero(T)

    for idx = ne:-1:1
        if terminal[idx]
            v = zero(T)
        end
        v = rewards[idx] + discount * v
        values[idx] = v
    end

    return values
end

function compute_state_value!(rollouts, discount)
    rollouts.rewards .= compute_returns(rollouts.rewards, rollouts.terminal, discount)
end

function collect_rollouts!(rollouts, env, policy, num_episodes)
    for _ in 1:num_episodes
        reset!(env)
        collect_episode_data!(rollouts, env, policy)
    end
end

function permute!(rollouts, idx)
    @assert length(idx) == length(rollouts)
    rollouts.state_data .= rollouts.state_data[idx]
    rollouts.selected_action_probabilities .= rollouts.selected_action_probabilities[idx]
    rollouts.selected_actions .= rollouts.selected_actions[idx]
    rollouts.rewards .= rollouts.rewards[idx]
    rollouts.terminal .= rollouts.terminal[idx]
end

function shuffle!(rollouts)
    idx = shuffle(1:length(rollouts))
    permute!(rollouts, idx)
end