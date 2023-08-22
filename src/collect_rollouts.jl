function collect_step_data!(buffer, env, policy)
    cpu_state = state(env)
    
    println("Using CPU state")

    ap = action_probabilities(policy, cpu_state)
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