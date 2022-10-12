struct EpisodeData
    state_data::Any
    selected_action_probabilities::Any
    selected_actions::Any
    rewards::Any
    terminal
end

function EpisodeData()
    state_data = []
    selected_action_probabilities = Float64[]
    selected_actions = Int64[]
    rewards = Float64[]
    terminal = Bool[]
    EpisodeData(state_data, selected_action_probabilities, selected_actions, rewards, terminal)
end

function update!(episode::EpisodeData, state, action_probability, action, reward, terminal)
    push!(episode.state_data, state)
    push!(episode.selected_action_probabilities, action_probability)
    push!(episode.selected_actions, action)
    push!(episode.rewards, reward)
    push!(episode.terminal, terminal)
    return
end

function Base.length(b::EpisodeData)
    @assert length(b.selected_action_probabilities) ==
            length(b.selected_actions) ==
            length(b.rewards) ==
            length(b.state_data) ==
            length(b.terminal)

    return length(b.selected_action_probabilities)
end

function Base.show(io::IO, data::EpisodeData)
    nd = length(data)
    println(io, "EpisodeData\n\t$nd data points")
end

function collect_step_data!(episode_data, env, policy)
    s = state(env)
    ap = action_probabilities(policy, s)
    a = rand(Categorical(ap))

    step!(env, a)

    r = reward(env)
    t = is_terminal(env)

    update!(episode_data, s, ap[a], a, r, t)
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

    values = zeros(ne)
    v = 0.0

    for idx = ne:-1:1
        if terminal[idx]
            v = 0.0
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