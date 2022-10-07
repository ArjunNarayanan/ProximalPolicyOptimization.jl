struct EpisodeData
    state_data::Any
    selected_action_probabilities::Any
    selected_actions::Any
    rewards::Any
end

function EpisodeData(state_data)
    selected_action_probabilities = Float64[]
    selected_actions = Int64[]
    rewards = Float64[]
    EpisodeData(state_data, selected_action_probabilities, selected_actions, rewards)
end

function update!(episode::EpisodeData, state, action_probability, action, reward)
    update!(episode.state_data, state)
    push!(episode.selected_action_probabilities, action_probability)
    push!(episode.selected_actions, action)
    push!(episode.rewards, reward)
    return
end

function Base.length(b::EpisodeData)
    @assert length(b.selected_action_probabilities) ==
            length(b.selected_actions) ==
            length(b.rewards)
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

    update!(episode_data, s, ap[a], a, r)
end

function collect_episode_data!(episode_data, env, policy)
    terminal = is_terminal(env)

    while !terminal
        collect_step_data!(episode_data, env, policy)
        terminal = is_terminal(env)
    end
end
