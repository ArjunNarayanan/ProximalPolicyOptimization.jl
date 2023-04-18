struct EpisodeData
    state_data::Any
    selected_action_probabilities::Any
    selected_actions::Any
    rewards::Any
    terminal
end

function EpisodeData()
    state_data = []
    selected_action_probabilities = Float32[]
    selected_actions = Int64[]
    rewards = Float32[]
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