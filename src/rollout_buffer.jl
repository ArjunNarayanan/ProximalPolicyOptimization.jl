struct BufferRollouts
    state_data::Any
    selected_action_probabilities::Any
    selected_actions::Any
    rewards::Any
    terminal
end

function BufferRollouts()
    state_data = []
    selected_action_probabilities = Float32[]
    selected_actions = Int64[]
    rewards = Float32[]
    terminal = Bool[]
    BufferRollouts(
        state_data, 
        selected_action_probabilities, 
        selected_actions, 
        rewards, 
        terminal
    )
end

function update!(
    episode::BufferRollouts, 
    state, 
    action_probability, 
    action, 
    reward, 
    terminal
)
    push!(episode.state_data, state)
    push!(episode.selected_action_probabilities, action_probability)
    push!(episode.selected_actions, action)
    push!(episode.rewards, reward)
    push!(episode.terminal, terminal)
    return
end

function Base.length(b::BufferRollouts)
    @assert length(b.selected_action_probabilities) ==
            length(b.selected_actions) ==
            length(b.rewards) ==
            length(b.state_data) ==
            length(b.terminal)

    return length(b.selected_action_probabilities)
end

function Base.show(io::IO, data::BufferRollouts)
    nd = length(data)
    println(io, "EpisodeData\n\t$nd data points")
end

function compute_state_value!(
    rollouts::BufferRollouts, 
    discount
)
    rollouts.rewards .= compute_returns(
        rollouts.rewards, 
        rollouts.terminal, 
        discount
    )
end

function collect_rollouts!(
    rollouts::BufferRollouts, 
    env, 
    policy, 
    num_episodes,
    discount
)

    for _ in 1:num_episodes
        reset!(env)
        collect_episode_data!(rollouts, env, policy)
    end
    compute_state_value!(rollouts, discount)
end

function permute!(rollouts::BufferRollouts, idx)
    @assert length(idx) == length(rollouts)
    rollouts.state_data .= rollouts.state_data[idx]
    rollouts.selected_action_probabilities .= rollouts.selected_action_probabilities[idx]
    rollouts.selected_actions .= rollouts.selected_actions[idx]
    rollouts.rewards .= rollouts.rewards[idx]
    rollouts.terminal .= rollouts.terminal[idx]
end

function shuffle!(rollouts::BufferRollouts)
    idx = shuffle(1:length(rollouts))
    permute!(rollouts, idx)
end

struct BufferDataset
    rollouts::BufferRollouts
end

function Base.length(dataset::BufferDataset)
    return length(dataset.rollouts)
end

function get_sample(dataset::BufferDataset, idx)
    rollouts = dataset.rollouts
    @assert idx isa Int
    @assert 1 <= idx <= length(rollouts)

    sample = Dict(
        "state" => rollouts.state_data[idx],
        "selected_action" => rollouts.selected_actions[idx],
        "selected_action_probability" => rollouts.selected_action_probabilities[idx],
        "returns" => rollouts.rewards[idx]
    )
    return sample
end

function get_batch(dataset::BufferDataset, indices)
    rollouts = dataset.rollouts

    @assert indices isa AbstractArray
    states = [s for s in rollouts.state_data[indices]]
    batched_state = batch_state(states)
    selected_action = rollouts.selected_actions[indices]
    selected_action_probabilities = rollouts.selected_action_probabilities[indices]
    returns = rollouts.rewards[indices]
    sample = Dict(
        "state" => batched_state,
        "selected_action" => selected_action,
        "selected_action_probability" => selected_action_probabilities,
        "returns" => returns
    )
    return sample
end

function Base.getindex(dataset::BufferDataset, idx)
    if idx isa Int
        return get_sample(dataset, idx)
    elseif idx isa AbstractArray
        return get_batch(dataset, idx)
    else
        error("Dataset index should be Int or Array, got ", type(idx))
    end
end

function construct_dataset(rollouts::BufferRollouts)
    return BufferDataset(rollouts)
end