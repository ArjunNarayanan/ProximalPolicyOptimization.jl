function step_batch!(policy, optimizer, state, current_action_probabilities, advantage, selected_actions, epsilon)
    weights = Flux.params(policy)
    local loss
    grad = Flux.gradient(weights) do
        loss = ppo_loss(policy, state, selected_actions, current_action_probabilities, advantage, epsilon)
        return loss
    end

    Flux.update!(optimizer, weights, grad)

    return loss
end

function step_epoch!(policy, optimizer, rollouts, epsilon, batch_size)
    num_data = length(rollouts)
    start = 1
    loss = []
    while start <= num_data
        stop = min(start + batch_size - 1, num_data)

        state = batch_state(rollouts.state_data[start:stop])
        current_action_probabilities = rollouts.action_probabilities[start:stop]
        advantage = rollouts.rewards[start:stop]
        selected_actions = rollouts.selected_actions[start:stop]

        l = step_batch!(policy, optimizer, state, current_action_probabilities, advantage, selected_actions, epsilon)
        append!(loss, l)
    end
end