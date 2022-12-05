function simplified_ppo_clip(epsilon, advantage)
    return [a >= 0 ? (1 + epsilon) * a : (1 - epsilon) * a for a in advantage]
end

function ppo_loss(policy, state, actions, old_action_probabilities, advantage, epsilon)
    ap = batch_action_probabilities(policy, state)
    selected_ap = [ap[a, idx] for (idx, a) in enumerate(actions)]

    ppo_gain = @. selected_ap / old_action_probabilities * advantage
    ppo_clip = simplified_ppo_clip(epsilon, advantage)

    loss = -Flux.mean(min.(ppo_gain, ppo_clip))

    return loss
end

function step_batch!(policy, optimizer, state, selected_actions, old_action_probabilities, advantage, epsilon)
    weights = Flux.params(policy)
    local loss
    grad = Flux.gradient(weights) do
        loss = ppo_loss(policy, state, selected_actions, old_action_probabilities, advantage, epsilon)
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
        current_action_probabilities = rollouts.selected_action_probabilities[start:stop]
        advantage = rollouts.rewards[start:stop]
        selected_actions = rollouts.selected_actions[start:stop]

        l = step_batch!(policy, optimizer, state, selected_actions, current_action_probabilities, advantage, epsilon)
        append!(loss, l)

        start = stop + 1
    end
    return Flux.mean(loss)
end

function ppo_train!(
    policy,
    optimizer,
    rollouts,
    epsilon,
    batch_size,
    num_epochs,
)
    for epoch = 1:num_epochs
        shuffle!(rollouts)
        l = step_epoch!(policy, optimizer, rollouts, epsilon, batch_size)
        @printf "EPOCH : %d \t AVG LOSS : %1.4f\n" epoch l
    end
end

function ppo_iterate!(
    policy,
    env,
    optimizer,
    episodes_per_iteration,
    minibatch_size,
    num_ppo_iterations;
    epochs_per_iteration = 10,
    discount = 0.95,
    epsilon = 0.05,
    num_evaluation_trajectories = 100
)

    returns, deviation = [], []
    for iter in 1:num_ppo_iterations
        println("\nPPO ITERATION : $iter")

        rollouts = EpisodeData()
        collect_rollouts!(rollouts, env, policy, episodes_per_iteration)

        prepare_state_data_for_batching(rollouts.state_data)
        compute_state_value!(rollouts, discount)

        ppo_train!(policy, optimizer, rollouts, epsilon, minibatch_size, epochs_per_iteration)

        ret, dev = average_returns(policy, env, num_evaluation_trajectories)
        push!(returns, ret)
        push!(deviation, dev)

        @printf "RET = %1.4f\tDEV = %1.4f\n" ret dev
    end
    return returns, deviation
end
