function simplified_ppo_clip(advantage, epsilon)
    if advantage >= 0
        return (1.0f0 + epsilon) * advantage
    else
        return (1.0f0 - epsilon) * advantage
    end
end

function ppo_loss(policy, state, actions, old_action_probabilities, advantage, epsilon)
    current_action_probabilities = batch_action_probabilities(policy, state)
    selected_action_probabilities = current_action_probabilities[actions]

    ppo_gain = @. selected_action_probabilities / old_action_probabilities * advantage
    ppo_clip = simplified_ppo_clip.(advantage, epsilon)

    loss = -Flux.mean(min.(ppo_gain, ppo_clip))

    return loss
end

function smoothed_entropy(action_probabilities, smooth = 1f-8)
    smoothed_probabilities = (1.0f0 - smooth)*action_probabilities .+ smooth/size(action_probabilities, 1)
    h = smoothed_probabilities .* log.(smoothed_probabilities)
    h = -sum(h, dims = 1)
    return Flux.mean(h)
end

function clamped_entropy(action_probabilities, tol = 1f-8)
    clamped_action_probabilities = max.(action_probabilities, tol)
    h = clamped_action_probabilities .* log.(clamped_action_probabilities)
    h = -sum(h, dims=1)
    return Flux.mean(h)
end

function ppo_loss_with_entropy(policy, state, actions, old_action_probabilities, advantage, epsilon, entropy_weight = 0.1f0)
    current_action_probabilities = batch_action_probabilities(policy, state)
    selected_action_probabilities = current_action_probabilities[actions]

    ppo_gain = @. selected_action_probabilities / old_action_probabilities * advantage
    ppo_clip = simplified_ppo_clip.(advantage, epsilon)

    ppoloss = -Flux.mean(min.(ppo_gain, ppo_clip))
    entropyloss = clamped_entropy(current_action_probabilities)

    loss = ppoloss - entropy_weight * entropyloss 

    return loss
end

function get_linear_action_index(selected_actions, num_actions_per_state)
    num_states = length(selected_actions)
    offset = range(start = 0, step = num_actions_per_state, length = num_states)
    return selected_actions + offset
end

function step_batch!(policy, optimizer, state, linear_action_index, old_action_probabilities, advantage, epsilon)
    weights = Flux.params(policy)
    local loss
    grad = Flux.gradient(weights) do
        loss = ppo_loss_with_entropy(policy, state, linear_action_index, old_action_probabilities, advantage, epsilon)
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
        num_actions_per_state = number_of_actions_per_state(state)
        current_action_probabilities = rollouts.selected_action_probabilities[start:stop]
        advantage = rollouts.rewards[start:stop]
        selected_actions = rollouts.selected_actions[start:stop]

        linear_action_index = get_linear_action_index(selected_actions, num_actions_per_state)
        l = step_batch!(policy, optimizer, state, linear_action_index, current_action_probabilities, advantage, epsilon)
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
    num_ppo_iterations,
    evaluator;
    epochs_per_iteration = 10,
    discount = 0.95,
    epsilon = 0.05,
)

    for iter in 1:num_ppo_iterations
        evaluator(policy, env)

        println("\nPPO ITERATION : $iter")

        rollouts = EpisodeData()
        collect_rollouts!(rollouts, env, policy, episodes_per_iteration)

        compute_state_value!(rollouts, discount)
        rollouts = prepare_rollouts_for_training(rollouts)

        ppo_train!(policy, optimizer, rollouts, epsilon, minibatch_size, epochs_per_iteration)

    end
end
