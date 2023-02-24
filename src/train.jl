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

function ppo_loss_with_entropy(policy, state, actions, old_action_probabilities, advantage, epsilon)
    current_action_probabilities = batch_action_probabilities(policy, state)
    selected_action_probabilities = current_action_probabilities[actions]

    ppo_gain = @. selected_action_probabilities / old_action_probabilities * advantage
    ppo_clip = simplified_ppo_clip.(advantage, epsilon)

    ppoloss = -Flux.mean(min.(ppo_gain, ppo_clip))
    entropyloss = smoothed_entropy(current_action_probabilities)

    return ppoloss, entropyloss
end

function get_linear_action_index(selected_actions, num_actions_per_state)
    num_states = length(selected_actions)
    offset = range(start = 0, step = num_actions_per_state, length = num_states)
    return selected_actions + offset
end

function step_batch!(policy, optimizer, state, linear_action_index, old_action_probabilities, advantage, epsilon, entropy_weight)
    weights = Flux.params(policy)
    local ppoloss, entropyloss
    grad = Flux.gradient(weights) do
        ppoloss, entropyloss = ppo_loss_with_entropy(
            policy, 
            state, 
            linear_action_index, 
            old_action_probabilities, 
            advantage, 
            epsilon
        )
        entropyloss = entropyloss * entropy_weight
        loss = ppoloss - entropyloss
        return loss
    end

    Flux.update!(optimizer, weights, grad)

    return ppoloss, entropyloss
end

function step_epoch!(policy, optimizer, rollouts, epsilon, batch_size, entropy_weight)
    num_data = length(rollouts)
    start = 1
    ppo_loss_history, entropy_loss_history = [], []
    while start <= num_data
        stop = min(start + batch_size - 1, num_data)

        state = batch_state(rollouts.state_data[start:stop]) |> gpu
        
        current_action_probabilities = rollouts.selected_action_probabilities[start:stop] |> gpu
        advantage = rollouts.rewards[start:stop] |> gpu
        selected_actions = rollouts.selected_actions[start:stop] |> gpu
        
        num_actions_per_state = number_of_actions_per_state(state)
        linear_action_index = get_linear_action_index(selected_actions, num_actions_per_state)
        
        ppoloss, entropyloss = step_batch!(
            policy, 
            optimizer, 
            state, 
            linear_action_index, 
            current_action_probabilities, 
            advantage, 
            epsilon, 
            entropy_weight
        )
        push!(ppo_loss_history, ppoloss)
        push!(entropy_loss_history, entropyloss)

        start = stop + 1
    end
    return Flux.mean(ppo_loss_history), Flux.mean(entropy_loss_history)
end

function ppo_train!(
    policy,
    optimizer,
    rollouts,
    epsilon,
    batch_size,
    num_epochs,
    entropy_weight
)
    ppo_loss_history = []
    entropy_loss_history = []
    for epoch = 1:num_epochs
        shuffle!(rollouts)
        ppoloss, entropyloss = step_epoch!(policy, optimizer, rollouts, epsilon, batch_size, entropy_weight)
        @printf "EPOCH : %d \t PPO LOSS : %1.4f\t ENTROPY LOSS : %1.4f\n" epoch ppoloss entropyloss
        push!(ppo_loss_history, ppoloss)
        push!(entropy_loss_history, entropyloss)
    end
    return ppo_loss_history, entropy_loss_history
end

function ppo_iterate!(
    policy,
    env,
    optimizer,
    episodes_per_iteration,
    minibatch_size,
    num_ppo_iterations,
    evaluator,
    epochs_per_iteration,
    discount,
    epsilon,
    entropy_weight
)

    loss = Dict("ppo" => [], "entropy" => [])
    for iter in 1:num_ppo_iterations
        evaluator(policy, env)

        println("\nPPO ITERATION : $iter")

        rollouts = EpisodeData()
        collect_rollouts!(rollouts, env, policy, episodes_per_iteration)

        compute_state_value!(rollouts, discount)
        rollouts = prepare_rollouts_for_training(rollouts)

        ppoloss, entropyloss = ppo_train!(policy, optimizer, rollouts, epsilon, minibatch_size, epochs_per_iteration, entropy_weight)
        append!(loss["ppo"], ppoloss)
        append!(loss["entropy"], entropyloss)

        save_loss(evaluator, loss)
    end
end
