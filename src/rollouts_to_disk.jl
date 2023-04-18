mutable struct ARTrajectory
    state_data_directory
    selected_action_probabilities
    selected_actions
    rewards
    terminal
    num_samples
end

function prepare_state_data_directory(path)
    if isdir(path)
        rm(path, recursive=true)
    end
    state_data_path = joinpath(path, "states")
    mkpath(state_data_path)
end

function ARTrajectory(state_data_dir)
    selected_action_probabilities = Float32[]
    selected_actions = Int[]
    rewards = Float32[]
    terminal = Bool[]
    num_samples = 0
    prepare_state_data_directory(state_data_dir)
    
    ARTrajectory(state_data_dir, selected_action_probabilities, selected_actions, rewards, terminal, num_samples)
end

function write_state_to_disk(buffer::ARTrajectory, state)
    state_file = "sample_" * string(buffer.num_samples) * ".bson"
    state_file_path = joinpath(buffer.state_data_directory, "states", state_file)
    BSON.@save state_file_path state
end

function update!(buffer::ARTrajectory, state, action_probability, action, reward, terminal)
    @assert 0 <= action_probability <= 1
    @assert terminal isa Bool

    buffer.num_samples += 1
    write_state_to_disk(buffer, state)
    push!(buffer.selected_action_probabilities, action_probability)
    push!(buffer.selected_actions, action)
    push!(buffer.rewards, reward)
    push!(buffer.terminal, terminal)
end

function Base.length(buffer::ARTrajectory)
    @assert length(buffer.selected_action_probabilities) ==
            length(buffer.selected_actions) ==
            length(buffer.rewards) ==
            length(buffer.terminal)

    return length(buffer.terminal)
end

function Base.show(io::IO, data::ARTrajectory)
    nd = length(data)
    println(io, "ARTrajectory\n\t$nd data points")
end

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

function collect_episode_data!(buffer, env, policy)
    terminal = is_terminal(env)

    while !terminal
        collect_step_data!(buffer, env, policy)
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

function collect_rollouts!(buffer::ARTrajectory, env, policy, num_episodes, discount)
    for _ in 1:num_episodes
        reset!(env)
        collect_episode_data!(buffer, env, policy)
    end
    state_values = compute_returns(buffer.rewards, buffer.terminal, discount)
    file_names = ["sample_" * string(i) * ".bson" for i in 1:buffer.num_samples]
    data = Dict("sample_names" => file_names, 
                "selected_actions" => buffer.selected_actions,
                "selected_action_probabilities" => buffer.selected_action_probabilities,
                "state_values" => state_values,
    )
    df = DataFrame(data)
    df_file_path = joinpath(buffer.state_data_directory, "trajectory.csv")
    CSV.write(df_file_path, df)
end