mutable struct Rollouts
    state_data_directory
    selected_action_probabilities
    selected_actions
    rewards
    terminal
    num_samples
    trajectory_filename
end

function prepare_state_data_directory(path)
    if isdir(path)
        rm(path, recursive=true)
    end
    state_data_path = joinpath(path, "states")
    mkpath(state_data_path)
end

function Rollouts(state_data_dir)
    selected_action_probabilities = Float32[]
    selected_actions = Int[]
    rewards = Float32[]
    terminal = Bool[]
    num_samples = 0
    trajectory_filename = "trajectory.csv"
    prepare_state_data_directory(state_data_dir)
    
    Rollouts(state_data_dir, selected_action_probabilities, selected_actions, rewards, terminal, num_samples, trajectory_filename)
end

function write_state_to_disk(buffer::Rollouts, state)
    state_file = "sample_" * string(buffer.num_samples) * ".bson"
    state_file_path = joinpath(buffer.state_data_directory, "states", state_file)
    BSON.@save state_file_path state
end

function update!(buffer::Rollouts, state, action_probability, action, reward, terminal)
    @assert 0 <= action_probability <= 1
    @assert terminal isa Bool

    buffer.num_samples += 1
    write_state_to_disk(buffer, state)
    push!(buffer.selected_action_probabilities, action_probability)
    push!(buffer.selected_actions, action)
    push!(buffer.rewards, reward)
    push!(buffer.terminal, terminal)
end

function Base.length(buffer::Rollouts)
    @assert length(buffer.selected_action_probabilities) ==
            length(buffer.selected_actions) ==
            length(buffer.rewards) ==
            length(buffer.terminal)

    return length(buffer.terminal)
end

function Base.show(io::IO, data::Rollouts)
    nd = length(data)
    println(io, "Rollouts\n\t$nd data points")
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

function collect_rollouts!(buffer::Rollouts, env, policy, num_episodes, discount)
    println("\n\nCOLLECTING ROLLOUTS :")
    for _ in 1:num_episodes
        reset!(env)
        collect_episode_data!(buffer, env, policy)
    end
    cumulative_returns = compute_returns(buffer.rewards, buffer.terminal, discount)
    file_names = ["sample_" * string(i) * ".bson" for i in 1:buffer.num_samples]
    data = Dict("sample_names" => file_names, 
                "selected_actions" => buffer.selected_actions,
                "selected_action_probabilities" => buffer.selected_action_probabilities,
                "returns" => cumulative_returns,
    )
    df = DataFrame(data)
    df = df[!, ["sample_names", "selected_actions", "selected_action_probabilities", "returns"]]
    
    df_file_path = joinpath(buffer.state_data_directory, buffer.trajectory_filename)
    CSV.write(df_file_path, df)
end