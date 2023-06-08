mutable struct Rollouts
    state_data_directory
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

function prepare_trajectory_data_file(filename)
    directory = dirname(filename)
    @assert isdir(directory)
    if isfile(filename)
        rm(filename)
    end
end

function Rollouts(state_data_dir)
    selected_action_probabilities = Float32[]
    selected_actions = Int[]
    rewards = Float32[]
    terminal = Bool[]
    num_samples = 0

    prepare_state_data_directory(state_data_dir)
    trajectory_filename = joinpath(state_data_dir, "trajectory.csv")
    prepare_trajectory_data_file(trajectory_filename)

    header = [
        "sample_names",
        "selected_actions",
        "selected_action_probabilities",
        "rewards",
        "terminal"
    ]
    line = Tables.table(reshape(header, 1, :))
    CSV.write(trajectory_filename, line, header=false)

    Rollouts(state_data_dir, num_samples, trajectory_filename)
end

function write_state_to_disk(buffer::Rollouts, state)
    state_file = "sample_" * string(buffer.num_samples) * ".bson"
    state_file_path = joinpath(buffer.state_data_directory, "states", state_file)
    BSON.@save state_file_path state
end

function write_action_history_to_disk(
    buffer::Rollouts,
    sample_name,
    selected_action_probability,
    selected_action,
    reward,
    terminal
)

    data = [
        sample_name,
        selected_action,
        selected_action_probability,
        reward,
        terminal
    ]
    line = Tables.table(reshape(data,1,:))
    CSV.write(buffer.trajectory_filename, line, append=true)
end

function update!(buffer::Rollouts, state, action_probability, action, reward, terminal)
    @assert 0 <= action_probability <= 1
    @assert terminal isa Bool

    buffer.num_samples += 1
    sample_name = "sample_" * string(buffer.num_samples) * ".bson"
    write_state_to_disk(buffer, state)
    write_action_history_to_disk(
        buffer,
        sample_name,
        action_probability,
        action,
        reward,
        terminal
    )
end

function Base.length(buffer::Rollouts)
    return buffer.num_samples
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

function write_returns_to_disk(buffer::Rollouts, discount)
    column_types = Dict(
            "selected_actions" => Int,
            "selected_action_probabilities" => Float32,
            "rewards" => Float32,
            "terminal" => Bool
        )
    df = CSV.read(buffer.trajectory_filename, DataFrame, types = column_types)
    cumulative_returns = compute_returns(
        df[!, "rewards"],
        df[!, "terminal"],
        discount
    )
    new_df = DataFrame(Dict(
        "sample_names" => df[!, "sample_names"],
        "selected_actions" => df[!, "selected_actions"],
        "selected_action_probabilities" => df[!, "selected_action_probabilities"],
        "returns" => cumulative_returns
    ))
    new_df = new_df[!, [
        "sample_names", 
        "selected_actions",
        "selected_action_probabilities",
        "returns"
    ]]
    CSV.write(buffer.trajectory_filename, new_df)
end

function collect_rollouts!(buffer::Rollouts, env, policy, num_episodes, discount)
    println("\n\nCOLLECTING ROLLOUTS :")
    for _ in 1:num_episodes
        reset!(env)
        collect_episode_data!(buffer, env, policy)
    end
    write_returns_to_disk(buffer, discount)
end

function archive_collect_rollouts!(buffer::Rollouts, env, policy, num_episodes, discount)
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