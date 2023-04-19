struct Dataset
    root_directory
    trajectory_df
    states_directory
    function Dataset(root_directory, trajectory_filename = "trajectory.csv", states_dirname="states")
        trajectory_filepath = joinpath(root_directory, trajectory_filename)
        @assert isfile(trajectory_filepath)

        trajectory_df = CSV.read(trajectory_filepath, DataFrame)
        
        states_directory = joinpath(root_directory, states_dirname)
        @assert isdir(states_directory)

        new(root_directory, trajectory_df, states_directory)
    end
end

function Base.show(io::IO, dataset::Dataset)
    num_samples = size(dataset.trajectory_df, 1)
    println(io, "Dataset\n\t$num_samples data points")
end

function load_sample(dataset::Dataset, idx)
    @assert idx isa Int
    @assert 1 <= idx <= size(dataset.trajectory_df,1)

    state_filename = dataset.trajectory_df[idx,"sample_names"]
    state_filepath = joinpath(dataset.states_directory, state_filename)
    @assert isfile(state_filepath)

    state = BSON.load(state_filepath)[:state]
    action = dataset.trajectory_df[idx,"selected_actions"]
    action_prob = dataset.trajectory_df[idx,"selected_action_probabilities"]
    value = dataset.trajectory_df[idx,"state_values"]

    sample = Dict(
        "state" => state,
        "selected_action" => action,
        "selected_action_probability" => action_prob,
        "state_value" => value
    )
    
    return sample
end

function load_batch(dataset::Dataset, indices)
    @assert indices isa AbstractArray
    samples = [dataset[idx] for idx in indices]
    
    states = [s["state"] for s in samples]
    batched_state = batch_state(states)

    actions = [s["selected_action"] for s in samples]
    action_probabilities = [s["selected_action_probability"] for s in samples]
    values = [s["state_value"] for s in samples]

    batched_sample = Dict(
        "state" => batched_state,
        "selected_action" => actions,
        "selected_action_probability" => action_probabilities,
        "state_value" => values
    )
    return batched_sample
end

function Base.getindex(dataset::Dataset, idx)
    if idx isa Int
        return load_sample(dataset, idx)
    elseif idx isa AbstractArray
        return load_batch(dataset, idx)
    else
        error("Dataset index should be Int or Array, got ", type(idx))
    end
end