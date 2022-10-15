include("quad_game_utilities.jl")

function all_active_quads(wrapper)
    mesh = wrapper.env.mesh
    for quad in 1:QM.quad_buffer(mesh)
        if QM.is_active_quad(mesh, quad)
            nbrs = mesh.q2q[:, quad]
            is_active_nbr_or_bdry = [QM.is_active_quad_or_boundary(mesh, q) for q in nbrs]
            if any(.!is_active_nbr_or_bdry)
                return false
            end
        end
    end
    return true
end

function detect_invalid_state(wrapper, policy)
    envs = [deepcopy(wrapper)]
    actions = []
    done = PPO.is_terminal(wrapper)
    all_active = all_active_quads(wrapper)

    while !done && all_active
        
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))
        push!(actions, action)
        
        PPO.step!(wrapper, action)
        push!(envs, deepcopy(wrapper))

        done = PPO.is_terminal(wrapper)
        all_active = all_active_quads(wrapper)

    end
    return envs, actions, all_active
end

function rollout(wrapper, policy; numtrials = 1000)
    for trial in 1:numtrials
        PPO.reset!(wrapper)
        envs, actions, all_active = detect_invalid_state(wrapper, policy)
        if !all_active
            println("DETECTED INVALID STATE: ")
            return envs, actions
        end
    end
end

poly_degree = 5
max_actions = 5

discount = 0.9
epsilon = 0.05
batch_size = 10
episodes_per_iteration = 50
num_epochs = 5
num_iter = 50

wrapper = GameEnvWrapper(poly_degree, max_actions)
policy = SimplePolicy.Policy(36, 64, 5, 4)

out = rollout(wrapper, policy)

# envs, actions = out
# isactive = [all_active_quads(w) for w in envs]

# fwrap = deepcopy(envs[end-2])
# action = actions[end-1]
# q, e, t = index_to_action(action)

# mesh = fwrap.env.mesh

# smooth_mesh!(fwrap)
# plot_env(fwrap, elem_numbers=true, internal_order=true)

# PPO.step!(env, action)
