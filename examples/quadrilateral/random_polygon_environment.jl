function initialize_random_mesh(poly_degree, quad_alg)
    boundary_pts = RQ.random_polygon(poly_degree)
    angles = QM.polygon_interior_angles(boundary_pts)
    bdry_d0 = QM.desired_degree.(angles)

    mesh = RQ.quad_mesh(boundary_pts, algorithm=quad_alg)
    mesh = QM.QuadMesh(mesh.p, mesh.t)

    mask = .![trues(poly_degree); falses(mesh.num_vertices - poly_degree)]
    mask = mask .& mesh.vertex_on_boundary[mesh.active_vertex]

    d0 = [bdry_d0; fill(4, mesh.num_vertices - poly_degree)]
    d0[mask] .= 3

    return mesh, d0
end

function _vanilla_global_score(vertex_score)
    return sum(abs.(vertex_score))
end

function _distance_weighted_global_score(vertex_score, distances)
    score = sum(abs.(vertex_score) .* distances)
    return score
end

function global_score(vertex_score)
    return _vanilla_global_score(vertex_score)
end

function optimal_score(vertex_score)
    return abs(sum(vertex_score))
end

function check_terminated(current_score, opt_score, num_actions, max_actions)
    return (num_actions >= max_actions) || (current_score <= opt_score)
end

mutable struct RandPolyEnv
    poly_degree
    quad_alg
    num_actions
    max_actions::Any
    env::Any
    current_score
    opt_score
    is_terminated
    reward
    function RandPolyEnv(poly_degree, max_actions, quad_alg)
        @assert max_actions > 0
        @assert poly_degree > 3

        mesh, d0 = initialize_random_mesh(poly_degree, quad_alg)
        env = QM.GameEnv(mesh, d0)
        current_score = global_score(env.vertex_score)
        opt_score = optimal_score(env.vertex_score)
        reward = 0
        num_actions = 0
        is_terminated = check_terminated(current_score, opt_score, num_actions, max_actions)
        new(poly_degree, quad_alg, num_actions, max_actions, env, current_score, opt_score, is_terminated, reward)
    end
end

function Base.show(io::IO, wrapper::RandPolyEnv)
    println(io, "RandPolyEnv")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper)
    mesh, d0 = initialize_random_mesh(wrapper.poly_degree, wrapper.quad_alg)
    wrapper.env = QM.GameEnv(mesh, d0)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.reward = 0
    wrapper.num_actions = 0
    wrapper.opt_score = optimal_score(wrapper.env.vertex_score)
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score,
        wrapper.num_actions, wrapper.max_actions)
end

function update_env_after_step!(wrapper)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.num_actions += 1
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score, 
        wrapper.num_actions, wrapper.max_actions)
end
