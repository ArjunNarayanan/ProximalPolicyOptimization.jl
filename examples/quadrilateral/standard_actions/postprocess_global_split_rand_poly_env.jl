function initialize_random_mesh(poly_degree, quad_alg)
    boundary_pts = RQ.random_polygon(poly_degree)
    angles = QM.polygon_interior_angles(boundary_pts)
    bdry_d0 = QM.desired_degree.(angles)

    mesh = RQ.quad_mesh(boundary_pts, algorithm=quad_alg)
    mesh = QM.QuadMesh(mesh.p, mesh.t, mesh.t2t, mesh.t2n)

    mask = .![trues(poly_degree); falses(mesh.num_vertices - poly_degree)]
    mask = mask .& mesh.vertex_on_boundary[mesh.active_vertex]

    d0 = [bdry_d0; fill(4, mesh.num_vertices - poly_degree)]
    d0[mask] .= 3

    return mesh, d0
end

function can_global_split(v1, v2, vertex_score, mesh)
    if !QM.vertex_on_boundary(mesh, v2) &&
       vertex_score[v1] == +1 &&
       vertex_score[v2] == -1

        return true
    else
        return false
    end
end

function update_vertex_score_for_global_split!(vertex_score, mesh)
    num_quads = QM.quad_buffer(mesh)

    for quad_idx in 1:num_quads
        if QM.is_active_quad(mesh, quad_idx)
            for vertex_idx in 1:4

                v1 = QM.vertex(mesh, quad_idx, vertex_idx)
                v2 = QM.vertex(mesh, quad_idx, QM.next(vertex_idx))

                if can_global_split(v1, v2, vertex_score, mesh)
                    vertex_score[v1] = 0
                    vertex_score[v2] = 0
                end
            end
        end
    end
end

function global_score(_vertex_score, mesh)
    vertex_score = copy(_vertex_score)
    update_vertex_score_for_global_split!(vertex_score, mesh)
    score = sum(abs.(vertex_score))
    return score
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
        mesh, d0 = initialize_random_mesh(poly_degree, quad_alg)
        env = QM.GameEnv(mesh, d0)
        score = global_score(env.vertex_score, env.mesh)
        reward = 0
        new(poly_degree, quad_alg, max_actions, env, score, reward)
    end
end

function Base.show(io::IO, wrapper::RandPolyEnv)
    println(io, "GameEnvWrapper")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper)
    mesh, d0 = initialize_random_mesh(wrapper.poly_degree, wrapper.quad_alg)
    wrapper.env = QM.GameEnv(mesh, d0, wrapper.max_actions)
    wrapper.current_score = global_score(wrapper.env.vertex_score, wrapper.env.mesh)
    wrapper.reward = 0
end

function update_env_score_after_step!(wrapper)
    wrapper.current_score = global_score(wrapper.env.vertex_score, wrapper.env.mesh)
end
