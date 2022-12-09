function initialize_random_mesh(poly_degree, quad_alg)
    boundary_pts = RQ.random_polygon(poly_degree)
    angles = QM.polygon_interior_angles(boundary_pts)
    bdry_d0 = QM.desired_degree.(angles)

    mesh = RQ.quad_mesh(boundary_pts, algorithm = quad_alg)
    mesh = QM.QuadMesh(mesh.p, mesh.t, mesh.t2t, mesh.t2n)

    mask = .![trues(poly_degree); falses(mesh.num_vertices - poly_degree)]
    mask = mask .& mesh.vertex_on_boundary[mesh.active_vertex]

    d0 = [bdry_d0; fill(4, mesh.num_vertices - poly_degree)]
    d0[mask] .= 3

    return mesh, d0
end

mutable struct RandPolyEnv
    poly_degree
    quad_alg
    max_actions::Any
    env::Any
    function RandPolyEnv(poly_degree, max_actions; quad_alg = "matching")
        mesh, d0 = initialize_random_mesh(poly_degree, quad_alg)
        env = QM.GameEnv(deepcopy(mesh), deepcopy(d0), max_actions)
        new(poly_degree, quad_alg, max_actions, env)
    end
end

function Base.show(io::IO, wrapper::RandPolyEnv)
    println(io, "GameEnvWrapper")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper)
    mesh, d0 = initialize_random_mesh(wrapper.poly_degree, wrapper.quad_alg)
    wrapper.env = QM.GameEnv(mesh, d0, wrapper.max_actions)
end
