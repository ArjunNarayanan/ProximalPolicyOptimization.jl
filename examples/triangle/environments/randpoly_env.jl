using TriMeshGame
using RandomQuadMesh
using ProximalPolicyOptimization

TM = TriMeshGame
RQ = RandomQuadMesh
PPO = ProximalPolicyOptimization

function get_desired_degree(initial_boundary_points, vertex_on_boundary)
    num_verts = length(vertex_on_boundary)
    num_initial_boundary_points = size(initial_boundary_points, 2)
    initial_boundary_points_desired_degree = TM.get_desired_degree.(
        TM.get_polygon_interior_angles(initial_boundary_points)
        )

    initial_vertex_on_boundary = falses(num_verts)
    initial_vertex_on_boundary[1:num_initial_boundary_points] .= true

    additional_vertex_on_boundary = (.!initial_vertex_on_boundary) .& vertex_on_boundary

    desired_degree = fill(6, num_verts)
    desired_degree[1:num_initial_boundary_points] .= initial_boundary_points_desired_degree
    desired_degree[additional_vertex_on_boundary] .= 4

    return desired_degree
end

function generate_random_game_environment(polygon_degree, hmax, max_actions)
    boundary_pts = RQ.random_polygon(polygon_degree)
    mesh = RQ.tri_mesh(boundary_pts, hmax = hmax, allow_vertex_insert = true)
    mesh = TM.Mesh(mesh.p, mesh.t)
    
    vertex_on_boundary = TM.active_vertex_on_boundary(mesh)
    desired_degree = get_desired_degree(boundary_pts, vertex_on_boundary)

    env = TM.GameEnv(mesh, desired_degree, max_actions)

    return env
end

mutable struct RandPolyWrapper
    polygon_degree
    hmax
    max_actions
    env
    function RandPolyWrapper(polygon_degree, hmax, max_actions)
        env = generate_random_game_environment(polygon_degree, hmax, max_actions)
        new(polygon_degree, hmax, max_actions, env)
    end
end

function PPO.reset!(wrapper)
    env = generate_random_game_environment(polygon_degree, hmax, max_actions)
    wrapper.env = env
end

env = generate_random_game_environment(20, 0.3, 50)
