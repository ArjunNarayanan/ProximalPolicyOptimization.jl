using LinearAlgebra
using TriMeshGame
using RandomQuadMesh
using ProximalPolicyOptimization

TM = TriMeshGame
RQ = RandomQuadMesh
PPO = ProximalPolicyOptimization

function enclosed_angle(v1,v2)
    @assert length(v1) == length(v2) == 2
    dotp = dot(v1,v2)
    detp = v1[1]*v2[2] - v1[2]*v2[1]
    rad = atan(detp, dotp)
    if rad < 0
        rad += 2pi
    end

    return rad2deg(rad) 
end

function get_polygon_interior_angles(p)
    n = size(p,2)
    angles = zeros(n)
    for i = 1:n
        previ = i == 1 ? n : i -1
        nexti = i == n ? 1 : i + 1

        v1 = p[:,nexti] - p[:,i]
        v2 = p[:,previ] - p[:,i]
        angles[i] = enclosed_angle(v1,v2)
    end
    return angles
end

function get_desired_degree(angle)
    ndiv = 1
    err = abs(angle - 60)
    while err > abs(angle/(ndiv+1) - 60)
        ndiv += 1
        err = abs(angle/ndiv - 60) 
    end
    return ndiv+1
end

function generate_random_mesh_and_desired_degree(polygon_degree)
    boundary_pts = RQ.random_polygon(polygon_degree)
    mesh = RQ.tri_mesh(boundary_pts)
    mesh = TM.Mesh(mesh.p, mesh.t)

    desired_degree = get_desired_degree.(get_polygon_interior_angles(boundary_pts))            
    return mesh, desired_degree
end

struct RandPolyEnv
    polygon_degree
    env
    max_actions
    function RandPolyEnv(polygon_degree, max_actions)
        mesh, desired_degree = generate_random_mesh_and_desired_degree(polygon_degree)
        env = TM.GameEnv(mesh, desired_degree, max_actions)
        new(polygon_degree, env, max_actions)
    end
end

function PPO.reset!(wrapper)
    mesh, desired_degree = generate_random_mesh_and_desired_degree(wrapper.polygon_degree)
    env = TM.GameEnv(mesh, desired_degree, wrapper.max_actions)
end