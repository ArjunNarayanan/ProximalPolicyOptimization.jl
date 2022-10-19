mutable struct SquareMeshWrapper
    mesh0
    num_random_actions
    max_actions::Any
    env::Any
    function SquareMeshWrapper(size, num_random_actions, max_actions)
        vb = 32
        qb = 30
        mesh0 = QM.square_mesh(size, vertex_buffer = vb, quad_buffer = qb)

        mesh = deepcopy(mesh0)
        random_actions!(mesh, num_random_actions)

        new_vertices = mesh.active_vertex .& (.!mesh0.active_vertex)
        desired_degree = deepcopy(mesh0.degree)
        desired_degree[new_vertices] .= 4

        env = QM.GameEnv(mesh, desired_degree[mesh.active_vertex], max_actions)
        new(mesh0, num_random_actions, max_actions, env)
    end
end

function PPO.reset!(wrapper)
    mesh = deepcopy(wrapper.mesh0)
    random_actions!(mesh, wrapper.num_random_actions)

    new_vertices = mesh.active_vertex .& (.!wrapper.mesh0.active_vertex)
    desired_degree = deepcopy(wrapper.mesh0.degree)
    desired_degree[new_vertices] .= 4

    wrapper.env = QM.GameEnv(mesh, desired_degree[mesh.active_vertex], wrapper.max_actions)
end

function Base.show(io::IO, wrapper::SquareMeshWrapper)
    println(io, "SquareMeshEnv")
    show(io, wrapper.env)
end

function step_mesh!(mesh, quad, edge, type)
    flag = false
    if type == 1
        flag = QM.left_flip!(mesh, quad, edge)
    elseif type == 2
        flag = QM.right_flip!(mesh, quad, edge)
    elseif type == 3
        flag = QM.split!(mesh, quad, edge)
    elseif type == 4
        flag = QM.collapse!(mesh, quad, edge)
    else
        error("Expected type = (1,2,3,4), got type = $type")
    end
    return flag
end

function random_action!(mesh)
    quad = rand(1:mesh.new_quad_pointer-1)
    edge = rand(1:4)
    type = rand(1:3)
    success = step_mesh!(mesh, quad, edge, type)
    return success
end

function random_actions!(mesh, num_actions)
    count = 0
    while count < num_actions
        success = random_action!(mesh)
        if success
            count += 1
        end
    end
end