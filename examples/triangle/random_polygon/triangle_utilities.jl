using Flux
using TriMeshGame
using MeshPlotter
using ProximalPolicyOptimization
using Distributions: Categorical


TM = TriMeshGame
PPO = ProximalPolicyOptimization
MP = MeshPlotter

struct StateData
    vertex_score::Any
    action_mask
end

function Base.show(io::IO, s::StateData)
    na = size(s.vertex_score, 2)
    println(io, "StateData")
    println(io, "\tNum Actions : $na")
end

function PPO.batch_state(state_data_vector)
    vs = [s.vertex_score for s in state_data_vector]
    am = [s.action_mask for s in state_data_vector]

    batch_vertex_score = cat(vs..., dims=3)
    batch_action_mask = cat(am..., dims=2)
    return StateData(batch_vertex_score, batch_action_mask)
end

function val_or_missing(vector, template, missing_val)
    return [t == 0 ? missing_val : vector[t] for t in template]
end

function make_action_mask(active_triangle; actions_per_edge=3)
    actions_per_triangle = 3 * actions_per_edge
    requires_mask = repeat(.!active_triangle', inner=(actions_per_triangle, 1))
    mask = vec([r ? -Inf32 : 0.0f0 for r in requires_mask])
    return mask
end

function make_template(mesh)
    pairs = make_edge_pairs(mesh)

    tri_vertices = reshape(mesh.connectivity, 1, :)

    ct = TM.cycle_edges(tri_vertices)

    p = TM.zero_pad(tri_vertices)[:, pairs]
    cp1 = TM.cycle_edges(p)

    p = TM.zero_pad(cp1)[[2, 3], pairs]
    cp2 = TM.cycle_edges(p)

    template = vcat(ct, cp1, cp2)

    return template
end

function PPO.state(wrapper)
    env = wrapper.env
    template = make_template(env.mesh)

    vs = val_or_missing(env.vertex_score, template, 0)
    vd = val_or_missing(env.mesh.degree, template, 0)
    matrix = vcat(vs, vd)

    am = action_mask(env.mesh.active_triangle)

    s = StateData(matrix, am)

    return s
end