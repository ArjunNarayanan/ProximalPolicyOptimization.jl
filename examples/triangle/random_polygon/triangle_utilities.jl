using Flux
using TriMeshGame
using MeshPlotter
using ProximalPolicyOptimization
using Distributions: Categorical

TM = TriMeshGame
PPO = ProximalPolicyOptimization
MP = MeshPlotter

const HALF_EDGES_PER_ELEMENT = 3
const ACTIONS_PER_EDGE = 3

struct StateData
    vertex_score::Any
    action_mask
end

function Base.show(io::IO, s::StateData)
    na = size(s.vertex_score, 2)
    println(io, "StateData")
    println(io, "\tNum Actions : $na")
end

function pad_vertex_scores(vertex_scores_vector)
    num_half_edges = [size(vs, 2) for vs in vertex_scores_vector]
    max_num_half_edges = maximum(num_half_edges)
    num_new_cols = max_num_half_edges .- num_half_edges
    padded_vertex_scores = [TM.zero_pad(vs, nc) for (vs, nc) in zip(vertex_scores_vector, num_new_cols)]
    return padded_vertex_scores
end

function pad_action_mask(action_mask_vector)
    num_actions = length.(action_mask_vector)
    max_num_actions = maximum(num_actions)
    num_new_actions = max_num_actions .- num_actions
    padded_action_mask = [TM.pad(am, nr, -Inf32) for (am, nr) in zip(action_mask_vector, num_new_actions)]
    return padded_action_mask
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

function get_action_mask(active_triangle)
    actions_per_triangle = HALF_EDGES_PER_ELEMENT * ACTIONS_PER_EDGE

    requires_mask = repeat(.!active_triangle', inner=(actions_per_triangle, 1))
    mask = vec([r ? -Inf32 : 0.0f0 for r in requires_mask])
    return mask
end

function make_template(mesh)
    pairs = TM.make_edge_pairs(mesh)

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
    vd = val_or_missing(env.mesh.degrees, template, 0)
    matrix = vcat(vs, vd)

    am = get_action_mask(env.mesh.active_triangle)

    s = StateData(matrix, am)

    return s
end

function PPO.action_probabilities(policy, state)
    vertex_score, action_mask = state.vertex_score, state.action_mask

    logits = vec(policy(vertex_score)) + action_mask
    p = softmax(logits)
    return p
end

function PPO.reward(wrapper)
    return wrapper.env.reward
end

function PPO.is_terminal(wrapper)
    return wrapper.env.is_terminated
end

function index_to_action(index)
    actions_per_triangle = HALF_EDGES_PER_ELEMENT * ACTIONS_PER_EDGE

    triangle_index = div(index - 1, actions_per_triangle) + 1

    triangle_action_idx = rem(index - 1, actions_per_triangle)
    half_edge_index = div(triangle_action_idx, ACTIONS_PER_EDGE) + 1
    action_type = rem(triangle_action_idx, ACTIONS_PER_EDGE) + 1

    return triangle_index, half_edge_index, action_type
end


function step_wrapper!(wrapper, triangle_index, half_edge_index, action_type, no_action_reward=-4)
    env = wrapper.env

    @assert TM.is_active_triangle(env.mesh, triangle_index) "Attempting to act on inactive triangle $triangle_index with action ($triangle_index, $half_edge_index, $action_type)"
    @assert action_type in 1:ACTIONS_PER_EDGE "Expected action type in 1:$ACTIONS_PER_EDGE, got type = $action_type"
    @assert half_edge_index in 1:HALF_EDGES_PER_ELEMENT "Expected edge in 1:$HALF_EDGES_PER_ELEMENT, got edge = $half_edge_index"

    TM.step!(env, triangle_index, half_edge_index, action_type, no_action_reward = no_action_reward)
end

function action_space_size(env)
    nt = TM.triangle_buffer(env.mesh)
    return nt * HALF_EDGES_PER_ELEMENT * ACTIONS_PER_EDGE
end

function PPO.step!(wrapper, linear_action_index; no_action_reward = -4)
    @assert 1 <= linear_action_index <= action_space_size(wrapper.env)

    triangle_index, half_edge_index, action_type = index_to_action(linear_action_index)
    step_wrapper!(wrapper, triangle_index, half_edge_index, action_type, no_action_reward)
end












#####################################################################################################################
# PLOTTING STUFF
function plot_env(_env)
    env = deepcopy(_env)
    TM.reindex!(env)

    mesh = env.mesh
    fig = MP.plot_mesh(TM.active_vertex_coordinates(mesh), TM.active_triangle_connectivity(mesh),
    vertex_score = TM.active_vertex_score(env), vertex_size = 20, fontsize = 15)

    return fig
end

function plot_wrapper(wrapper)
    fig = plot_env(wrapper.env)
    return fig
end
#####################################################################################################################