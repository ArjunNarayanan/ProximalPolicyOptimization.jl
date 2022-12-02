include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")


polygon_degree = 10
hmax = 0.2
max_actions = 50
episodes_per_iteration = 10

wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)
policy = Policy(24, 32, 2, ACTIONS_PER_EDGE)

rollouts = PPO.EpisodeData()
PPO.collect_rollouts!(rollouts, wrapper, policy, episodes_per_iteration)

PPO.shuffle!(rollouts)

vertex_score = [s.vertex_score for s in rollouts.state_data]
action_mask = [s.action_mask for s in rollouts.state_data]

padded_vertex_scores = pad_vertex_scores(vertex_score)
padded_action_mask = pad_action_mask(action_mask)

num_half_edges = [size(vs, 2) for vs in vertex_score]
max_num_half_edges = maximum(num_half_edges)
num_new_cols = max_num_half_edges .- num_half_edges
vertex_score = [TM.zero_pad(vs, nc) for (vs, nc) in zip(vertex_score, num_new_cols)]
num_half_edges = [size(vs, 2) for vs in vertex_score]