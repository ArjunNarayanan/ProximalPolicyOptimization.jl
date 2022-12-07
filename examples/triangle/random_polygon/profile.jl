include("../environments/randpoly_env.jl")
include("triangle_utilities.jl")
include("../policy.jl")

output_dir = "examples/triangle/random_polygon/output/level4"
polygon_degree = 20
hmax = 0.25
max_actions = 50

epsilon = 0.10
episodes_per_iteration = 20
minibatch_size = 32
num_ppo_iterations = 100
num_evaluation_trajectories = 100


policy = Policy(96, 32, 2, ACTIONS_PER_EDGE)
wrapper = RandPolyWrapper(polygon_degree, hmax, max_actions)

optimizer = Adam(1e-4)
