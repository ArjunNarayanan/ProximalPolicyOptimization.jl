using Test
include("quad_game_utilities.jl")
include("useful_routines.jl")

actions = index_to_action.(1:120)
q = [a[1] for a in actions]
e = [a[2] for a in actions]
a = [a[3] for a in actions]

test_q = repeat(1:6, inner=20)
@test allequal(test_q, q)

test_e = repeat(1:4, inner=5, outer=6)
@test allequal(test_e, e)

test_a = repeat(1:5, outer=24)
@test allequal(test_a, a)


mesh0 = QM.square_mesh(2)
d0 = deepcopy(mesh0.degree)
QM.left_flip!(mesh0, 1, 3)
wrapper = GameEnvWrapper(mesh0, d0, 4)

episodes = PPO.EpisodeData(PPO.initialize_state_data(wrapper))
policy = SimplePolicy.Policy(36, 64, 5)

PPO.collect_episode_data!(episodes, wrapper, policy)