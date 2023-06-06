using Revise
using Test
using ProximalPolicyOptimization
using BSON
PPO = ProximalPolicyOptimization
include("useful_routines.jl")

dir = "~/.julia/dev/ProximalPolicyOptimization/examples/rollout_to_disk/"
file = joinpath(dir, "test.txt")
cmd = `echo hello '>' $file`
run(cmd)

trajectory = PPO.Rollouts(dir)
@test !isfile(file)
@test isdir(joinpath(dir, "states"))

PPO.update!(trajectory, [1,2,3,4,5], 0.2, 1, 0.5, false)

file_path = joinpath(dir, "states", "sample_1.bson")
@test isfile(file_path)

state = BSON.load(file_path)[:state]
@test allequal(state, [1,2,3,4,5])

