using ProximalPolicyOptimization
using Tables
using CSV
PPO = ProximalPolicyOptimization

rollouts = PPO.Rollouts("output/")

PPO.update!(rollouts, [1,2,3,4,5], 0.5, 4, 1, true)
PPO.write_returns_to_disk(rollouts, 1.0)