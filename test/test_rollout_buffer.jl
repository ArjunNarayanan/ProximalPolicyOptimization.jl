using ProximalPolicyOptimization
PPO = ProximalPolicyOptimization

mutable struct TestEnv 
    num_steps
    max_steps
    function TestEnv(max_steps)
        return new(0, max_steps)
    end
end

function PPO.state(env::TestEnv)
    return rand(3,3)
end

function PPO.action_probabilities(policy, state)
    return [1.,0.,0.]
end

function PPO.step!(env, a)
    env.num_steps += 1
end


function PPO.reward(env)
    return 1.0
end

function PPO.is_terminal(env)
    if env.num_steps >= env.max_steps
        return true
    else
        return false
    end
end

function PPO.reset!(env)
    env.num_steps = 0
end

env = TestEnv(10)
policy = nothing
rollouts = PPO.RolloutBuffer()
PPO.collect_rollouts!(
    rollouts,
    env,
    policy,
    10,
    1.0
)