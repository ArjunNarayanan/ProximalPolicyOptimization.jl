module ProximalPolicyOptimization

function not_implemented(name)
    error("Function $name needs to be overloaded")
end

function state(env) not_implemented("state") end
function reward(env) not_implemented("reward") end
function is_terminal(env) not_implemented("is_terminal") end
function reset!(env) not_implemented("reset!") end
function step!(env, action) not_implemented("step!") end

function initialize_state_data(env) not_implemented("initialize_state_data") end
function update!(state_data, state) not_implemented("update!") end
function action_probabilities(policy, state) not_implemented("action_probabilities") end
function batch_action_probabilities(policy, state) not_implemented("batch_action_probabilities") end
function episode_state(state_data) not_implemented("episode_state") end
function episode_returns(rewards, state_data, discount) not_implemented("episode_returns") end
function batch_state(state_data) not_implemented("batch_state") end
function batch_advantage(episodes) not_implemented("batch_advantage") end

end
