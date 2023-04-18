module ProximalPolicyOptimization

using Distributions: Categorical
using Flux
using Random
using Printf
using BSON
using DataFrames
using CSV

function not_implemented(name)
    error("Function $name needs to be overloaded")
end

function state(env) not_implemented("state") end
function reward(env) not_implemented("reward") end
function is_terminal(env) not_implemented("is_terminal") end
function reset!(env) not_implemented("reset!") end
function step!(env, action) not_implemented("step!") end


function action_probabilities(policy, state) not_implemented("action_probabilities") end
function batch_action_probabilities(policy, state) not_implemented("batch_action_probabilities") end
function episode_returns(rewards, state_data, discount) not_implemented("episode_returns") end
function prepare_rollouts_for_training(rollouts) return rollouts end
function batch_state(state_data) not_implemented("batch_state") end
function number_of_actions_per_state(state) not_implemented("number_of_actions_per_state") end
function batch_advantage(episodes) not_implemented("batch_advantage") end
function save_loss(evaluator, loss) not_implemented("save_loss") end


# include("rollouts.jl")
include("rollouts_to_disk.jl")
include("train.jl")
include("evaluate.jl")

end
