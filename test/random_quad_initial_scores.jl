include("quad_game_utilities.jl")


function get_initial_opt_scores(npts)
    wrapper = GameEnvWrapper(npts, 1)
    cs = wrapper.env.current_score
    os = wrapper.env.opt_score
    return cs, os
end


npts = 10
scores = [get_initial_opt_scores(10) for idx in 1:1000]
initial_score = first.(scores)
opt_score = last.(scores)

using PyPlot
fig, ax = subplots()
ax.hist(initial_score, alpha = 0.8, label = "initial")
ax.hist(opt_score, alpha = 0.8, label = "optimum")
ax.legend()
ax.grid()
ax.set_title("Distribution of initial / optimal scores for degree 10 polygon")
fig.tight_layout()
fig.savefig("output/initial_score_dist_poly10.png")