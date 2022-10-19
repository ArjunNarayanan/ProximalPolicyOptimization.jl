include("quad_game_utilities.jl")
include("square_mesh_env.jl")


wrapper = SquareMeshWrapper(3, 5, 10)

smooth_wrapper!(wrapper)
fig = plot_wrapper(wrapper, elem_numbers=true, internal_order=true)
fig.savefig("output/initial-state.png")

mesh = QM.square_mesh(3, vertex_buffer=20, quad_buffer=20)
fig, ax = PQ.plot_mesh(QM.active_vertex_coordinates(mesh), QM.active_quad_connectivity(mesh),
elem_numbers=true, internal_order=true, node_numbers=true)
fig.savefig("output/initial_mesh.png")