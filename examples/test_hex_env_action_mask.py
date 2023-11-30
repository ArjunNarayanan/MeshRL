from src.render import Renderer
from envs.hex_env_with_insert import HexEnv

env = HexEnv(
    template_size=10,
    max_actions=10,
    randomize=False,
    incremental_reward=True,
    no_action_reward=0
)

renderer = Renderer(env.graph, env.graph.vertex_coordinates)
renderer.plot()

env._step_insert_edge(1, 1)
renderer.plot()

env._step_insert_vertex(6)
renderer.plot()

env._step_insert_edge(7,2)
renderer.plot()

env._step_delete_edge(6)
renderer.plot()

total_num_actions = env.template_size * env.num_actions_per_halfedge
# action_mask = [env.is_valid_action(idx) for idx in range(total_num_actions)]
