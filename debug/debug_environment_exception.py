import pickle
import os
import sys

sys.path.append(os.getcwd())
from src.tiler import Tiler
from src.render import Renderer
from envs.angle_env import AngleEnv
from src.utils import load_yaml_config

if __name__ == "__main__":
    filename = "../debug/experiments/except-env-4.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)

    graph = data["graph"]
    env = AngleEnv(3, [5])
    env.graph = graph
    env._update_scores_on_reset()
    env.template_center = env._select_half_edge_template_center(env.graph.half_edge_list())
    env._build_template()

    actions = data["actions"]
    # renderer = Renderer(env.graph, env.graph.vertex_coordinates, label_halfedge=True)
    # renderer.plot()

    for idx, data in enumerate(actions):
        # renderer = Renderer(env.graph, env.graph.vertex_coordinates, label_halfedge=True)
        # renderer.plot()

        hidx = data[0]
        action = data[1]
        print("idx : ", idx, "\thidx : ", hidx, "\taction : ", action)
        env._step_half_edge_action(hidx, action)
        env.num_steps += 1

        env._update_half_edge_template_center()
        print("Template center : ", env.template_center)
        env._build_template()
        env.terminated = env.is_terminated()
        observation = env._get_obs()
        # reward = env._get_reward()

    # renderer = Renderer(env.graph, env.graph.vertex_coordinates)
    # renderer.plot()
