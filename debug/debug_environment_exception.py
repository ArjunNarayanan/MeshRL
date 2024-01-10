import pickle
import os
import sys

sys.path.append(os.getcwd())
from src.tiler import Tiler
from src.render import Renderer
from envs.random_polygon_tiler_env import RandomPolygonEnv
from src.utils import load_yaml_config

# config_fn = "experiments/random-polygon/tri-poly-10-optuna/config.yml"
# config = load_yaml_config(config_fn)
# env = RandomPolygonEnv.from_config(config["environment"])
# env._step_insert_vertex(None)


filename = "debug/except-env-1.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)

graph = data["graph"]

env = RandomPolygonEnv(3, [6])
env.graph = graph

actions = data["actions"]

idx = 0

print("Index : ", idx)
action = actions[idx]
print("Action : ", action, "\n")
halfedge, action_type = action
env._step_half_edge_action(halfedge, action_type)
env._update_half_edge_template_center()
env._build_template()
terminated = env.is_terminated()
observation = env._get_obs()
idx += 1

graph = env.graph