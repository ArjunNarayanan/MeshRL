import pickle
import os
import sys
sys.path.append(os.getcwd())
from src.polygraph import PolyGraph
from src.render import Renderer
from envs.random_polygon_env import RandomPolygonEnv
from src.utils import load_yaml_config



# config_fn = "experiments/random-polygon/tri-poly-10-optuna/config.yml"
# config = load_yaml_config(config_fn)
# env = RandomPolygonEnv.from_config(config["environment"])
# env._step_insert_vertex(None)






filename = "experiments/random-polygon/tri-poly-10-optuna/except-env-1.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)

# renderer = Renderer(env.graph, env.graph.vertex_coordinates)
# renderer.plot()


actions = data["actions"]

# for idx, action in enumerate(actions[:-2]):
#     print("Index : ", idx)
#     halfedge, action_type = action
#     env._step_halfedge_action(halfedge, action_type)
#     env._update_halfedge_template_center()
#     env._build_template()
#     terminated = env.is_terminated()
#     observation = env._get_obs()
#     renderer.plot()
#
# graph = env.graph
