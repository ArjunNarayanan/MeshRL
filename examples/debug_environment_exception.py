import pickle
from src.polygraph import PolyGraph
from src.render import Renderer
from envs.regular_polygon_env import RegularPolygonEnv
from src.utils import load_yaml_config


filename = "experiments/regular-polygon/poly-20/debug-log/except-env-1.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)

config_fn = "experiments/regular-polygon/poly-20/config.yml"
config = load_yaml_config(config_fn)
env = RegularPolygonEnv.from_config(config["environment"])
renderer = Renderer(env.graph, env.graph.vertex_coordinates)
renderer.plot()

actions = data["actions"]

for idx, action in enumerate(actions[:-2]):
    print("Index : ", idx)
    halfedge, action_type = action
    env._step_halfedge_action(halfedge, action_type)
    env._update_halfedge_template_center()
    env._build_template()
    terminated = env.is_terminated()
    observation = env._get_obs()
    renderer.plot()

graph = env.graph
