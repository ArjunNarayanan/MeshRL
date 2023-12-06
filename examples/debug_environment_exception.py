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

for idx, action in enumerate(actions[:-1]):
    print("Index : ", idx)
    halfedge, action_type = action
    env._step_halfedge_action(halfedge, action_type)
    env._update_halfedge_template_center()
    env._build_template()
    terminated = env.is_terminated()
    observation = env._get_obs()
    renderer.plot()

graph = env.graph

# graph.insert_halfedge(2,1)
# renderer.plot()
#
# graph.insert_halfedge(7,2)
# renderer.plot()
#
# graph.insert_vertex(8)
# renderer.plot()
#
# graph.insert_vertex(8)
# renderer.plot()
#
# graph.insert_halfedge(5, 2)
# renderer.plot()
#
# graph.insert_vertex(13)
# renderer.plot()
#
# graph.insert_vertex(5)
# renderer.plot()
#
# graph.insert_halfedge(10, 1)
# renderer.plot()
#
# graph.insert_vertex(12)
# renderer.plot()
#
# graph.insert_halfedge(13,1)
# renderer.plot()
#
# graph.delete_halfedge(21)
# renderer.plot()
#
# graph.delete_halfedge(8)
# renderer.plot()
#
# # graph.insert_vertex(0)
# # graph.insert_vertex(7)
# # graph.insert_halfedge(10, 2)
# # graph.insert_vertex(6)
# # graph.insert_halfedge(13,2)
# # graph.delete_halfedge(13)
# # graph.insert_vertex(3)
# # graph.insert_halfedge(15, 2)
#
#
