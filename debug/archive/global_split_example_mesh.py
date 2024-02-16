import pickle
from src.tiler import Tiler
from src.render import Renderer
from src.quad_global_splitter import QuadGlobalSplitter


filename = "experiments/tiler-random-polygon/quad/models/convolution/best-mesh/rollout-6/best_mesh.pkl"
with open(filename, "rb") as input_file:
    data = pickle.load(input_file)

graph = data["best_env"].graph
splitter = QuadGlobalSplitter(graph, max_aspect_ratio=0.8)

renderer = Renderer(graph, graph.vertex_coordinates)
renderer.plot()

# splitter.update_half_edge_aspect_ratios()
splitter.global_split_loop(iterations=10, smooth=5)
renderer.plot()