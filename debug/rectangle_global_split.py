from src.render import Renderer
from src.tiler import Tiler
from src.quad_global_splitter import QuadGlobalSplitter
import os


def initialize_graph():
    coords = generate_coordinates()
    faces = [
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7]
    ]
    graph = Tiler.from_face_loops(faces, vertex_coordinates=coords)
    graph.user_defined_vertices.discard(4)
    return graph


def generate_coordinates():
    coords = [
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 0.25],
        [1, 0.25],
        [2, 0.25],
        [0, 3],
        [1, 3],
        [2, 3]
    ]
    coords = dict(zip(range(len(coords)), coords))
    return coords


graph = initialize_graph()
renderer = Renderer(graph, graph.vertex_coordinates, label_vertices=True)
renderer.plot()
output_file = os.path.join("experiments", "median-splits", "figures", "initial.png")
# renderer.save_figure()

splitter = QuadGlobalSplitter(graph, max_aspect_ratio=1.2)
splitter.global_split_loop(iterations=10, smooth=0)

renderer = Renderer(graph, graph.vertex_coordinates, label_vertices=True)
renderer.plot()
