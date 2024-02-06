import os
from src.render import Renderer
from src.tiler import refine, Tiler
import pickle


def plot_graph(graph, face_desired_degree, vertex_desired_degree=None, filename=None):
    renderer = Renderer(graph, graph.vertex_coordinates)
    renderer.coords = graph.vertex_coordinates
    renderer.plot()
    if vertex_desired_degree is not None:
        renderer.plot_vertex_scores(vertex_desired_degree)
    renderer.plot_face_scores(face_desired_degree)
    if filename is not None:
        renderer.fig.savefig(filename)


filename = "experiments/tiler-random-polygon/triangle/tri-5-50-scaled/best-mesh/rollout-2/best_mesh.pkl"
with open(filename, "rb") as input_file:
    data = pickle.load(input_file)

input_folder = os.path.dirname(filename)

initial_env = data["initial"]
best_env = data["best_env"]
best_graph = best_env.graph
vertex_desired_degree = best_env.vertex_desired_degree

outputfile = os.path.join(input_folder, "initial.png")
plot_graph(initial_env.graph, 3, vertex_desired_degree=initial_env.vertex_desired_degree, filename=outputfile)

outputfile = os.path.join(input_folder, "coarse.png")
plot_graph(best_graph, 3, vertex_desired_degree=vertex_desired_degree, filename=outputfile)

refined_graph = refine(best_graph)
refined_graph.smooth_vertices()
outputfile = os.path.join(input_folder, "refine-1.png")
plot_graph(refined_graph, 3, vertex_desired_degree=vertex_desired_degree, filename=outputfile)

refined_graph = refine(refined_graph)
refined_graph.smooth_vertices()
outputfile = os.path.join(input_folder, "refine-2.png")
plot_graph(refined_graph, 3, vertex_desired_degree=vertex_desired_degree, filename=outputfile)



refined_graph = refine(refined_graph)
refined_graph.smooth_vertices()
# outputfile = os.path.join(input_folder, "refine-2.png")
plot_graph(refined_graph, 3, vertex_desired_degree=vertex_desired_degree)
