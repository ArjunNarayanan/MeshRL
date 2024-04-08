import argparse
import os
import sys

sys.path.append(os.getcwd())
from src.render import Renderer
import pickle


def plot_graph(
        graph,
        face_desired_degree,
        vertex_desired_degree=None,
        filename=None,
        vertex_size=30,
        fontsize=18,
        figsize=12
):
    renderer = Renderer(graph, graph.vertex_coordinates, vertex_size=vertex_size, fontsize=fontsize, figsize=figsize)
    renderer.coords = graph.vertex_coordinates
    renderer.plot()
    if vertex_desired_degree is not None:
        renderer.plot_vertex_scores(vertex_desired_degree)
    renderer.plot_face_scores(face_desired_degree)
    if filename is not None:
        renderer.fig.tight_layout()
        renderer.fig.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="path to pickle data file", required=True)
    parser.add_argument("-faces", required=True, help="Face degree of polygon", type=int)
    parser.add_argument("-vertex_size", default=20, type=int)
    parser.add_argument("-fontsize", default=14, type=int)
    args = parser.parse_args()

    vertex_size = args.vertex_size
    fontsize = args.fontsize
    filename = args.input
    with open(filename, "rb") as input_file:
        data = pickle.load(input_file)

    input_folder = os.path.dirname(filename)

    initial_env = data["initial"]
    best_env = data["best_env"]
    best_graph = best_env.graph
    vertex_desired_degree = best_env.vertex_desired_degree

    outputfile = os.path.join(input_folder, "initial.pdf")
    print("\nPLOTTING INITIAL POLYGON : ", outputfile)
    plot_graph(
        initial_env.graph,
        args.faces,
        vertex_desired_degree=initial_env.vertex_desired_degree,
        filename=outputfile,
        vertex_size=vertex_size,
        fontsize=fontsize
    )

    outputfile = os.path.join(input_folder, "coarse.pdf")
    print("\nPLOTTING COARSE MESH : ", outputfile)
    plot_graph(
        best_graph,
        args.faces,
        vertex_desired_degree=vertex_desired_degree,
        filename=outputfile,
        vertex_size=vertex_size,
        fontsize=fontsize
    )
