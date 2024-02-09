import argparse
import os
import sys

sys.path.append(os.getcwd())
from src.render import Renderer
from src.tiler import refine
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="path to pickle data file", required=True)
    parser.add_argument("-faces", required=True, help="Face degree of polygon", type=int)
    parser.add_argument("-refine", default=3, type=int, help="number of refinement levels")
    args = parser.parse_args()

    filename = args.input
    with open(filename, "rb") as input_file:
        data = pickle.load(input_file)

    input_folder = os.path.dirname(filename)

    initial_env = data["initial"]
    best_env = data["best_env"]
    best_graph = best_env.graph
    vertex_desired_degree = best_env.vertex_desired_degree

    outputfile = os.path.join(input_folder, "initial.png")
    print("\nPLOTTING INITIAL POLYGON : ", outputfile)
    plot_graph(
        initial_env.graph,
        args.faces,
        vertex_desired_degree=initial_env.vertex_desired_degree,
        filename=outputfile
    )

    outputfile = os.path.join(input_folder, "coarse.png")
    print("\nPLOTTING COARSE MESH : ", outputfile)
    plot_graph(best_graph, args.faces, vertex_desired_degree=vertex_desired_degree, filename=outputfile)

    refined_graph = best_graph

    for refinement_level in range(args.refine):
        refined_graph = refine(refined_graph, args.faces)
        refined_graph.smooth_vertices()
        outputfile = os.path.join(input_folder, "refine-" + str(refinement_level).zfill(2) + ".png")
        print("\nPLOTTING REFINE MESH : ", outputfile)
        plot_graph(
            refined_graph,
            args.faces,
            vertex_desired_degree=vertex_desired_degree,
            filename=outputfile
        )
