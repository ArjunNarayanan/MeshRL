import numpy as np
import sys
import os
import argparse

sys.path.append(os.getcwd())
from src.tiler import Tiler
from src.render import Renderer


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    return coords


def initialize_graph():
    coords = generate_coordinates()
    coords = dict(zip(range(6), coords))
    graph = Tiler.from_face_loops(
        [[0, 1, 2, 3, 4, 5]],
        vertex_coordinates=coords
    )
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", help="output directory", default="examples/figures/operations")
    parser.add_argument("-ext", help="output file extension", default="png")
    args = parser.parse_args()

    graph = initialize_graph()
    renderer = Renderer(
        graph,
        coords=graph.vertex_coordinates,
        label_vertices=True,
        label_halfedge=True,
        vertex_size=40,
        fontsize=25,
        figsize=9
    )
    renderer.plot()
    output_dir = args.output
    extension = args.ext

    print("\tGenerating initial geometry ...")
    figname = "initial." + extension
    outputfile = os.path.join(output_dir, figname)
    renderer.fig.savefig(outputfile)

    print("\tEdge insert ...")
    graph.insert_half_edge(0, 2)
    renderer.plot()
    figname = "insert-edge." + extension
    outputfile = os.path.join(output_dir, figname)
    renderer.fig.savefig(outputfile)

    figname = "insert-vertex." + extension
    graph.insert_vertex(6)
    renderer.plot()
    print("\tVertex insert ...")
    outputfile = os.path.join(output_dir, figname)
    renderer.fig.savefig(outputfile)

    print("\tEdge insert ...")
    graph.insert_half_edge(7, 1)
    renderer.plot()
    figname = "insert-edge2." + extension
    outputfile = os.path.join(output_dir, figname)
    renderer.fig.savefig(outputfile)

    print("\tEdge delete ...")
    graph.delete_half_edge(6)
    renderer.plot()
    figname = "delete-edge." + extension
    outputfile = os.path.join(output_dir, figname)
    renderer.fig.savefig(outputfile)

    print("\tVertex delete ...")
    graph.delete_source_vertex(11)
    renderer.plot()
    figname = "delete-vertex." + extension
    outputfile = os.path.join(output_dir, figname)
    renderer.fig.savefig(outputfile)
