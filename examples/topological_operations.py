import numpy as np
from src.tiler import Tiler
from src.render import Renderer
import os


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


graph = initialize_graph()
renderer = Renderer(
    graph,
    coords=graph.vertex_coordinates,
    label_vertices=True,
    # label_halfedge=True,
    vertex_size=40,
    fontsize=35,
    figsize=9
)
renderer.plot()
output_dir = "examples/figures/operations"

figname = "initial.pdf"
outputfile = os.path.join(output_dir, figname)
renderer.fig.savefig(outputfile)

graph.insert_half_edge(0, 2)
renderer.plot()
figname = "insert-edge.pdf"
outputfile = os.path.join(output_dir, figname)
renderer.fig.savefig(outputfile)

graph.insert_vertex(6)
renderer.plot()
figname = "insert-vertex.pdf"
outputfile = os.path.join(output_dir, figname)
renderer.fig.savefig(outputfile)

graph.insert_half_edge(7, 1)
renderer.plot()
figname = "insert-edge2.pdf"
outputfile = os.path.join(output_dir, figname)
renderer.fig.savefig(outputfile)

graph.delete_half_edge(6)
renderer.plot()
figname = "delete-edge.pdf"
outputfile = os.path.join(output_dir, figname)
renderer.fig.savefig(outputfile)

graph.delete_source_vertex(11)
renderer.plot()
figname = "delete-vertex.pdf"
outputfile = os.path.join(output_dir, figname)
renderer.fig.savefig(outputfile)
