import numpy as np
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


graph = initialize_graph()
renderer = Renderer(
    graph,
    coords=graph.vertex_coordinates,
    label_vertices=True,
    label_halfedge=True
)
renderer.plot()

figname = "examples/figures/ops-initial.png"
# renderer.fig.savefig(figname)

graph.insert_half_edge(0, 1)
renderer.plot()
figname = "examples/figures/insert-edge.png"
# renderer.fig.savefig(figname)

# graph.insert_vertex(6)
# renderer.plot()
# figname = "examples/figures/insert-vertex.pdf"
# renderer.fig.savefig(figname)
#
# graph.insert_half_edge(7, 1)
# renderer.plot()
# figname = "examples/figures/insert-edge2.pdf"
# renderer.fig.savefig(figname)
#
# graph.delete_half_edge(6)
# renderer.plot()
# figname = "examples/figures/delete-edge.pdf"
# renderer.fig.savefig(figname)
#
# graph.delete_source_vertex(11)
# renderer.plot()
# figname = "examples/figures/delete-vertex.pdf"
# renderer.fig.savefig(figname)
