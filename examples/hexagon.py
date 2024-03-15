import numpy as np
from src.tiler import Tiler
from src.render import Renderer


################################################################################################################
def vertex_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    # vertex coordinates are expected to be provided as a dictionary mapping vertex labels
    # to their coordinate values
    coords = dict(zip(range(6), coords))
    return coords


coords = vertex_coordinates()
mesh = Tiler.from_face_loops(
    [[0, 1, 2, 3, 4, 5]],
    vertex_coordinates=coords
)
renderer = Renderer(
    mesh,
    coords=mesh.vertex_coordinates,
    label_vertices=True,
)
renderer.plot()
# renderer.fig.savefig("examples/figures/hexagon.png")
################################################################################################################


################################################################################################################
coords[6] = coords[0] - np.array([0, 1])
coords[7] = coords[1] - np.array([0, 1])
coords[8] = np.array([1.5, np.sin(np.pi / 3)])
mesh = Tiler.from_face_loops(
    [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 1, 0],
        [3, 2, 8]
    ],
    vertex_coordinates=coords,
)
renderer = Renderer(
    mesh,
    coords=mesh.vertex_coordinates,
    label_vertices=True,
)
renderer.plot()
# renderer.fig.savefig("examples/figures/mixed.png")
################################################################################################################


################################################################################################################
renderer = Renderer(
    mesh,
    coords=mesh.vertex_coordinates,
    label_vertices=True,
    label_halfedge=True
)
renderer.plot()
renderer.fig.savefig("examples/figures/mixed-halfedge.png")
################################################################################################################
