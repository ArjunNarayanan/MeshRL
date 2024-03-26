from envs.environment_initializers import Hexagon
import numpy as np
from src.tiler import Tiler
from src.render import Renderer


################################################################################################################
init = Hexagon()
mesh, _ = init()

renderer = Renderer(
    mesh,
    coords=mesh.vertex_coordinates,
    label_vertices=True,
)
renderer.plot()
# renderer.fig.savefig("examples/figures/hexagon.png")
################################################################################################################


################################################################################################################
coords = Hexagon.generate_coordinates()
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
    # label_halfedge=True,
    vertex_size=30,
    fontsize=20
)
renderer.plot()
renderer.fig.tight_layout()
renderer.fig.savefig("examples/figures/mixed-mesh.pdf")
################################################################################################################
