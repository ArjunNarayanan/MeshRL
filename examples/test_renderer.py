from envs.hex_env import HexEnv
from src.render import Renderer
import numpy as np
import matplotlib.pyplot as plt


def generate_coordinates():
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    coords = [[-c, -s],
              [c, -s],
              [1, 0],
              [c, s],
              [-c, s],
              [-1, 0]]
    coords = dict(zip(range(6), coords))
    return coords


env = HexEnv()
coords = generate_coordinates()
renderer = Renderer(env.graph, coords)
renderer.plot()
# halfedge = env.graph.first_face_halfedge(0)
# face_loop = env.graph.generate_halfedge_face_loop(halfedge)
# face_vertices = [env.graph.source_vertex(hidx, tag=False) for hidx in face_loop]
#
# face_coords = [coords[v] for v in face_vertices]
# x = [c[0] for c in face_coords]
# y = [c[1] for c in face_coords]
#
# fig, ax = plt.subplots()
# ax.axis("equal")
# ax.axis("off")
# ax.fill(x, y, facecolor="lightgray", edgecolor="black", linewidth=2)
