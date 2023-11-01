import matplotlib.pyplot as plt
from src.polygraph import PolyGraph


class Visualizer:
    def __init__(self, graph, coords, vertex_size=20, shrink=0.2):
        self.graph = graph
        self.vertex_size = vertex_size
        self.shrink = shrink
        assert self.graph.num_vertices == len(coords)
        self.coords = coords

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.face_centroids = self._compute_face_centroids(self.graph, self.coords)

    def plot_vertices(self, label=True):
        x = [c[0] for c in self.coords.values()]
        y = [c[1] for c in self.coords.values()]
        self.ax.scatter(x, y, s=self.vertex_size ** 2, color="black")

        if label:
            for idx, coord in self.coords.items():
                x, y = coord
                self.ax.text(x, y, str(idx), color="white", verticalalignment="center", horizontalalignment="center")

    @staticmethod
    def _compute_face_centroids(graph, coords):
        centroids = []
        facenodes = [f for f, data in graph.nodes(data=True) if data.get("type") == "face"]
        for face_id in facenodes:
            halfedges = [target for face, target, data in graph.edges(face_id, data=True) if
                         data.get("type") == "face"]
            source_vertices = [graph.source_vertex(h, tag=False) for h in halfedges]
            vertex_coordinates = [coords[v] for v in source_vertices]
            x_centroid = sum(v[0] for v in vertex_coordinates) / len(vertex_coordinates)
            y_centroid = sum(v[1] for v in vertex_coordinates) / len(vertex_coordinates)
            centroids.append([x_centroid, y_centroid])

        return centroids

    def plot_face_centroids(self, label=True):
        x = [c[0] for c in self.face_centroids]
        y = [c[1] for c in self.face_centroids]
        self.ax.scatter(x, y, s=self.vertex_size ** 2, color="blue")

        if label:
            for idx, coord in enumerate(self.face_centroids):
                x, y = coord
                self.ax.text(x, y, str(idx), color="white", verticalalignment="center", horizontalalignment="center")

    def plot_halfedge(self, idx):
        shrink = self.shrink
        source_vertex = self.graph.source_vertex(idx, tag=False)
        target_vertex = self.graph.target_vertex(idx, tag=False)
        face_id = self.graph.face(idx, tag=False)

        source_coords = self.coords[source_vertex]
        target_coords = self.coords[target_vertex]
        face_coords = self.face_centroids[face_id]

        x = source_coords[0] + shrink * (face_coords[0] - source_coords[0])
        y = source_coords[1] + shrink * (face_coords[1] - source_coords[1])

        dx = target_coords[0] + shrink * (face_coords[0] - target_coords[0]) - x
        dy = target_coords[1] + shrink * (face_coords[1] - target_coords[1]) - y

        self.ax.arrow(x, y, dx, dy, color="red", width=0.02, length_includes_head=True)

        lx = x + dx / 2
        ly = y + dy / 2
        self.ax.scatter(lx, ly, s=self.vertex_size ** 2, color="red")
        self.ax.text(lx, ly, str(idx), color="white", verticalalignment="center", horizontalalignment="center")

    def plot_all_halfedges(self):
        for idx in range(self.graph.num_halfedges):
            self.plot_halfedge(idx)


def initialize_tri_quad_graph():
    face_loops = [
        [0, 1, 4],
        [1, 2, 3, 4]
    ]

    graph = PolyGraph(face_loops)

    return graph


graph = initialize_tri_quad_graph()
coordinates = [[0, 0],
               [1, -1],
               [2, -1],
               [2, 1],
               [1, 1]]
coords = dict(zip(range(5), coordinates))

vis = Visualizer(graph, coords)
vis.plot_vertices()
vis.plot_face_centroids()
vis.plot_all_halfedges()
