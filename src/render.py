import matplotlib.pyplot as plt

import src.polygraph


class Renderer:
    def __init__(self, graph: src.polygraph.PolyGraph, coords, vertex_size=20, shrink=0.1):
        self.graph = graph
        self.coords = coords
        self.vertex_size = vertex_size
        self.shrink = shrink

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.face_centroids = self._compute_face_centroids()

    def _compute_face_centroids(self):
        centroids = []
        facenodes = self.graph.face_list(tag=False)
        for face_id in facenodes:
            first_halfedge = self.graph.first_face_halfedge(face_id)
            halfedges = self.graph.generate_halfedge_face_loop(first_halfedge)

            source_vertices = [self.graph.source_vertex(h, tag=False) for h in halfedges]
            vertex_coordinates = [self.coords[v] for v in source_vertices]
            x_centroid = sum(v[0] for v in vertex_coordinates) / len(vertex_coordinates)
            y_centroid = sum(v[1] for v in vertex_coordinates) / len(vertex_coordinates)
            centroids.append([x_centroid, y_centroid])

        centroids = dict(zip(facenodes, centroids))
        return centroids

    def plot_face(self, face_id):
        halfedge = self.graph.first_face_halfedge(face_id)
        face_loop = self.graph.generate_halfedge_face_loop(halfedge)
        face_vertices = [self.graph.source_vertex(hidx, tag=False) for hidx in face_loop]
        face_coords = [self.coords[v] for v in face_vertices]
        x = [c[0] for c in face_coords]
        y = [c[1] for c in face_coords]
        self.ax.fill(x, y, facecolor="lightgray", edgecolor="black", linewidth=2)

    def plot_all_faces(self):
        for face_id in range(self.graph.number_of_faces()):
            self.plot_face(face_id)

    def plot_vertices(self, label=True):
        x = [c[0] for c in self.coords.values()]
        y = [c[1] for c in self.coords.values()]
        self.ax.scatter(x, y, s=self.vertex_size ** 2, color="black")

        if label:
            for idx, coord in self.coords.items():
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

        halfedge_center = [0.5 * (source_coords[idx] + target_coords[idx]) for idx in range(2)]
        x = halfedge_center[0] + shrink * (face_coords[0] - halfedge_center[0])
        y = halfedge_center[1] + shrink * (face_coords[1] - halfedge_center[1])

        # self.ax.scatter(lx, ly, s=self.vertex_size ** 2, color="red")
        self.ax.text(x, y, str(idx), color="red", verticalalignment="center", horizontalalignment="center")

    def plot_all_halfedges(self):
        for idx in self.graph.halfedge_list(tag=False):
            self.plot_halfedge(idx)

    def plot(self):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.face_centroids = self._compute_face_centroids()
        self.plot_all_faces()
        self.plot_vertices()
        self.plot_all_halfedges()
