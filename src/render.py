import matplotlib.pyplot as plt


class Renderer:
    def __init__(self, graph, coords, vertex_size=20):
        self.graph = graph
        self.coords = coords
        self.vertex_size = vertex_size

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect("equal")
        self.ax.axis("off")

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

    def plot(self):
        self.plot_all_faces()
        self.plot_vertices()
