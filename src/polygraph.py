import networkx as nx
import itertools


class PolyGraph(nx.DiGraph):
    def __init__(self, face_loops):
        super().__init__()
        num_faces = len(face_loops)
        num_halfedges = sum(len(l) for l in face_loops)
        vertex_ids = set(itertools.chain.from_iterable(face_loops))
        num_vertices = len(vertex_ids)

        # TODO: technically these are not the number of halfedges, faces, vertices etc.
        #       they are just they index of the next halfedge, face, etc. Change nomenclature, and add a method
        #       that actually computes number of halfedges etc. by iterating over the nodes in the graph.
        self.num_halfedges = num_halfedges
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        self.num_boundary = 0

        self.halfedge_tag = "h"
        self.vertex_tag = "v"
        self.face_tag = "f"
        self.boundary_tag = "b"

        halfedge_ids = [(idx, self.halfedge_tag) for idx in range(self.num_halfedges)]
        self.add_nodes_from(halfedge_ids, type="halfedge")

        vertex_ids = [(idx, self.vertex_tag) for idx in vertex_ids]
        self.add_nodes_from(vertex_ids, type="vertex")

        face_ids = [(idx, self.face_tag) for idx in range(self.num_faces)]
        self.add_nodes_from(face_ids, type="face")

        self.initialize_half_edges_from_face_loops(face_loops)
        self.add_halfedge_to_vertex_edges(face_loops)
        self.add_halfedge_to_face_edges(face_loops)
        self._add_twin_edges()
        self._add_twin_source_target_edges()

    def _add_face_loop(self, source_half_edges, next_half_edges):
        assert len(source_half_edges) == len(next_half_edges)
        assert next_half_edges[-1] == source_half_edges[0]

        source_edges_with_tags = [(idx, self.halfedge_tag) for idx in source_half_edges]
        next_edges_with_tags = [(idx, self.halfedge_tag) for idx in next_half_edges]

        next_edges = zip(source_edges_with_tags, next_edges_with_tags)
        self.add_edges_from(next_edges, type="next")
        prev_edges = zip(next_edges_with_tags, source_edges_with_tags)
        self.add_edges_from(prev_edges, type="previous")

    def add_face_loop(self, face_loop):
        next_indices = face_loop[1:] + [face_loop[0]]
        self._add_face_loop(face_loop, next_indices)

    def add_sequential_face_loop(self, halfedge_start, halfedge_stop):
        # !ASSUMES THAT HALF EDGES IN A FACE LOOP ARE INDEXED SEQUENTIALLY!
        assert halfedge_stop - halfedge_start + 1 >= 3
        source_indices = list(range(halfedge_start, halfedge_stop + 1))
        next_indices = source_indices[1:] + [source_indices[0]]
        self._add_face_loop(source_indices, next_indices)

    def initialize_half_edges_from_face_loops(self, face_loops):
        start = 0
        for loop in face_loops:
            stop = start + len(loop) - 1
            self.add_sequential_face_loop(start, stop)
            start = stop + 1

    def add_undirected_edge(self, source, dst, name):
        self.add_edge(source, dst, type=name)
        self.add_edge(dst, source, type=name)

    def add_undirected_edges(self, source, dst, name):
        # adds edges from source to dst and vice verse
        # both sets of edges are assigned the same name
        self.add_edges_from(zip(source, dst), type=name)
        self.add_edges_from(zip(dst, source), type=name)

    def remove_undirected_edge(self, source, dst):
        self.remove_edge(source, dst)
        self.remove_edge(dst, source)

    def add_edges_to_halfedge_source(self, face_loops):
        source_vertices = [(v, self.vertex_tag) for v in itertools.chain.from_iterable(face_loops)]
        halfedge_ids = [(h, self.halfedge_tag) for h in range(len(source_vertices))]
        self.add_undirected_edges(halfedge_ids, source_vertices, "source")

    def add_edges_to_halfedge_target(self, face_loops):
        target_vertices = []
        for loop in face_loops:
            rotated_loop = loop[1:] + [loop[0]]
            target_vertices += rotated_loop

        target_vertex_ids = [(v, self.vertex_tag) for v in target_vertices]
        halfedge_ids = [(h, self.halfedge_tag) for h in range(len(target_vertex_ids))]
        self.add_undirected_edges(halfedge_ids, target_vertex_ids, "target")

    def add_halfedge_to_vertex_edges(self, face_loops):
        self.add_edges_to_halfedge_source(face_loops)
        self.add_edges_to_halfedge_target(face_loops)

    def add_halfedge_to_face_edges(self, face_loops):
        face_ids = []
        for face_idx, loop in enumerate(face_loops):
            loop_face_ids = len(loop) * [face_idx]
            face_ids += loop_face_ids

        face_nodes = [(f, self.face_tag) for f in face_ids]
        halfedge_nodes = [(h, self.halfedge_tag) for h in range(len(face_nodes))]
        self.add_undirected_edges(halfedge_nodes, face_nodes, "face")

    @staticmethod
    def _ensure_tag_form(idx, tag):
        if isinstance(idx, int):
            return idx, tag
        else:
            assert isinstance(idx, tuple)
            assert len(idx) == 2
            # assert idx[1] == tag
            return idx

    def _add_twin_edges(self):
        halfedge_nodes = [(h, self.halfedge_tag) for h in range(self.num_halfedges)]
        src_target = [(self.source_vertex(h), self.target_vertex(h)) for h in halfedge_nodes]
        src_target_to_halfedge = dict(zip(src_target, halfedge_nodes))
        twins = []
        boundary_nodes = []
        boundary_count = 0

        for idx in range(len(src_target)):
            src, target = src_target[idx]
            if (target, src) in src_target_to_halfedge:
                twin_edge = src_target_to_halfedge[(target, src)]
                twins.append(twin_edge)
            else:
                boundary_node = (boundary_count, self.boundary_tag)
                boundary_nodes.append(boundary_node)
                twins.append(boundary_node)
                boundary_count += 1

        self.num_boundary = boundary_count
        self.add_nodes_from(boundary_nodes, type="boundary")
        self.add_undirected_edges(halfedge_nodes, twins, "twin")

    def _add_twin_source_target_edges(self):
        boundary_nodes = [node for node in self.nodes() if node[1] == self.boundary_tag]
        boundary_source = []
        boundary_target = []
        for bnode in boundary_nodes:
            twin_edge = self.twin_halfedge(bnode)
            twin_source = self.source_vertex(twin_edge)
            twin_target = self.target_vertex(twin_edge)
            boundary_source.append(twin_target)
            boundary_target.append(twin_source)

        self.add_undirected_edges(boundary_nodes, boundary_source, "source")
        self.add_undirected_edges(boundary_nodes, boundary_target, "target")

    def _number_of_nodes(self, type):
        return sum(1 for vert, data in self.nodes(data=True) if data.get("type") == type)

    def number_of_vertices(self):
        return self._number_of_nodes("vertex")

    def number_of_halfedges(self):
        return self._number_of_nodes("halfedge")

    def number_of_faces(self):
        return self._number_of_nodes("face")

    def source_vertex(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        vertex_id = next(dst for _, dst, data in self.edges(halfedge_index, data=True) if data.get("type") == "source")
        if tag:
            return vertex_id
        else:
            return vertex_id[0]

    def target_vertex(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        vertex_id = next(dst for _, dst, data in self.edges(halfedge_index, data=True) if data.get("type") == "target")
        if tag:
            return vertex_id
        else:
            return vertex_id[0]

    def twin_halfedge(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        twin = next(target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "twin")
        if tag:
            return twin
        else:
            return twin[0]

    def next_halfedge(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        next_edge = next(
            target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "next"
        )
        if tag:
            return next_edge
        else:
            return next_edge[0]

    def previous_halfedge(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        prev_edge = next(
            target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "previous"
        )
        if tag:
            return prev_edge
        else:
            return prev_edge[0]

    def face(self, halfedge_index, tag=True):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        face = next(
            target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "face"
        )
        if tag:
            return face
        else:
            return face[0]

    def vertex_degree(self, vertex_index):
        vertex_index = self._ensure_tag_form(vertex_index, self.vertex_tag)
        d = self.in_degree(vertex_index) // 2
        return d

    def face_degree(self, face_index):
        face_index = self._ensure_tag_form(face_index, self.face_tag)
        d = self.in_degree(face_index)
        return d

    def create_face(self):
        new_face_idx = (self.num_faces, self.face_tag)
        self.add_node(new_face_idx, type="face")
        self.num_faces += 1
        return new_face_idx

    def create_halfedge(self, next_halfedge, prev_halfedge):
        source_vertex = self.target_vertex(prev_halfedge)
        target_vertex = self.source_vertex(next_halfedge)
        face_idx = self.face(next_halfedge)
        halfedge_idx = (self.num_halfedges, self.halfedge_tag)
        self.add_node(halfedge_idx, type="halfedge")
        self.add_undirected_edge(halfedge_idx, source_vertex, "source")
        self.add_undirected_edge(halfedge_idx, target_vertex, "target")
        self.add_undirected_edge(halfedge_idx, face_idx, "face")
        self.num_halfedges += 1
        return halfedge_idx

    def associate_previous_next(self, prev_halfedge, next_halfedge):
        self.add_edge(prev_halfedge, next_halfedge, type="next")
        self.add_edge(next_halfedge, prev_halfedge, type="previous")

    def insert_edge(self, hidx, k):
        """insert an edge from source of `hidx` to target of halfedge that is k `next` steps away"""
        hidx = self._ensure_tag_form(hidx, self.halfedge_tag)
        face_idx = self.face(hidx)
        assert 1 <= k <= self.face_degree(face_idx)

        new_face_idx = self.create_face()
        # remove edges to current face and add edge to new face
        current_halfedge = hidx
        for count in range(k + 1):
            self.remove_undirected_edge(current_halfedge, face_idx)
            self.add_undirected_edge(current_halfedge, new_face_idx, "face")
            current_halfedge = self.next_halfedge(current_halfedge)

        new_face_next_halfedge = hidx
        new_face_prev_halfedge = self.previous_halfedge(current_halfedge)
        old_face_next_halfedge = current_halfedge
        old_face_prev_halfedge = self.previous_halfedge(hidx)

        self.remove_undirected_edge(new_face_prev_halfedge, old_face_next_halfedge)
        self.remove_undirected_edge(new_face_next_halfedge, old_face_prev_halfedge)

        new_face_new_halfedge = self.create_halfedge(new_face_next_halfedge, new_face_prev_halfedge)
        old_face_new_halfedge = self.create_halfedge(old_face_next_halfedge, old_face_prev_halfedge)

        self.associate_previous_next(new_face_prev_halfedge, new_face_new_halfedge)
        self.associate_previous_next(new_face_new_halfedge, new_face_next_halfedge)
        self.associate_previous_next(old_face_new_halfedge, old_face_next_halfedge)
        self.associate_previous_next(old_face_prev_halfedge, old_face_new_halfedge)
        self.add_undirected_edge(new_face_new_halfedge, old_face_new_halfedge, "twin")
