import networkx as nx
import itertools
from collections import deque


class HalfEdge:
    def __init__(self, id, face=None):
        self.id = id
        self.face = face
        self.next = None
        self.previous = None
        self.twin = None
        self.source = None
        self.target = None


class Tiler(nx.Graph):
    def __init__(self):
        super().__init__()
        self.next_half_edge_index = 0
        self.next_vertex_index = 0
        self.next_face_index = 0
        self.next_boundary_index = 0

        self.half_edge_tag = "h"
        self.vertex_tag = "v"
        self.face_tag = "f"
        self.boundary_tag = "b"

        self.vertex_coordinates = None
        self.user_defined_vertices = set()

        self.half_edges = dict()

    @classmethod
    def from_face_loops(cls, face_loops, vertex_coordinates=None):
        graph = cls()

        num_faces = len(face_loops)
        num_half_edges = sum(len(l) for l in face_loops)
        vertex_ids = set(itertools.chain.from_iterable(face_loops))
        num_vertices = len(vertex_ids)

        if vertex_coordinates is not None:
            assert all(
                v in vertex_coordinates for v in vertex_ids), "Some vertices were not found in vertex_coordinates"

        graph.vertex_coordinates = vertex_coordinates
        graph.user_defined_vertices = vertex_ids
        graph.next_half_edge_index = num_half_edges
        graph.next_vertex_index = num_vertices
        graph.next_face_index = num_faces
        graph.next_boundary_index = 0

        half_edge_ids = [(idx, graph.half_edge_tag) for idx in range(graph.next_half_edge_index)]
        graph.add_nodes_from(half_edge_ids, type="halfedge")
        half_edges = [HalfEdge(hidx) for hidx in half_edge_ids]
        graph.half_edges = dict(zip(half_edge_ids, half_edges))

        vertex_ids = [(idx, graph.vertex_tag) for idx in vertex_ids]
        graph.add_nodes_from(vertex_ids, type="vertex")

        face_ids = [(idx, graph.face_tag) for idx in range(graph.next_face_index)]
        graph.add_nodes_from(face_ids, type="face")

        graph.initialize_half_edges_from_face_loops(face_loops)
        graph.initialize_half_edge_source_associations(face_loops)
        graph.initialize_half_edge_target_associations(face_loops)
        graph.initialize_half_edge_face_associations(face_loops)
        graph.initialize_twin_associations()
        graph.initialize_boundary_source_target_associations()

        return graph

    def associate_previous_next_half_edge(self, previous_hidx, next_hidx):
        previous_hidx = self._ensure_tag_form(previous_hidx, self.half_edge_tag)
        next_hidx = self._ensure_tag_form(next_hidx, self.half_edge_tag)

        self.half_edges[previous_hidx].next = next_hidx
        self.half_edges[next_hidx].previous = previous_hidx

    def associate_half_edge_source_vertex(self, hidx, vidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        self.half_edges[hidx].source = vidx
        self.add_edge(hidx, vidx, type="source")

    def associate_half_edge_target_vertex(self, hidx, vidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        self.half_edges[hidx].target = vidx
        self.add_edge(hidx, vidx, type="target")

    def associate_half_edge_face(self, hidx, fidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        fidx = self._ensure_tag_form(fidx, self.face_tag)

        self.half_edges[hidx].face = fidx
        self.add_edge(hidx, fidx, type="face")

    def associate_half_edge_twin(self, hidx, twin_hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        twin_hidx = self._ensure_tag_form(twin_hidx, self.half_edge_tag)

        self.half_edges[hidx].twin = twin_hidx
        self.half_edges[twin_hidx].twin = hidx

    def _add_face_loop(self, source_half_edges, next_half_edges):
        assert len(source_half_edges) == len(next_half_edges)
        assert next_half_edges[-1] == source_half_edges[0]

        source_edges_with_tags = [(idx, self.half_edge_tag) for idx in source_half_edges]
        next_edges_with_tags = [(idx, self.half_edge_tag) for idx in next_half_edges]

        for src_hidx, next_hidx in zip(source_edges_with_tags, next_edges_with_tags):
            self.associate_previous_next_half_edge(src_hidx, next_hidx)

    def add_sequential_face_loop(self, half_edge_start, half_edge_stop):
        # !ASSUMES THAT HALF EDGES IN A FACE LOOP ARE INDEXED SEQUENTIALLY!
        assert half_edge_stop - half_edge_start + 1 > 1
        source_indices = list(range(half_edge_start, half_edge_stop + 1))
        next_indices = source_indices[1:] + [source_indices[0]]
        self._add_face_loop(source_indices, next_indices)

    def initialize_half_edges_from_face_loops(self, face_loops):
        start = 0
        for loop in face_loops:
            stop = start + len(loop) - 1
            self.add_sequential_face_loop(start, stop)
            start = stop + 1

    def initialize_half_edge_source_associations(self, face_loops):
        source_vertices = [(v, self.vertex_tag) for v in itertools.chain.from_iterable(face_loops)]
        half_edge_ids = [(h, self.half_edge_tag) for h in range(len(source_vertices))]

        for hidx, vidx in zip(half_edge_ids, source_vertices):
            self.associate_half_edge_source_vertex(hidx, vidx)

    def initialize_half_edge_target_associations(self, face_loops):
        target_vertices = []
        for loop in face_loops:
            rotated_loop = loop[1:] + [loop[0]]
            target_vertices += rotated_loop

        target_vertex_ids = [(v, self.vertex_tag) for v in target_vertices]
        half_edge_ids = [(h, self.half_edge_tag) for h in range(len(target_vertex_ids))]

        for hidx, vidx in zip(half_edge_ids, target_vertices):
            self.associate_half_edge_target_vertex(hidx, vidx)

    def initialize_half_edge_face_associations(self, face_loops):
        face_ids = []
        for face_idx, loop in enumerate(face_loops):
            loop_face_ids = len(loop) * [face_idx]
            face_ids += loop_face_ids

        face_nodes = [(f, self.face_tag) for f in face_ids]
        half_edge_nodes = [(h, self.half_edge_tag) for h in range(len(face_nodes))]

        for hidx, fidx in zip(half_edge_nodes, face_nodes):
            self.associate_half_edge_face(hidx, fidx)

    @staticmethod
    def _ensure_tag_form(idx, tag):
        if isinstance(idx, tuple):
            # assert len(idx) == 2
            return idx
        else:
            return idx, tag

    @staticmethod
    def _ensure_untagged_form(idx):
        if isinstance(idx, tuple):
            return idx[0]
        else:
            return idx

    def initialize_twin_associations(self):
        half_edge_nodes = [(h, self.half_edge_tag) for h in range(self.next_half_edge_index)]
        src_target = [(self.source_vertex(h), self.target_vertex(h)) for h in half_edge_nodes]
        src_target_to_half_edge = dict(zip(src_target, half_edge_nodes))
        boundary_nodes = []
        boundary_count = 0

        for src_target, hidx in src_target_to_half_edge.items():
            src, target = src_target
            if (target, src) in src_target_to_half_edge:
                twin_hidx = src_target_to_half_edge[(target, src)]
                self.associate_half_edge_twin(hidx, twin_hidx)
            else:
                boundary_hidx = (boundary_count, self.boundary_tag)
                boundary_nodes.append(boundary_hidx)
                boundary_half_edge = HalfEdge(boundary_hidx)
                self.half_edges[boundary_hidx] = boundary_half_edge
                self.associate_half_edge_twin(hidx, boundary_hidx)
                boundary_count += 1

        self.next_boundary_index = boundary_count
        self.add_nodes_from(boundary_nodes, type="boundary")

    def initialize_boundary_source_target_associations(self):
        boundary_nodes = [node for node in self.nodes() if node[1] == self.boundary_tag]
        boundary_source = []
        boundary_target = []
        for bnode in boundary_nodes:
            twin_edge = self.twin_half_edge(bnode)
            twin_source = self.source_vertex(twin_edge)
            twin_target = self.target_vertex(twin_edge)

            self.associate_half_edge_source_vertex(bnode, twin_target)
            self.associate_half_edge_target_vertex(bnode, twin_source)

            boundary_source.append(twin_target)
            boundary_target.append(twin_source)

    def _number_of_nodes(self, type):
        return sum(1 for vert, data in self.nodes(data=True) if data.get("type") == type)

    def number_of_half_edges(self):
        return self._number_of_nodes("halfedge")

    def number_of_vertices(self):
        return self._number_of_nodes("vertex")

    def number_of_faces(self):
        return self._number_of_nodes("face")

    def number_of_edges_of_type(self, type=None):
        return sum(1 for src, dst, data in self.edges(data=True) if data.get("type") == type)

    def source_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.half_edges[hidx].source

    def target_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.half_edges[hidx].target

    def twin_half_edge(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.half_edges[hidx].twin

    def next_half_edge(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.half_edges[hidx].next

    def previous_half_edge(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.half_edges[hidx].previous

    def face(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.half_edges[hidx].face

    def vertex_degree(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        degree = self.degree(vidx) // 2
        return degree

    def face_degree(self, fidx):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        return self.degree(fidx)

    def half_edge_on_boundary(self, hidx):
        tidx = self.twin_half_edge(hidx)
        return self.half_edges[tidx].face is None

    def face_half_edges(self, face_idx, tag=True):
        """return all half_edges connected to a face"""
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        halfedges = [h for f, h in self.edges(face_idx)]
        if tag:
            return halfedges
        else:
            halfedges = [h[0] for h in halfedges]
            return halfedges

    def generate_half_edge_face_loop(self, halfedge_idx):
        """generate list of halfedges in a face loop"""
        halfedge_idx = self._ensure_tag_form(halfedge_idx, self.half_edge_tag)
        loop = [halfedge_idx]
        next_halfedge = self.next_half_edge(halfedge_idx)
        while next_halfedge != halfedge_idx:
            loop.append(next_halfedge)
            next_halfedge = self.next_half_edge(next_halfedge)

        return loop
