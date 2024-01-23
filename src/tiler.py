import networkx as nx
import itertools
from collections import deque


class HalfEdge:
    def __init__(
            self,
            id,
            face=None,
            next=None,
            previous=None,
            twin=None,
            source=None,
            target=None
    ):
        self.id = id
        self.face = face
        self.next = next
        self.previous = previous
        self.twin = twin
        self.source = source
        self.target = target


class Tiler(nx.MultiGraph):
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
        self.vertex_degrees = dict()
        self.face_degrees = dict()

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
        graph.add_nodes_from(half_edge_ids, type="half_edge")
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
        graph.initialize_vertex_degrees()
        graph.initialize_face_degrees()

        return graph

    def is_user_defined_vertex(self, vidx):
        vidx = self._ensure_untagged_form(vidx)
        return vidx in self.user_defined_vertices

    def associate_previous_next_half_edge(self, previous_hidx, next_hidx):
        previous_hidx = self._ensure_tag_form(previous_hidx, self.half_edge_tag)
        next_hidx = self._ensure_tag_form(next_hidx, self.half_edge_tag)

        self.half_edges[previous_hidx].next = next_hidx
        self.half_edges[next_hidx].previous = previous_hidx

    def associate_half_edge_source_vertex(self, hidx, vidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        self.half_edges[hidx].source = vidx
        self.add_edge(hidx, vidx, key="source")

    def associate_half_edge_target_vertex(self, hidx, vidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        self.half_edges[hidx].target = vidx
        self.add_edge(hidx, vidx, key="target")

    def associate_half_edge_face(self, hidx, fidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        fidx = self._ensure_tag_form(fidx, self.face_tag)

        self.half_edges[hidx].face = fidx
        self.add_edge(hidx, fidx, key="face")

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

    def initialize_vertex_degrees(self):
        vertices = self.vertex_list(tag=True)
        degrees = self.vertex_degree_of_list(vertices)
        self.vertex_degrees.update(zip(vertices, degrees))

    def initialize_face_degrees(self):
        faces = self.face_list(tag=True)
        degrees = self.face_degree_of_list(faces)
        self.face_degrees.update(zip(faces, degrees))

    def _number_of_nodes(self, type):
        return sum(1 for vert, data in self.nodes(data=True) if data.get("type") == type)

    def number_of_half_edges(self):
        return self._number_of_nodes("half_edge")

    def number_of_vertices(self):
        return self._number_of_nodes("vertex")

    def number_of_faces(self):
        return self._number_of_nodes("face")

    def number_of_edges_of_type(self, type=None):
        return sum(1 for src, dst, key in self.edges(keys=True) if key == type)

    def _node_list_by_type(self, node_type, tag):
        nodes = [node for node, data in self.nodes(data=True) if data.get("type") == node_type]
        if not tag:
            nodes = [self._ensure_untagged_form(idx) for idx in nodes]
        return nodes

    def half_edge_list(self, tag=True):
        return self._node_list_by_type("half_edge", tag)

    def halfedge_list(self, tag=True):
        return self.half_edge_list(tag)

    def face_list(self, tag=True):
        return self._node_list_by_type("face", tag)

    def vertex_list(self, tag=True):
        return self._node_list_by_type("vertex", tag)

    def source_vertex(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self.half_edges[hidx].source
        if tag:
            return vidx
        else:
            return self._ensure_untagged_form(vidx)

    def target_vertex(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self.half_edges[hidx].target
        if tag:
            return vidx
        else:
            return self._ensure_untagged_form(vidx)

    def twin_half_edge(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        twin_idx = self.half_edges[hidx].twin
        if tag:
            return twin_idx
        else:
            return self._ensure_untagged_form(twin_idx)

    def next_half_edge(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        next_idx = self.half_edges[hidx].next
        if tag:
            return next_idx
        else:
            return self._ensure_untagged_form(next_idx)

    def previous_half_edge(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        prev_idx = self.half_edges[hidx].previous
        if tag:
            return prev_idx
        else:
            return self._ensure_untagged_form(prev_idx)

    def face(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        fidx = self.half_edges[hidx].face
        if tag:
            return fidx
        else:
            return self._ensure_untagged_form(fidx)

    def vertex_degree(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        # degree = self.degree(vidx) // 2
        degree = self.vertex_degrees[vidx]

        return degree

    def vertex_degree_of_list(self, vertex_list):
        vertex_list = [self._ensure_tag_form(vidx, self.vertex_tag) for vidx in vertex_list]
        degree_dict = self.degree(vertex_list)
        degrees = [degree_dict[vidx] // 2 for vidx in vertex_list]
        return degrees

    def face_degree(self, fidx):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        return self.degree(fidx)

    def face_degree_of_list(self, face_list):
        face_list = [self._ensure_tag_form(fidx, self.face_tag) for fidx in face_list]
        degree_dict = self.degree(face_list)
        degrees = [degree_dict[fidx] for fidx in face_list]
        return degrees

    def half_edge_on_boundary(self, hidx):
        tidx = self.twin_half_edge(hidx)
        return self.half_edges[tidx].face is None

    def face_half_edges(self, face_idx, tag=True):
        """return all half_edges connected to a face"""
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        half_edges = [h for f, h in self.edges(face_idx)]
        if tag:
            return half_edges
        else:
            half_edges = [self._ensure_untagged_form(h) for h in half_edges]
            return half_edges

    def first_face_halfedge(self, face_idx, tag=True):
        face_idx = self._ensure_tag_form(face_idx, self.face_tag)
        halfedge = next(h for f, h, key in self.edges(face_idx, keys=True) if key == "face")
        if tag:
            return halfedge
        else:
            return halfedge[0]

    def generate_half_edge_face_loop(self, half_edge_idx):
        """generate list of half_edges in a face loop"""
        half_edge_idx = self._ensure_tag_form(half_edge_idx, self.half_edge_tag)
        loop = [half_edge_idx]
        next_half_edge = self.next_half_edge(half_edge_idx)
        while next_half_edge != half_edge_idx:
            loop.append(next_half_edge)
            next_half_edge = self.next_half_edge(next_half_edge)

        return loop

    def generate_halfedge_face_loop(self, halfedge_idx):
        return self.generate_half_edge_face_loop(halfedge_idx)

    def is_half_edge(self, hidx):
        if hidx is None:
            return False

        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        return self.has_node(hidx) and self.nodes[hidx].get("type") == "half_edge"

    def is_boundary_half_edge(self, hidx):
        if hidx is None:
            return False

        hidx = self._ensure_tag_form(hidx, self.boundary_tag)
        return self.has_node(hidx) and self.nodes[hidx].get("type") == "boundary"

    def create_face(self):
        new_face_idx = (self.next_face_index, self.face_tag)
        self.add_node(new_face_idx, type="face")
        self.next_face_index += 1
        return new_face_idx

    def set_face_degree(self, fidx, degree):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        self.face_degrees[fidx] = degree

    def create_half_edge(self, next_half_edge_idx, prev_half_edge_idx):
        next_half_edge_idx = self._ensure_tag_form(next_half_edge_idx, self.half_edge_tag)
        prev_half_edge_idx = self._ensure_tag_form(prev_half_edge_idx, self.half_edge_tag)

        next_half_edge = self.half_edges[next_half_edge_idx]
        prev_half_edge = self.half_edges[prev_half_edge_idx]
        assert next_half_edge.face == prev_half_edge.face, "next/previous edges must share face"

        source_vertex = prev_half_edge.target
        target_vertex = next_half_edge.source
        face_idx = next_half_edge.face

        new_half_edge_idx = (self.next_half_edge_index, self.half_edge_tag)
        self.next_half_edge_index += 1

        new_half_edge = HalfEdge(
            new_half_edge_idx,
            face=face_idx,
            next=next_half_edge_idx,
            previous=prev_half_edge_idx,
            source=source_vertex,
            target=target_vertex
        )
        self.half_edges[new_half_edge_idx] = new_half_edge
        self.add_node(new_half_edge_idx, type="half_edge")

        next_half_edge.previous = new_half_edge_idx
        prev_half_edge.next = new_half_edge_idx

        self.add_edge(new_half_edge_idx, source_vertex, key="source")
        self.add_edge(new_half_edge_idx, target_vertex, key="target")
        self.add_edge(new_half_edge_idx, face_idx, key="face")

        return new_half_edge_idx

    def create_boundary_half_edge(self, target_vertex, source_vertex):
        target_vertex = self._ensure_tag_form(target_vertex, self.vertex_tag)
        source_vertex = self._ensure_tag_form(source_vertex, self.vertex_tag)

        boundary_edge_idx = (self.next_boundary_index, self.boundary_tag)
        self.next_boundary_index += 1

        self.add_node(boundary_edge_idx, type="boundary")
        boundary_edge = HalfEdge(
            boundary_edge_idx,
            source=source_vertex,
            target=target_vertex
        )
        self.half_edges[boundary_edge_idx] = boundary_edge

        self.add_edge(boundary_edge_idx, source_vertex, key="source")
        self.add_edge(boundary_edge_idx, target_vertex, key="target")

        return boundary_edge_idx

    def is_valid_edge_insert(self, hidx, k):
        if not self.is_half_edge(hidx):
            return False

        face_idx = self.face(hidx)
        if not (0 <= k < self.face_degree(face_idx) - 1):
            return False

        return True

    def insert_half_edge(self, hidx, k):
        """
        Insert edge from source of hidx to target of half edge that is
        k next steps ahead.
        """
        assert self.is_valid_edge_insert(hidx, k), "Invalid edge insert encountered"

        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        face_idx = self.face(hidx)
        face_degree = self.face_degree(face_idx)
        self.set_face_degree(face_idx, face_degree - 2)

        new_face_idx = self.create_face()
        self.set_face_degree(new_face_idx, k + 2)

        # remove edges to current face and add edge to new face
        current_half_edge = hidx
        for count in range(k + 1):
            self.remove_edge(current_half_edge, face_idx, key="face")
            self.associate_half_edge_face(current_half_edge, new_face_idx)
            current_half_edge = self.next_half_edge(current_half_edge)

        new_face_next_idx = hidx
        new_face_prev_idx = self.previous_half_edge(current_half_edge)
        old_face_next_idx = current_half_edge
        old_face_prev_idx = self.previous_half_edge(hidx)

        new_face_source_vertex = self.target_vertex(new_face_prev_idx)
        new_face_target_vertex = self.source_vertex(hidx)
        self.vertex_degrees[new_face_source_vertex] += 1
        self.vertex_degrees[new_face_target_vertex] += 1

        new_face_new_idx = self.create_half_edge(new_face_next_idx, new_face_prev_idx)
        old_face_new_idx = self.create_half_edge(old_face_next_idx, old_face_prev_idx)

        self.associate_half_edge_twin(new_face_new_idx, old_face_new_idx)

    def vertex_coordinate(self, vertex_idx):
        vertex_idx = self._ensure_untagged_form(vertex_idx)
        return self.vertex_coordinates[vertex_idx]

    def get_new_vertex_coordinate(self, next_vertex, previous_vertex):
        if self.vertex_coordinates is not None:
            coord1 = self.vertex_coordinate(next_vertex)
            coord2 = self.vertex_coordinate(previous_vertex)
            new_coord = [(c1 + c2) / 2 for c1, c2 in zip(coord1, coord2)]
            return new_coord
        else:
            return None

    def create_vertex(self, coord=None):
        new_vertex_idx = (self.next_vertex_index, self.vertex_tag)
        self.add_node(new_vertex_idx, type="vertex")
        if coord is not None and self.vertex_coordinates is not None:
            # assert len(coord) == 2
            self.vertex_coordinates[self.next_vertex_index] = coord
        self.next_vertex_index += 1
        return new_vertex_idx

    def set_vertex_degree(self, vidx, degree):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        self.vertex_degrees[vidx] = degree

    def set_target_vertex(self, hidx, vidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)

        current_target = self.target_vertex(hidx)
        twin_edge = self.twin_half_edge(hidx)
        self.remove_edge(hidx, current_target, key="target")
        self.remove_edge(twin_edge, current_target, key="source")

        self.half_edges[hidx].target = vidx
        self.half_edges[twin_edge].source = vidx
        self.add_edge(hidx, vidx, key="target")
        self.add_edge(twin_edge, vidx, key="source")

    def _insert_boundary_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert self.half_edge_on_boundary(hidx)

        current_target_vertex = self.target_vertex(hidx)
        current_source_vertex = self.source_vertex(hidx)

        new_vertex_coord = self.get_new_vertex_coordinate(current_target_vertex, current_source_vertex)
        new_vertex_idx = self.create_vertex(new_vertex_coord)
        self.set_vertex_degree(new_vertex_idx, 2)

        next_half_edge = self.next_half_edge(hidx)
        self.set_target_vertex(hidx, new_vertex_idx)

        new_half_edge = self.create_half_edge(next_half_edge, hidx)
        new_boundary_edge = self.create_boundary_half_edge(new_vertex_idx, current_target_vertex)
        self.associate_half_edge_twin(new_half_edge, new_boundary_edge)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] += 1

        return new_vertex_idx

    def _insert_interior_vertex(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert not self.half_edge_on_boundary(hidx)

        next_hidx = self.next_half_edge(hidx)
        twin_hidx = self.twin_half_edge(hidx)
        twin_prev_hidx = self.previous_half_edge(twin_hidx)

        current_target_vertex = self.target_vertex(hidx)
        current_source_vertex = self.source_vertex(hidx)
        new_vertex_coord = self.get_new_vertex_coordinate(current_target_vertex, current_source_vertex)
        new_vertex_idx = self.create_vertex(new_vertex_coord)
        self.set_vertex_degree(new_vertex_idx, 2)

        self.set_target_vertex(hidx, new_vertex_idx)

        new_next_hidx = self.create_half_edge(next_hidx, hidx)
        new_twin_prev_hidx = self.create_half_edge(twin_hidx, twin_prev_hidx)
        self.associate_half_edge_twin(new_next_hidx, new_twin_prev_hidx)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] += 1
        twin_face_idx = self.face(twin_hidx)
        self.face_degrees[twin_face_idx] += 1

        return new_vertex_idx

    def insert_vertex(self, hidx, tag=True):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        if self.half_edge_on_boundary(hidx):
            new_vertex = self._insert_boundary_vertex(hidx)
        else:
            new_vertex = self._insert_interior_vertex(hidx)

    def _delete_vertex_coordinates(self, vidx):
        vidx = self._ensure_untagged_form(vidx)
        if self.vertex_coordinates is not None:
            coords = self.vertex_coordinates.pop(vidx, None)
            if coords is None:
                print("Warning: deleted vertex not found in vertex coordinates")

    def _delete_half_edge(self, hidx):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        twin = self.twin_half_edge(hidx)

        self.remove_node(hidx)
        self.remove_node(twin)
        self.half_edges.pop(hidx)
        self.half_edges.pop(twin)

    def _delete_vertex(self, vidx):
        vidx = self._ensure_tag_form(vidx, self.vertex_tag)
        self.remove_node(vidx)
        self.vertex_degrees.pop(vidx)
        self._delete_vertex_coordinates(vidx)

    def _delete_boundary_vertex(self, hidx):
        """deletes vertex at source of hidx"""
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.half_edge_on_boundary(hidx)
        source_vertex = self.source_vertex(hidx)
        assert self.vertex_degree(source_vertex) == 2
        assert not self.is_user_defined_vertex(source_vertex)

        next_half_edge = self.next_half_edge(hidx)
        prev_half_edge = self.previous_half_edge(hidx)

        target_vertex = self.target_vertex(hidx)
        self.set_target_vertex(prev_half_edge, target_vertex)
        self.associate_previous_next_half_edge(prev_half_edge, next_half_edge)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] -= 1

        self._delete_half_edge(hidx)
        self._delete_vertex(source_vertex)

    def _delete_interior_vertex(self, hidx):
        """deletes vertex at source of hidx"""
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert not self.half_edge_on_boundary(hidx)
        source_vertex = self.source_vertex(hidx)
        # assert self.vertex_degree(source_vertex) == 2

        next_half_edge = self.next_half_edge(hidx)
        prev_half_edge = self.previous_half_edge(hidx)
        twin_half_edge = self.twin_half_edge(hidx)

        next_twin_half_edge = self.next_half_edge(twin_half_edge)
        prev_twin_half_edge = self.previous_half_edge(twin_half_edge)

        target_vertex = self.target_vertex(hidx)
        self.set_target_vertex(prev_half_edge, target_vertex)
        self.associate_previous_next_half_edge(prev_half_edge, next_half_edge)
        self.associate_previous_next_half_edge(prev_twin_half_edge, next_twin_half_edge)

        face_idx = self.face(hidx)
        self.face_degrees[face_idx] -= 1
        twin_face_idx = self.face(twin_half_edge)
        self.face_degrees[twin_face_idx] -= 1

        self._delete_half_edge(hidx)
        self._delete_vertex(source_vertex)

    def is_valid_delete_source_vertex(self, hidx):
        """Check if vertex at source of hidx can be deleted"""
        if not self.is_half_edge(hidx):
            return False

        vidx = self.source_vertex(hidx)
        if self.is_user_defined_vertex(vidx):
            return False

        if self.vertex_degree(vidx) != 2:
            return False

        fidx = self.face(hidx)
        if self.face_degree(fidx) < 3:
            return False

        if not self.half_edge_on_boundary(hidx):
            twin_edge = self.twin_half_edge(hidx)
            twin_face = self.face(twin_edge)
            if self.face_degree(twin_face) < 3:
                return False

        return True

    def delete_source_vertex(self, hidx):
        assert self.is_valid_delete_source_vertex(hidx), "cannot delete vertex at source of : " + str(hidx)
        if self.half_edge_on_boundary(hidx):
            self._delete_boundary_vertex(hidx)
        else:
            self._delete_interior_vertex(hidx)

    def is_valid_delete_half_edge(self, hidx):
        if not self.is_half_edge(hidx):
            return False

        if self.half_edge_on_boundary(hidx):
            return False

        source_vertex = self.source_vertex(hidx)
        target_vertex = self.target_vertex(hidx)
        if self.vertex_degree(source_vertex) <= 2 or self.vertex_degree(target_vertex) <= 2:
            return False

        # Deleting a half-edge results in merging of faces on either side.
        # If these faces are already the same, this is an invalid delete
        twin_edge = self.twin_half_edge(hidx)
        current_face = self.face(hidx)
        twin_face = self.face(twin_edge)
        if current_face == twin_face:
            return False

        return True

    def _half_edges_of_face(self, fidx):
        fidx = self._ensure_tag_form(fidx, self.face_tag)
        edges = (target for src, target, key in self.edges(fidx, keys=True) if key == "face")
        return edges

    def delete_half_edge(self, hidx):
        assert self.is_valid_delete_half_edge(hidx)

        next_edge = self.next_half_edge(hidx)
        prev_edge = self.previous_half_edge(hidx)
        twin_edge = self.twin_half_edge(hidx)
        prev_twin = self.previous_half_edge(twin_edge)
        next_twin = self.next_half_edge(twin_edge)
        source_vertex = self.source_vertex(hidx)
        target_vertex = self.target_vertex(hidx)

        self.vertex_degrees[source_vertex] -= 1
        self.vertex_degrees[target_vertex] -= 1

        current_face = self.face(hidx)
        twin_face = self.face(twin_edge)

        current_face_degree = self.face_degree(current_face)
        twin_face_degree = self.face_degree(twin_face)
        self.set_face_degree(current_face, current_face_degree + twin_face_degree - 2)

        # Associate all edges from neighboring face with current face and delete neighboring face
        twin_face_edges = self._half_edges_of_face(twin_face)
        for half_edge in twin_face_edges:
            self.associate_half_edge_face(half_edge, current_face)

        # delete the twin face
        self.remove_node(twin_face)
        self.face_degrees.pop(twin_face)

        self.associate_previous_next_half_edge(prev_edge, next_twin)
        self.associate_previous_next_half_edge(prev_twin, next_edge)
        self._delete_half_edge(hidx)

    def knn_half_edges(self, hidx, k):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert k >= 1

        seen = set()
        queue = deque()

        seen.add(hidx)
        queue.append(hidx)
        neighbors = []

        def append_if_not_seen(edge):
            if edge not in seen:
                seen.add(edge)
                queue.append(edge)

        while queue:
            edge = queue.popleft()
            neighbors.append(edge)
            if len(neighbors) >= k:
                return neighbors

            next_edge = self.next_half_edge(edge)
            append_if_not_seen(next_edge)

            prev_edge = self.previous_half_edge(edge)
            append_if_not_seen(prev_edge)

            if not self.half_edge_on_boundary(edge):
                twin_edge = self.twin_half_edge(edge)
                append_if_not_seen(twin_edge)

        return neighbors

    def knn_half_edges_with_boundary(self, hidx, k):
        hidx = self._ensure_tag_form(hidx, self.half_edge_tag)
        assert self.is_half_edge(hidx)
        assert k >= 1

        seen = set()
        queue = deque()

        seen.add(hidx)
        queue.append(hidx)
        neighbors = []

        def append_if_not_seen(edge):
            if edge not in seen:
                seen.add(edge)
                queue.append(edge)

        while queue:
            edge = queue.popleft()
            neighbors.append(edge)
            if len(neighbors) >= k:
                return neighbors

            if self.is_half_edge(edge):
                next_edge = self.next_half_edge(edge)
                append_if_not_seen(next_edge)

                prev_edge = self.previous_half_edge(edge)
                append_if_not_seen(prev_edge)

                twin_edge = self.twin_half_edge(edge)
                append_if_not_seen(twin_edge)

        return neighbors
