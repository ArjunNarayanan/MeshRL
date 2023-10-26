import networkx as nx
import numpy as np


class PolyGraph(nx.DiGraph):
    def __init__(self, num_halfedges, num_vertices, num_faces):
        super().__init__()
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

        vertex_ids = [(idx, self.vertex_tag) for idx in range(self.num_vertices)]
        self.add_nodes_from(vertex_ids, type="vertex")

        face_ids = [(idx, self.face_tag) for idx in range(self.num_faces)]
        self.add_nodes_from(face_ids, type="face")

    def add_face_loop(self, source_half_edges, next_half_edges):
        assert len(source_half_edges) == len(next_half_edges)
        assert next_half_edges[-1] == source_half_edges[0]

        source_edges_with_tags = [(idx, self.halfedge_tag) for idx in source_half_edges]
        next_edges_with_tags = [(idx, self.halfedge_tag) for idx in next_half_edges]

        next_edges = zip(source_edges_with_tags, next_edges_with_tags)
        self.add_edges_from(next_edges, type="next")
        prev_edges = zip(next_edges_with_tags, source_edges_with_tags)
        self.add_edges_from(prev_edges, type="previous")

    def add_sequential_face_loop(self, halfedge_start, halfedge_stop):
        # !ASSUMES THAT HALF EDGES IN A FACE LOOP ARE INDEXED SEQUENTIALLY!
        assert halfedge_stop - halfedge_start + 1 >= 3
        source_indices = list(range(halfedge_start, halfedge_stop + 1))
        next_indices = source_indices[1:] + [source_indices[0]]
        self.add_face_loop(source_indices, next_indices)

    def add_undirected_edges(self, source, dst, name):
        # adds edges from source to dst and vice verse
        # both sets of edges are assigned the same name
        self.add_edges_from(zip(source, dst), type=name)
        self.add_edges_from(zip(dst, source), type=name)

    def add_halfedge_to_vertex_edges(self, vertex_connectivity):
        # ASSUMES that rows of vertex_connectivity correspond to half edges indexed in order
        # this will not be true after the graph has been manipulated so make sure you only use this
        # when the graph is initialized
        # TODO : add a flag to check if the graph is in initial state

        assert vertex_connectivity.shape == (self.num_halfedges, 2)

        halfedge_ids = [(idx, self.halfedge_tag) for idx in range(self.num_halfedges)]

        source_vertex_ids = [(idx, self.vertex_tag) for idx in vertex_connectivity[:, 0]]
        self.add_undirected_edges(halfedge_ids, source_vertex_ids, "source")

        dst_vertex_ids = [(idx, self.vertex_tag) for idx in vertex_connectivity[:, 1]]
        self.add_undirected_edges(halfedge_ids, dst_vertex_ids, "target")

    def add_halfedge_to_face_edges(self, halfedge_to_faces):
        # ASSUMES that face_ids corresponds to half edges indexed in order. This will not be true after the
        # graph has been manipulated so make sure to use this only when graph is initialized
        # TODO : add a flag to check if the graph is in initial state

        assert len(halfedge_to_faces) == self.num_halfedges
        face_ids = [(idx, self.face_tag) for idx in halfedge_to_faces]
        halfedge_ids = [(idx, self.halfedge_tag) for idx in range(self.num_halfedges)]

        self.add_undirected_edges(face_ids, halfedge_ids, "face")

    def _ensure_tag_form(self, idx, tag):
        if isinstance(idx, int):
            return (idx, tag)
        else:
            assert isinstance(idx, tuple)
            assert len(idx) == 2
            assert idx[1] == tag
            return idx

    def add_twin_edges(self, vertex_connectivity):
        # assumes that each column in `vertex_connectivity` represents the source-target vertices for each
        # halfedge
        assert vertex_connectivity.shape == (self.num_halfedges, 2)

        # sort the edges by vertex ids
        sorted_connectivity = np.sort(vertex_connectivity, axis=1)

        # get the indices to sort the edges lexicographically, this sorts twin edges to be next to each other
        sort_index = np.lexsort((sorted_connectivity[:, 1], sorted_connectivity[:, 0]))

        # get the connectivity in sorted order
        sorted_connectivity = sorted_connectivity[sort_index, :]
        # shift the connectivity and compare it to find edges that have duplicates
        # these are twin edges
        shifted_connectivity = np.roll(sorted_connectivity, -1, axis=0)
        has_twin = (sorted_connectivity == shifted_connectivity).all(axis=1)

        # get the half edge indices in sorted order
        half_edge_ids = np.arange(self.num_halfedges)[sort_index]

        # associate half-edges to their twins
        src_halfedge = half_edge_ids[has_twin]
        dst_halfedge = half_edge_ids[np.roll(has_twin, 1)]

        # add twin edges between these half edges
        src_halfedge_verts = [(idx, self.halfedge_tag) for idx in src_halfedge]
        dst_halfedge_verts = [(idx, self.halfedge_tag) for idx in dst_halfedge]
        self.add_undirected_edges(src_halfedge_verts, dst_halfedge_verts, "twin")

        # find all boundary edges which do not have twins
        # make sure you select ALL indices that have twins -- for this you need to roll forward
        # the has_twin vector
        _has_twin = np.logical_or(has_twin, np.roll(has_twin, +1))
        is_boundary = ~_has_twin
        self.num_boundary = np.count_nonzero(is_boundary)
        # create num_boundary boundary half edges
        boundary_node_ids = [(idx, self.boundary_tag) for idx in range(self.num_boundary)]
        self.add_nodes_from(boundary_node_ids, type="boundary")

        # add twin edges between half edges and boundary nodes
        boundary_halfedge_ids = half_edge_ids[is_boundary]
        boundary_halfedge_verts = [(idx, self.halfedge_tag) for idx in boundary_halfedge_ids]
        self.add_undirected_edges(boundary_halfedge_verts, boundary_node_ids, name="twin")

        # target vertices of half edges on boundary are source vertices of boundary nodes
        boundary_src_nodes = [self.target_vertex(half_edge_idx) for half_edge_idx in boundary_halfedge_verts]
        self.add_undirected_edges(boundary_node_ids, boundary_src_nodes, name="source")

        # source vertices of half edges on boundary are target vertices of boundary nodes
        boundary_target_nodes = [self.source_vertex(half_edge_idx) for half_edge_idx in boundary_halfedge_verts]
        self.add_undirected_edges(boundary_node_ids, boundary_target_nodes, name="target")

    def source_vertex(self, halfedge_index):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        return next(dst for _, dst, data in self.edges(halfedge_index, data=True) if data.get("type") == "source")

    def target_vertex(self, halfedge_index):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        return next(dst for _, dst, data in self.edges(halfedge_index, data=True) if data.get("type") == "target")

    def twin_halfedge(self, halfedge_index):
        halfedge_index = self._ensure_tag_form(halfedge_index, self.halfedge_tag)
        twin = next(target for src, target, data in self.edges(halfedge_index, data=True) if data.get("type") == "twin")
        return twin


