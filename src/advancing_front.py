from src.tiler import Tiler
import numpy as np


class AdvancingFront:
    def __init__(self, tiler: Tiler):
        assert isinstance(tiler, Tiler), "tiler must be an instance of Tiler"
        self.tiler = tiler
    
    def is_valid_advance(self, hidx: int, coord: np.ndarray) -> bool:
        # Get source and target vertex indices (untagged)
        src_idx = self.tiler.source_vertex(hidx, tag=False)
        tgt_idx = self.tiler.target_vertex(hidx, tag=False)
        # Get their coordinates
        src_coord = self.tiler.vertex_coordinate(src_idx)
        tgt_coord = self.tiler.vertex_coordinate(tgt_idx)
        # Compute vectors
        v1 = tgt_coord - src_coord
        v2 = coord - src_coord
        # Compute 2D cross product (z-component)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        return cross > 0

    def advance_front(self, hidx: int, coord: np.ndarray):
        assert self.is_valid_advance(hidx, coord), "Invalid advance"

        # Insert a new half-edge between the source and target vertex
        self.tiler.insert_half_edge(next_hidx, 1)
        prev_hidx = self.tiler.previous_half_edge(next_hidx)

        
        