"""Interactive mesh-editing game server.

Wraps src.tiler.Tiler with a small JSON API and serves a single-page UI.
Run from the repo root (or anywhere):

    venv/bin/python game/server.py
"""

import json
import math
import sys
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import envs.polygon_utils as utils  # noqa: E402
from envs.environment_initializers import LEnv, SquareHole  # noqa: E402
from src.tiler import Tiler  # noqa: E402

TARGET_ANGLE = 90
FACE_DESIRED = 4
# Minimum scaled Jacobian (sin of corner angle) required to win: rules out
# near-degenerate elements with corner angles outside roughly (24, 156) degrees.
QUALITY_THRESHOLD = 0.4
INTERIOR_DESIRED = utils.rounded_desired_degree(360, TARGET_ANGLE) - 1  # 4
BOUNDARY_DESIRED = utils.rounded_desired_degree(180, TARGET_ANGLE)  # 3


def regular_polygon(n, phase=0.0):
    return [
        (math.cos(phase + 2 * math.pi * k / n), math.sin(phase + 2 * math.pi * k / n))
        for k in range(n)
    ]


def star_polygon():
    """Five-pointed star: tips at radius 1, reflex notches at pentagram radius."""
    inner = math.cos(math.radians(72)) / math.cos(math.radians(36))
    pts = []
    for k in range(5):
        tip = math.radians(90 + 72 * k)
        notch = math.radians(126 + 72 * k)
        pts.append((math.cos(tip), math.sin(tip)))
        pts.append((inner * math.cos(notch), inner * math.sin(notch)))
    return pts


def semicircle_polygon():
    """Base corners plus a 5-point arc (150-degree corners, demotable)."""
    pts = [(-1.0, 0.0), (1.0, 0.0)]
    for k in range(1, 6):
        t = math.radians(30 * k)
        pts.append((math.cos(t), math.sin(t)))
    return pts


def pacman_polygon():
    """Disk with a 90-degree wedge bite: reflex center + 7-point arc."""
    pts = [(0.0, 0.0)]
    for k in range(7):
        t = math.radians(45 + 45 * k)
        pts.append((math.cos(t), math.sin(t)))
    return pts


def gear_polygon(teeth=6, outer=1.0, root=0.62):
    """Square-wave gear: convex tooth corners and reflex roots."""
    pts = []
    w = 2 * math.pi / teeth
    for k in range(teeth):
        base = w * k
        for frac, rad in [(0.05, outer), (0.45, outer), (0.55, root), (0.95, root)]:
            t = base + frac * w
            pts.append((rad * math.cos(t), rad * math.sin(t)))
    return pts


def triforce_ring():
    """Triangle with a same-orientation triangular hole (chi = 0, par 0).

    The slit joining hole to rim runs between two mid-edge points, which both
    want degree 3 and so are already satisfied by the slit itself: unlike a
    corner-to-corner slit, this one is a legitimate edge of a perfect mesh and
    never has to be deleted.
    """
    outer = [
        (math.cos(math.radians(90 + 120 * k)), math.sin(math.radians(90 + 120 * k)))
        for k in range(3)
    ]
    hole = [
        (0.4 * math.cos(math.radians(90 + 120 * k)),
         0.4 * math.sin(math.radians(90 + 120 * k)))
        for k in range(3)
    ]
    coords = dict(enumerate([list(p) for p in outer] + [list(p) for p in hole]))
    # slit endpoints: midpoints of outer edge 0-1 and of inner edge 3-4
    coords[6] = [(outer[0][0] + outer[1][0]) / 2, (outer[0][1] + outer[1][1]) / 2]
    coords[7] = [(hole[0][0] + hole[1][0]) / 2, (hole[0][1] + hole[1][1]) / 2]
    loop = [[0, 6, 7, 3, 5, 4, 7, 6, 1, 2]]
    graph = Tiler.from_face_loops(loop, coords)
    desired = {0: 2, 1: 2, 2: 2, 3: 4, 4: 4, 5: 4, 6: 3, 7: 3}
    return graph, desired


def square_hole_ring():
    """Square with a square hole; slit runs mid-edge to mid-edge (par 0).

    Same idea as triforce_ring: both slit endpoints are flat boundary points
    wanting degree 3, which the slit already provides.
    """
    coords = {
        0: [0.0, 0.0],
        1: [0.5, 0.0],
        2: [0.5, 0.25],
        3: [0.25, 0.25],
        4: [0.25, 0.75],
        5: [0.75, 0.75],
        6: [0.75, 0.25],
        7: [1.0, 0.0],
        8: [1.0, 1.0],
        9: [0.0, 1.0],
    }
    loop = [[0, 1, 2, 3, 4, 5, 6, 2, 1, 7, 8, 9]]
    graph = Tiler.from_face_loops(loop, coords)
    desired = dict(zip(range(10), [2, 3, 3, 4, 4, 4, 4, 2, 2, 2]))
    return graph, desired


def simple_polygon(coords):
    """Build an initializer from a CCW simple-polygon outline."""

    def make():
        coord_dict = dict(zip(range(len(coords)), [list(c) for c in coords]))
        loop = [list(range(len(coords)))]
        graph = Tiler.from_face_loops(loop, coord_dict)
        angles = utils.get_polygon_interior_angles(loop[0], graph.vertex_coordinates)
        desired = {
            v: utils.rounded_desired_degree(a, TARGET_ANGLE) for v, a in angles.items()
        }
        return graph, desired

    return make


INITIALIZERS = {
    "L-shape": lambda: LEnv(TARGET_ANGLE)(),
    "T-bracket": simple_polygon(
        [(1, 0), (2, 0), (2, 2), (3, 2), (3, 3), (0, 3), (0, 2), (1, 2)]
    ),
    "I-bracket": simple_polygon(
        [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (3, 2),
         (3, 3), (0, 3), (0, 2), (1, 2), (1, 1), (0, 1)]
    ),
    "U-channel": simple_polygon(
        [(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)]
    ),
    "Z-shape": simple_polygon(
        [(0, 0), (2, 0), (2, 1), (3, 1), (3, 2), (1, 2), (1, 1), (0, 1)]
    ),
    "Plus": simple_polygon(
        [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2),
         (2, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1)]
    ),
    "Staircase": simple_polygon(
        [(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2), (1, 3), (0, 3)]
    ),
    "Triangle": simple_polygon([(0, 0), (1, 0), (0.5, math.sin(math.pi / 3))]),
    "Pentagon": simple_polygon(regular_polygon(5, math.pi / 2)),
    "Semicircle": simple_polygon(semicircle_polygon()),
    "Pac-Man": simple_polygon(pacman_polygon()),
    "Star": simple_polygon(star_polygon()),
    "Square hole": square_hole_ring,
    "Triforce ring": triforce_ring,
    "Gear": simple_polygon(gear_polygon()),
}

PORT = 8123
STATIC_DIR = Path(__file__).resolve().parent / "static"
RECORDS_PATH = Path(__file__).resolve().parent / "records.json"


def load_records():
    try:
        return json.loads(RECORDS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_records(records):
    RECORDS_PATH.write_text(json.dumps(records, indent=2))


class GameError(Exception):
    pass


class Game:
    def __init__(self, shape="L-shape"):
        self.shape = shape
        self.reset(shape)

    def reset(self, shape=None):
        shape = shape or self.shape
        if shape not in INITIALIZERS:
            raise GameError(f"Unknown shape: {shape}")
        graph, desired = INITIALIZERS[shape]()
        self.shape = shape
        self.graph = graph
        self.initial_desired = desired
        self.undo_stack = []
        self.moves = 0
        self.records = load_records()
        self.record_just_set = False
        self.par = self._compute_par()

    def _current_scores(self):
        g = self.graph
        vertex_score = sum(
            abs(g.vertex_degree(v) - self.desired_degree(v))
            for v in g.vertex_list(tag=False)
        )
        face_score = sum(
            abs(g.face_degree(f) - FACE_DESIRED) for f in g.face_list()
        )
        min_quality = min(
            math.sin(math.radians(a)) for a in g.half_edge_angles().values()
        )
        return vertex_score, face_score, min_quality

    def _check_record(self):
        vertex_score, face_score, min_quality = self._current_scores()
        won = (
            vertex_score == self.par
            and face_score == 0
            and min_quality >= QUALITY_THRESHOLD
        )
        if not won:
            return
        best = self.records.get(self.shape)
        if best is None or self.moves < best["moves"]:
            self.records[self.shape] = {"moves": self.moves}
            self.record_just_set = True
            save_records(self.records)

    def _compute_par(self):
        """Topological lower bound on the vertex score for an all-quad mesh.

        For any all-quad mesh of this domain, discrete Gauss-Bonnet gives
        sum(generic_degree - degree) = 4*chi with generic degree 3 on the
        boundary and 4 in the interior. Hence sum(desired - degree) is the
        invariant C + 4*chi, and the L1 vertex score is at least its absolute
        value.
        """
        g = self.graph
        corner_excess = sum(
            self.desired_degree(vid)
            - (BOUNDARY_DESIRED if g.is_boundary_vertex(vid) else INTERIOR_DESIRED)
            for vid in g.vertex_list(tag=False)
        )
        num_edges = 0
        seen = set()
        for h in g.half_edge_list():
            if h in seen:
                continue
            seen.add(h)
            seen.add(g.twin_half_edge(h))
            num_edges += 1
        chi = len(g.vertex_list()) - num_edges + len(g.face_list())
        return abs(corner_excess + 4 * chi)

    def desired_degree(self, vid):
        if vid in self.initial_desired:
            return self.initial_desired[vid]
        if self.graph.is_boundary_vertex(vid):
            return BOUNDARY_DESIRED
        return INTERIOR_DESIRED

    def halfedge(self, edge_id):
        h = (edge_id, self.graph.half_edge_tag)
        if not self.graph.is_half_edge(h):
            raise GameError(f"Unknown edge: {edge_id}")
        return h

    def delete_candidates(self, vid):
        g = self.graph
        vt = (vid, g.vertex_tag)
        for h in list(g._vertex_to_halfedge(vid)):
            if g.half_edges[h].source == vt and g.is_valid_delete_source_vertex(h):
                yield h

    def resolve_chord(self, a, b):
        """Find (halfedge, k) such that insert_half_edge connects vertex a to b."""
        g = self.graph
        for f in g.face_list():
            loop = g.generate_half_edge_face_loop(g.first_face_halfedge(f))
            srcs = [g.source_vertex(h, tag=False) for h in loop]
            n = len(loop)
            for i, h in enumerate(loop):
                if srcs[i] != a:
                    continue
                for k in range(n - 1):
                    if srcs[(i + k + 1) % n] == b and g.is_valid_edge_insert(h, k):
                        return h, k
        return None

    def apply_op(self, op, params, smooth_after=False):
        snapshot = deepcopy(self.graph)
        try:
            if op == "insert_vertex":
                h = self.halfedge(int(params["edge"]))
                self.graph.insert_vertex(h)
            elif op == "delete_edge":
                h = self.halfedge(int(params["edge"]))
                if not self.graph.is_valid_delete_half_edge(h):
                    raise GameError("This edge cannot be deleted")
                self.graph.delete_half_edge(h)
            elif op == "delete_vertex":
                vid = int(params["vertex"])
                h = next(self.delete_candidates(vid), None)
                if h is None:
                    raise GameError(
                        "Vertex not deletable: must be degree 2 and not an original corner"
                    )
                self.graph.delete_source_vertex(h)
            elif op == "insert_edge":
                a, b = int(params["a"]), int(params["b"])
                chord = self.resolve_chord(a, b) or self.resolve_chord(b, a)
                if chord is None:
                    raise GameError("Vertices must lie on the same face")
                self.graph.insert_half_edge(*chord)
            else:
                raise GameError(f"Unknown operation: {op}")

            if smooth_after:
                self.graph.smooth_vertices(num_iter=2)
        except GameError:
            self.graph = snapshot
            raise
        except Exception as exc:
            self.graph = snapshot
            raise GameError(str(exc) or "Operation not permitted")

        self.undo_stack.append(snapshot)
        self.moves += 1
        self.record_just_set = False
        self._check_record()

    def smooth(self, num_iter=3):
        self.undo_stack.append(deepcopy(self.graph))
        self.graph.smooth_vertices(num_iter=num_iter)
        self.record_just_set = False
        self._check_record()

    def undo(self):
        if not self.undo_stack:
            raise GameError("Nothing to undo")
        self.graph = self.undo_stack.pop()
        self.moves = max(0, self.moves - 1)
        self.record_just_set = False

    def serialize(self):
        g = self.graph

        # scaled Jacobian per corner: sin of the interior angle at each
        # half-edge source. 1 = right angle, 0 = flat/collapsed, < 0 = concave.
        angles = g.half_edge_angles()
        corner_quality = {h: math.sin(math.radians(a)) for h, a in angles.items()}
        vertex_quality = {}
        face_quality = {}
        for h, q in corner_quality.items():
            vid = g.source_vertex(h, tag=False)
            fid = g.face(h, tag=False)
            vertex_quality[vid] = min(q, vertex_quality.get(vid, 1.0))
            face_quality[fid] = min(q, face_quality.get(fid, 1.0))
        min_quality = min(corner_quality.values())

        vertices = []
        for vid in g.vertex_list(tag=False):
            coord = g.vertex_coordinates[vid]
            vertices.append(
                {
                    "id": vid,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "degree": int(g.vertex_degree(vid)),
                    "desired": int(self.desired_degree(vid)),
                    "boundary": bool(g.is_boundary_vertex(vid)),
                    "user": bool(g.is_user_defined_vertex(vid)),
                    "deletable": next(self.delete_candidates(vid), None) is not None,
                    "quality": round(float(vertex_quality.get(vid, 1.0)), 3),
                }
            )

        edges = []
        seen = set()
        for h in g.half_edge_list():
            if h in seen:
                continue
            seen.add(h)
            seen.add(g.twin_half_edge(h))
            edges.append(
                {
                    "id": h[0],
                    "v1": g.source_vertex(h, tag=False),
                    "v2": g.target_vertex(h, tag=False),
                    "boundary": bool(g.half_edge_on_boundary(h)),
                    "deletable": bool(g.is_valid_delete_half_edge(h)),
                }
            )

        faces = []
        insertable_pairs = []
        for f in g.face_list():
            loop = g.generate_half_edge_face_loop(g.first_face_halfedge(f))
            srcs = [g.source_vertex(h, tag=False) for h in loop]
            faces.append(
                {
                    "id": f[0],
                    "degree": int(g.face_degree(f)),
                    "vertices": srcs,
                    "quality": round(float(face_quality.get(f[0], 1.0)), 3),
                }
            )
            n = len(loop)
            for i, h in enumerate(loop):
                for k in range(n - 1):
                    b = srcs[(i + k + 1) % n]
                    if srcs[i] != b and g.is_valid_edge_insert(h, k):
                        insertable_pairs.append([srcs[i], b])

        vertex_score = sum(abs(v["degree"] - v["desired"]) for v in vertices)
        face_score = sum(abs(f["degree"] - FACE_DESIRED) for f in faces)
        angle_score = float(
            sum(abs(a - TARGET_ANGLE) / TARGET_ANGLE for a in angles.values())
        )
        defect_sum = sum(v["desired"] - v["degree"] for v in vertices)

        return {
            "shape": self.shape,
            "shapes": list(INITIALIZERS.keys()),
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "insertable_pairs": insertable_pairs,
            "scores": {
                "vertex": int(vertex_score),
                "face": int(face_score),
                "angle": round(angle_score, 2),
                "defect_sum": int(defect_sum),
                "min_quality": round(float(min_quality), 3),
            },
            "par": self.par,
            "quality_threshold": QUALITY_THRESHOLD,
            "best": (self.records.get(self.shape) or {}).get("moves"),
            "new_record": self.record_just_set,
            "moves": self.moves,
            "can_undo": bool(self.undo_stack),
        }


GAME = Game()
LOCK = Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_state(self):
        self._send_json(200, GAME.serialize())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            with LOCK:
                self._send_state()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        try:
            with LOCK:
                if self.path == "/api/op":
                    GAME.apply_op(
                        body.get("op"),
                        body.get("params", {}),
                        smooth_after=bool(body.get("smooth")),
                    )
                elif self.path == "/api/undo":
                    GAME.undo()
                elif self.path == "/api/smooth":
                    GAME.smooth(num_iter=int(body.get("iters", 3)))
                elif self.path == "/api/new":
                    GAME.reset(body.get("shape") or GAME.shape)
                else:
                    self._send_json(404, {"error": "not found"})
                    return
                self._send_state()
        except GameError as exc:
            self._send_json(400, {"error": str(exc)})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Mesh game running at http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
