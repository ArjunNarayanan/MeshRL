from src.tiler import Tiler
import numpy as np
import envs.polygon_utils as utils


class LEnv:
    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [2, 0],
            [2, 1],
            [1, 1],
            [1, 2],
            [0, 2],
        ]
        coords = dict(zip(range(6), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [[0, 1, 2, 3, 4, 5]]
        desired_degree = dict(zip(range(6), [2, 2, 2, 4, 2, 2]))

        graph = Tiler.from_face_loops(loop, coords)
        return graph, desired_degree


class RandomLEnv:
    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [2, 0],
            [2, 1],
            [1, 1],
            [1, 2],
            [0, 2],
        ]
        coords = dict(zip(range(6), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [0, 1, 2, 3, 4, 5]
        desired_degree = dict(zip(range(6), [2, 2, 2, 4, 2, 2]))

        new_ids = loop.copy()
        np.random.shuffle(new_ids)
        new_desired_degree = {new_ids[idx]: desired_degree[idx] for idx in range(6)}
        new_coords = {new_ids[idx]: coords[idx] for idx in range(6)}
        new_loop = [new_ids]

        graph = Tiler.from_face_loops(new_loop, new_coords)
        return graph, new_desired_degree


class RandomPolygon:
    def __init__(self, polygon_degree_range, target_angle):
        self.polygon_degree_range = polygon_degree_range
        self.target_angle = target_angle

    def __call__(self):
        polygon_degree = np.random.choice(self.polygon_degree_range)
        coordinates = self.generate_random_coordinates(polygon_degree, self.target_angle)
        node_ids = list(range(polygon_degree))
        face_loop = [node_ids]
        coordinates = dict(zip(node_ids, coordinates))
        graph = Tiler.from_face_loops(face_loop, coordinates)
        interior_angles = utils.get_polygon_interior_angles(face_loop[0], graph.vertex_coordinates)
        desired_degree = {vidx: utils.rounded_desired_degree(angle, self.target_angle) for vidx, angle in
                          interior_angles.items()}
        return graph, desired_degree

    @staticmethod
    def generate_random_coordinates(polygon_degree, scale=0.5):
        assert polygon_degree >= 3
        angle = 2 * np.pi / polygon_degree
        angular_increments = angle * np.arange(polygon_degree)
        radii = (1 - scale) + scale * np.random.rand(polygon_degree)
        x_coord = np.cos(angular_increments) * radii
        y_coord = np.sin(angular_increments) * radii
        coords = [[x, y] for x, y in zip(x_coord, y_coord)]
        return coords


class Hexagon:
    @staticmethod
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

    def __call__(self):
        face_loops = [
            [0, 1, 2, 3, 4, 5]
        ]
        coords = self.generate_coordinates()
        graph = Tiler.from_face_loops(face_loops, coords)
        desired_degree = dict(zip(range(6), 6 * [3]))
        return graph, desired_degree


class CenterCrack:
    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0., 0.5],
            [0.25, 0.5],
            [0.5, 0.5 + 1e-9],
            [0.75, 0.5],
            [0.5, 0.5 - 1e-9]
        ]
        coords = dict(zip(range(len(coords)), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 4]
        ]
        graph = Tiler.from_face_loops(loop, coords)
        desired_degree = dict(zip(range(9), [2, 2, 2, 2, 3, 5, 3, 5, 3]))

        return graph, desired_degree


class SquareHole:
    @staticmethod
    def generate_coordinates():
        coords = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75]
        ]
        coords = dict(zip(range(len(coords)), coords))
        return coords

    def __call__(self):
        coords = self.generate_coordinates()
        loop = [
            [0, 1, 2, 3, 0, 4, 7, 6, 5, 4]
        ]
        graph = Tiler.from_face_loops(loop, coords)
        desired_degree = dict(zip(range(8), [2, 2, 2, 2, 4, 4, 4, 4]))

        return graph, desired_degree
