import numpy as np


def average_face_angle(polygon_degree):
    return (polygon_degree - 2) * 180 / polygon_degree


def angle_between(v1, v2):
    dotp = v1[0] * v2[0] + v1[1] * v2[1]
    detp = (v1[0] * v2[1] - v2[0] * v1[1])
    angle = np.degrees(np.arctan2(detp, dotp))
    if angle < 0:
        angle = angle + 360
    return angle


def generate_coordinates(polygon_degree, scale=0.5):
    assert polygon_degree >= 3
    angle = 2 * np.pi / polygon_degree
    angular_increments = angle * np.arange(polygon_degree)
    radii = (1 - scale) + scale * np.random.rand(polygon_degree)
    x_coord = np.cos(angular_increments) * radii
    y_coord = np.sin(angular_increments) * radii
    coords = [[x, y] for x, y in zip(x_coord, y_coord)]
    return coords


def get_polygon_interior_angles(face_loop, vertex_coords):
    num_verts = len(face_loop)
    angles = {}
    for idx, vcurrent in enumerate(face_loop):
        vprev = face_loop[idx - 1]
        vnext = face_loop[(idx + 1) % num_verts]

        v1 = np.array(vertex_coords[vnext]) - np.array(vertex_coords[vcurrent])
        v2 = np.array(vertex_coords[vprev]) - np.array(vertex_coords[vcurrent])
        angles[vcurrent] = angle_between(v1, v2)
    return angles


def rounded_desired_degree(angle, target_angle):
    degree = max(int(np.round(angle / target_angle)) + 1, 2)
    return degree
