import collections
import numpy as np
from envs.random_polygon_tiler_env import RandomPolygonEnv


def get_initial_scores(face_desired_degree, polygon_degree, num_trials=100):
    env = RandomPolygonEnv(face_desired_degree, [polygon_degree])
    face_scores = []
    vertex_scores = []
    for step in range(num_trials):
        env.reset()
        face_scores.append(env.initial_face_score)
        vertex_scores.append(env.initial_vertex_score)

    face_mean = np.mean(face_scores)
    face_std = np.std(face_scores)
    vertex_mean = np.mean(vertex_scores)
    vertex_std = np.std(vertex_scores)

    return face_mean, face_std, vertex_mean, vertex_std


def plot_stats_vs_poly_degree(poly_degrees, avg_returns, std_returns):
    avg_returns = np.array(avg_returns)
    std_returns = np.array(std_returns)
    lower = avg_returns - std_returns
    upper = avg_returns + std_returns

    fig, ax = plt.subplots()
    ax.plot(poly_degrees, avg_returns)
    ax.fill_between(poly_degrees, lower, upper, alpha=0.3)
    ax.grid()
    ax.set_xlabel("Polygon degree")
    # ax.set_ylabel("Average returns")
    # ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


face_desired_degree = 4
polygon_degree_range = range(5, 51)
stats = collections.defaultdict(list)

for polygon_degree in polygon_degree_range:
    print("Processing polygon degree : ", polygon_degree)
    fm, fs, vm, vs = get_initial_scores(face_desired_degree, polygon_degree)
    stats["face-mean"].append(fm)
    stats["face-std"].append(fs)
    stats["vertex-mean"].append(vm)
    stats["vertex-std"].append(vs)

import matplotlib.pyplot as plt
plot_stats_vs_poly_degree(polygon_degree_range, stats["face-mean"], stats["face-std"])
plot_stats_vs_poly_degree(polygon_degree_range, stats["vertex-mean"], stats["vertex-std"])