import collections
import numpy as np
from envs.global_angle_env import AngleEnv
from envs.environment_initializers import RandomPolygon
from envs.polygon_utils import average_face_angle
import matplotlib.pyplot as plt
import sys
import os


def get_angle_env_score(face_desired_degree, polygon_degree, num_trials=50):
    target_angle = average_face_angle(face_desired_degree)
    init = RandomPolygon([polygon_degree], target_angle, scale=0.9)
    env = AngleEnv(face_desired_degree, init)
    angle_scores = []
    face_scores = []
    vertex_scores = []
    for step in range(num_trials):
        env.reset()
        angle_scores.append(env.global_angle_score)
        face_scores.append(env.global_face_score)
        vertex_scores.append(env.global_vertex_score)

    angle_mean = np.mean(angle_scores)
    angle_std = np.std(angle_scores)
    face_mean = np.mean(face_scores)
    face_std = np.std(face_scores)
    vertex_mean = np.mean(vertex_scores)
    vertex_std = np.std(vertex_scores)

    return face_mean, face_std, vertex_mean, vertex_std, angle_mean, angle_std


def plot_stats_vs_poly_degree(ax, poly_degrees, avg_returns, std_returns, marker, label, num_marks=10):
    avg_returns = np.array(avg_returns)
    std_returns = np.array(std_returns)
    lower = avg_returns - std_returns
    upper = avg_returns + std_returns

    mark_every = len(poly_degrees) // num_marks + 1

    ax.plot(poly_degrees, avg_returns, marker=marker, color="black", markevery=mark_every, label=label)
    ax.fill_between(poly_degrees, lower, upper, alpha=0.3, color="gray")


def add_to_plots(face_desired_degree, polygon_degree_range, marker, label):
    stats = collections.defaultdict(list)

    for polygon_degree in polygon_degree_range:
        print("Processing polygon degree : ", polygon_degree)
        # fm, fs, vm, vs = get_initial_scores(face_desired_degree, polygon_degree)
        fm, fs, vm, vs, tm, ts = get_angle_env_score(face_desired_degree, polygon_degree)
        stats["face-mean"].append(fm)
        stats["face-std"].append(fs)
        stats["vertex-mean"].append(vm)
        stats["vertex-std"].append(vs)
        stats["angle-mean"].append(tm)
        stats["angle-std"].append(ts)

    plot_stats_vs_poly_degree(ax2, polygon_degree_range, stats["vertex-mean"], stats["vertex-std"], marker, label)
    plot_stats_vs_poly_degree(ax3, polygon_degree_range, stats["angle-mean"], stats["angle-std"], marker, label)


if __name__ == "__main__":
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    # print(os.getcwd())

    polygon_degree_range = range(10, 100, 4)

    ax2.set_title("Scaling of angle score")
    ax2.grid()
    ax2.set_xlabel("Polygon degree", fontsize=16)
    ax2.set_ylabel(r"$s_\theta^0$", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # ax3.set_title("Scaling of vertex score")
    ax3.grid()
    ax3.set_xlabel("Polygon degree", fontsize=16)
    ax3.set_ylabel(r"$s_v^0$", fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    add_to_plots(3, polygon_degree_range, marker="^", label="f*=3")
    add_to_plots(4, polygon_degree_range, marker="s", label="f*=4")
    add_to_plots(6, polygon_degree_range, marker="H", label="f*=6")
    ax2.legend(loc="upper left", fontsize=16)
    ax3.legend(loc="upper left", fontsize=16)

    fig2.tight_layout()
    fig3.tight_layout()

    vertex_outputfile = "examples/figures/initial-vertex-score-scaling.pdf"
    fig3.savefig(vertex_outputfile)

    angle_outputfile = "examples/figures/initial-angle-score-scaling.pdf"
    fig2.savefig(angle_outputfile)
