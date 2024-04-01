import os
import sys

sys.path.append(os.getcwd())
from src.render import Renderer
from envs.environment_initializers import RandomPolygon
from envs.global_angle_env import AngleEnv
import envs.polygon_utils as utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="face desired degree", required=True, type=int)
    parser.add_argument("-dir", help="output directory", required=True)
    parser.add_argument("-num_plots", default=5)
    parser.add_argument("-min", help="min polygon degree", default=10, type=int)
    parser.add_argument("-max", help="max polygon degree", default=20, type=int)
    parser.add_argument("-rscale", help="radius scale", default=0.8, type=float)
    args = parser.parse_args()

    face_desired_degree = args.f
    output_dir = os.path.join("examples", "figures", "random-polygon", args.dir)
    num_plots = args.num_plots
    min_polygon_degree = args.min
    max_polygon_degree = args.max
    radius_scale = args.rscale

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    target_angle = utils.average_face_angle(face_desired_degree)
    polygon_degree_range = list(range(min_polygon_degree, max_polygon_degree + 1))
    init = RandomPolygon(polygon_degree_range, target_angle, scale=radius_scale)
    env = AngleEnv(face_desired_degree, init)

    for step in range(num_plots):
        env.reset()
        graph = env.graph

        renderer = Renderer(graph, graph.vertex_coordinates, vertex_size=30, fontsize=20)
        renderer.plot()
        renderer.plot_vertex_scores(env.vertex_desired_degree)
        renderer.fig.tight_layout()

        filename = "step-" + str(step) + ".pdf"
        filepath = os.path.join(output_dir, filename)
        renderer.fig.savefig(filepath)
