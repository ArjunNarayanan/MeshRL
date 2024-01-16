import numpy as np
import os
import sys
sys.path.append(os.getcwd())
# from envs.random_polygon_env import RandomPolygonEnv
from envs.random_polygon_tiler_env import RandomPolygonEnv
# import cProfile


def run_environment():
    face_degree = 3
    polygon_degree = 100
    template_size = 40
    max_steps_factor = 4

    env = RandomPolygonEnv(
        face_degree,
        [polygon_degree],
        template_size,
        max_steps_factor
    )

    # num_actions_per_halfedge = env.num_actions_per_halfedge
    done = env.is_terminated()

    while not done:
        obs = env._get_obs()
        mask = obs["mask"]
        candidates = np.nonzero(mask == 0)[0]
        linear_action_index = np.random.choice(candidates)

        env.step(linear_action_index)
        done = env.is_terminated()


# cProfile.run("run_environment()", sort=0)
if __name__ == "__main__":
    run_environment()
