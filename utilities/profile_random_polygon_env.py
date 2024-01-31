import numpy as np
import os
import sys
sys.path.append(os.getcwd())
# from envs.random_polygon_env import RandomPolygonEnv
from envs.random_polygon_tiler_env import RandomPolygonEnv
from envs.angle_env import AngleEnv
# import cProfile


def run_environment():
    face_degree = 3
    polygon_degree = 20
    template_size = 30
    max_steps_factor = 2
    num_resets = 100

    env = AngleEnv(
        face_degree,
        [polygon_degree],
        template_size,
        max_steps_factor
    )

    for step in range(num_resets):
        print("\n\nTRIAL : ", step, "\n\n")
        env.reset()
        done = env.is_terminated()
        while not done:
            print("\tNum steps : ", env.num_steps)
            obs = env._get_obs()
            mask = obs["mask"]
            candidates = np.nonzero(mask == 0)[0]
            linear_action_index = np.random.choice(candidates)

            env.step(linear_action_index)
            done = env.is_terminated()




# cProfile.run("run_environment()", sort=0)
if __name__ == "__main__":
    run_environment()
