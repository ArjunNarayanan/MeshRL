from envs.random_polygon_tiler_env import RandomPolygonEnv
import numpy as np

def setUp():
    env = RandomPolygonEnv(
        3,
        [6],
        template_size=6,
    )
    for vidx, _ in env.vertex_desired_degree.items():
        env.vertex_desired_degree[vidx] = 3
    env._update_scores_on_reset()
    return env


env = setUp()
f = env._get_feature_matrix()

test_matrix = np.zeros((env.template_size, 5), dtype=np.float32)
test_matrix[:6, :] = [
    2 / 3,
    3,
    6 / 3,
    3,
    1
]
