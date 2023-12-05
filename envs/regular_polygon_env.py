import os
from src.polygraph import PolyGraph
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
from copy import deepcopy
import pickle
import uuid


def interior_angle(polygon_degree):
    return (polygon_degree - 2) * 180 / polygon_degree


def boundary_desired_degree(polygon_degree, desired_angle):
    theta = interior_angle(polygon_degree)
    desired_degree = max(int(round(theta / desired_angle)) + 1, 2)
    return desired_degree


def generate_coordinates(polygon_degree):
    angle = 2 * np.pi / polygon_degree
    angular_increments = angle * np.arange(polygon_degree)
    x_coord = np.cos(angular_increments)
    y_coord = np.sin(angular_increments)
    coords = [[x, y] for x, y in zip(x_coord, y_coord)]
    return coords


def initialize_graph_and_desired_degree(polygon_degree, desired_angle):
    desired_degree = boundary_desired_degree(polygon_degree, desired_angle)
    coordinates = generate_coordinates(polygon_degree)

    node_ids = list(range(polygon_degree))

    face_loop = [node_ids]
    coordinates = dict(zip(node_ids, coordinates))

    graph = PolyGraph.from_face_loops(face_loop, coordinates)
    desired_degree = dict(zip(node_ids, polygon_degree * [desired_degree]))

    return graph, desired_degree


class RegularPolygonEnv(gym.Env):
    def __init__(
            self,
            polygon_degree,
            template_size,
            max_substeps,
            max_steps,
            incremental_reward,
            logdir,
            max_edge_addition_steps=3,
            face_reward_weight=1,
            vertex_reward_weight=1
    ):
        super().__init__()
        self.polygon_degree = polygon_degree
        self.template_size = template_size
        self.max_substeps = max_substeps
        self.max_steps = max_steps
        self.incremental_reward = incremental_reward
        self.logdir = logdir

        self.max_edge_addition_steps = max_edge_addition_steps
        self.num_actions_per_halfedge = self.get_num_actions_per_halfedge(self.max_edge_addition_steps)
        self.total_num_actions_in_template = self.template_size * self.num_actions_per_halfedge

        # Assuming we are working with triangles
        self.face_desired_degree = 3
        self.interior_vertex_desired_degree = 6
        self.boundary_vertex_desired_degree = 4
        self.desired_angle = interior_angle(self.face_desired_degree)

        graph, desired_degree = initialize_graph_and_desired_degree(self.polygon_degree, self.desired_angle)
        self.graph = graph
        self.vertex_desired_degree = desired_degree

        self.num_substeps = 0
        self.num_steps = 0
        self.reward = 0
        self.face_reward_weight = face_reward_weight
        self.vertex_reward_weight = vertex_reward_weight

        self.face_score = self.global_face_score()
        self.vertex_score = self.global_vertex_score()
        self.score = self.face_score + self.vertex_score
        self.initial_score = self.score
        self.initial_face_score = self.face_score
        self.initial_vertex_score = self.vertex_score
        # since we use an action mask, invalid actions should not be selected
        # but we use this reward in case they are accidentally selected for some reason
        self.no_action_reward = -4

        # Attributes for excpetion handling
        self.initial_graph = deepcopy(self.graph)
        self.exception_occurred = False
        self.exception_count = 0
        self.action_sequence = []
        if logdir is None:
            logdir = os.getcwd()
        self.logdir = logdir

        self.template_center = self._select_halfedge_template_center(self.graph.halfedge_list())
        self._build_template()

        self.template_boundary_index = -2
        self.geometric_boundary_index = -1
        self.num_features = self.get_feature_size()

        self.action_space = Discrete(self.num_actions_per_halfedge)
        self.observation_space = Dict(
            {
                "features": Box(low=0, high=10, shape=(self.template_size, self.num_features)),
                "next": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "previous": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "twin": Box(low=-2, high=self.template_size, shape=(self.template_size,), dtype=np.int64),
                "mask": Box(low=-np.inf, high=0, shape=(self.total_num_actions_in_template,)),
                "progress": Box(low=0, high=1, shape=(1,))
            }
        )

    @staticmethod
    def get_feature_size():
        return 5

    @staticmethod
    def get_num_actions_per_halfedge(max_edge_addition_steps):
        # 3 more actions are - delete edge, insert vertex, delete vertex
        return max_edge_addition_steps + 3

    @classmethod
    def from_config(cls, config):
        polygon_degree = config["polygon_degree"]
        template_size = config["template_size"]
        max_substeps = config["max_substeps"]
        max_steps = config["max_steps"]
        incremental_reward = config["incremental_reward"]
        max_edge_addition_steps = config.get("max_edge_addition_steps", 3)
        logdir = config.get("logdir", None)

        return cls(
            polygon_degree,
            template_size,
            max_substeps,
            max_steps,
            incremental_reward,
            logdir=logdir,
            max_edge_addition_steps=max_edge_addition_steps
        )

    def _vertex_score(self, vidx):
        vertex_degree = self.graph.vertex_degree(vidx)
        return abs(vertex_degree - self.vertex_desired_degree[vidx])

    def _face_score(self, fidx):
        face_degree = self.graph.face_degree(fidx)
        return abs(face_degree - self.face_desired_degree)

    def _halfedge_score(self, hidx):
        source_vertex = self.graph.source_vertex(hidx, tag=False)
        vertex_score = self._vertex_score(source_vertex)
        face = self.graph.face(hidx)
        face_score = self._face_score(face)
        return vertex_score + face_score

    def global_face_score(self):
        faces = self.graph.face_list(tag=False)
        face_score = sum(self._face_score(fidx) for fidx in faces)
        return face_score

    def global_vertex_score(self):
        vertices = self.graph.vertex_list(tag=False)
        vertex_score = sum(self._vertex_score(vidx) for vidx in vertices)
        return vertex_score

    def global_score(self):
        vertex_score = self.global_vertex_score()
        face_score = self.global_face_score()

        return face_score + vertex_score

    def _select_halfedge_template_center(self, halfedges):
        halfedge_score = np.array([self._halfedge_score(hidx) for hidx in halfedges])
        max_indices = np.nonzero(halfedge_score == halfedge_score.max())[0]
        template_center_idx = np.random.choice(max_indices)
        return halfedges[template_center_idx]

    def _build_template(self):
        self.index_to_halfedge = self.graph.knn_halfedges(self.template_center, self.template_size)
        self.halfedge_to_index = {halfedge: idx for idx, halfedge in enumerate(self.index_to_halfedge)}

    def _get_feature_matrix(self):
        matrix = np.zeros((self.template_size, self.num_features), dtype=np.float32)
        for order, hidx in enumerate(self.index_to_halfedge):
            vidx = self.graph.source_vertex(hidx, tag=False)
            vertex_degree = self.graph.vertex_degree(vidx)
            vertex_desired_degree = self.vertex_desired_degree[vidx]
            is_user_defined_vertex = 1.0 if self.graph.is_user_defined_vertex(vidx) else 0.0

            fidx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(fidx)

            matrix[order, :] = [
                vertex_degree / vertex_desired_degree,
                vertex_desired_degree,
                face_degree / self.face_desired_degree,
                self.face_desired_degree,
                is_user_defined_vertex
            ]

        return matrix

    def _get_next_edges(self):
        next_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_halfedge):
            next_hidx = self.graph.next_halfedge(hidx)
            next_idx = self.halfedge_to_index.get(next_hidx, self.template_boundary_index)
            next_edges[idx] = next_idx
        return next_edges

    def _get_previous_edges(self):
        prev_edges = np.arange(self.template_size)
        for (idx, hidx) in enumerate(self.index_to_halfedge):
            prev_hidx = self.graph.previous_halfedge(hidx)
            prev_idx = self.halfedge_to_index.get(prev_hidx, self.template_boundary_index)
            prev_edges[idx] = prev_idx
        return prev_edges

    def _get_twin_edges(self):
        twin_edges = np.arange(self.template_size)
        for idx, hidx in enumerate(self.index_to_halfedge):
            if self.graph.halfedge_on_boundary(hidx):
                twin_edges[idx] = self.geometric_boundary_index
            else:
                twin_hidx = self.graph.twin_halfedge(hidx)
                twin_idx = self.halfedge_to_index.get(twin_hidx, self.template_boundary_index)
                twin_edges[idx] = twin_idx
        return twin_edges

    def _linear_action_index_to_halfedge_and_action(self, linear_action_index):
        hidx = linear_action_index // self.num_actions_per_halfedge
        local_action_index = linear_action_index % self.num_actions_per_halfedge

        if hidx < len(self.index_to_halfedge):
            halfedge = self.index_to_halfedge[hidx]
        else:
            halfedge = None

        return halfedge, local_action_index

    def _get_action_mask(self):
        total_num_actions = self.total_num_actions_in_template
        action_mask = np.array([0 if self.is_valid_action(idx) else -np.inf for idx in range(total_num_actions)],
                               dtype=np.float32)
        return action_mask

    def is_valid_action(self, linear_action_index):
        halfedge, local_action_index = self._linear_action_index_to_halfedge_and_action(linear_action_index)

        if not self.graph.is_halfedge(halfedge):
            return False

        if local_action_index < self.max_edge_addition_steps:
            if not self.graph.is_valid_edge_insert(halfedge, local_action_index + 1):
                return False

        if local_action_index == self.max_edge_addition_steps:
            if not self.graph.is_valid_delete_halfedge(halfedge):
                return False

        if local_action_index == self.num_actions_per_halfedge - 1:
            if not self.graph.is_valid_delete_source_vertex(halfedge):
                return False

        return True

    def _get_progress(self):
        return np.array([self.num_substeps / self.max_substeps])

    def _get_blank_obs(self):
        features = np.zeros((self.template_size, self.num_features))
        next_edges = np.arange(self.template_size)
        prev_edges = np.arange(self.template_size)
        twin_edges = np.arange(self.template_size)
        mask = np.zeros(self.total_num_actions_in_template)
        obs = {
            "features": features,
            "next": next_edges,
            "previous": prev_edges,
            "twin": twin_edges,
            "mask": mask,
            "progress": np.array([0.0])
        }
        return obs

    def _get_obs(self):
        if self.exception_occurred:
            return self._get_blank_obs()
        else:
            obs = {
                "features": self._get_feature_matrix(),
                "next": self._get_next_edges(),
                "previous": self._get_previous_edges(),
                "twin": self._get_twin_edges(),
                "mask": self._get_action_mask(),
                "progress": self._get_progress()
            }
            return obs

    def _update_scores_and_reward(self, face_reward, vertex_reward):
        self.face_score -= face_reward
        self.vertex_score -= vertex_reward
        self.score = self.face_score + self.vertex_score
        self.reward = self.face_reward_weight * face_reward + self.vertex_reward_weight * vertex_reward

    def _step_insert_edge(self, hidx, num_steps):
        if self.graph.is_valid_edge_insert(hidx, num_steps):
            original_face_idx = self.graph.face(hidx)
            original_face_degree = self.graph.face_degree(original_face_idx)

            new_face_idx = self.graph.insert_halfedge(hidx, num_steps)

            original_face_new_degree = self.graph.face_degree(original_face_idx)
            new_face_degree = self.graph.face_degree(new_face_idx)
            face_reward = abs(original_face_degree - self.face_desired_degree) - \
                          (abs(original_face_new_degree - self.face_desired_degree) +
                           abs(new_face_degree - self.face_desired_degree))

            source_vertex = self.graph.source_vertex(hidx, tag=False)
            prev_halfedge = self.graph.previous_halfedge(hidx)
            prev_source_vertex = self.graph.source_vertex(prev_halfedge, tag=False)

            source_vertex_degree = self.graph.vertex_degree(source_vertex)
            prev_source_vertex_degree = self.graph.vertex_degree(prev_source_vertex)

            source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]
            prev_source_vertex_desired_degree = self.vertex_desired_degree[prev_source_vertex]
            vertex_reward = (abs(source_vertex_degree - 1 - source_vertex_desired_degree) +
                             abs(prev_source_vertex_degree - 1 - prev_source_vertex_desired_degree)) - \
                            (abs(source_vertex_degree - source_vertex_desired_degree) +
                             abs(prev_source_vertex_degree - prev_source_vertex_desired_degree))

            self._update_scores_and_reward(face_reward, vertex_reward)
        else:
            self.reward = self.no_action_reward

    def _step_insert_vertex(self, hidx):
        if self.graph.halfedge_on_boundary(hidx):
            face_idx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(face_idx)
            new_vertex_idx = self.graph.insert_vertex(hidx, tag=False)
            self.vertex_desired_degree[new_vertex_idx] = self.boundary_vertex_desired_degree

            face_reward = abs(face_degree - self.face_desired_degree) - abs(face_degree + 1 - self.face_desired_degree)
            vertex_reward = 0 - abs(2 - self.boundary_vertex_desired_degree)
        else:
            face_idx = self.graph.face(hidx)
            face_degree = self.graph.face_degree(face_idx)

            twin_hidx = self.graph.twin_halfedge(hidx)
            twin_face = self.graph.face(twin_hidx)
            twin_face_degree = self.graph.face_degree(twin_face)

            new_vertex_idx = self.graph.insert_vertex(hidx, tag=False)
            self.vertex_desired_degree[new_vertex_idx] = self.interior_vertex_desired_degree

            face_reward = (abs(face_degree - self.face_desired_degree) +
                           abs(twin_face_degree - self.face_desired_degree)) - \
                          (abs(face_degree + 1 - self.face_desired_degree) +
                           abs(twin_face_degree + 1 - self.face_desired_degree))
            vertex_reward = 0 - abs(2 - self.interior_vertex_desired_degree)

        self._update_scores_and_reward(face_reward, vertex_reward)

    def _step_delete_edge(self, hidx):
        if self.graph.is_valid_delete_halfedge(hidx):
            face = self.graph.face(hidx)
            twin_hidx = self.graph.twin_halfedge(hidx)
            twin_face = self.graph.face(twin_hidx)

            source_vertex = self.graph.source_vertex(hidx, tag=False)
            target_vertex = self.graph.target_vertex(hidx, tag=False)

            current_face_degree = self.graph.face_degree(face)
            twin_face_degree = self.graph.face_degree(twin_face)

            source_vertex_degree = self.graph.vertex_degree(source_vertex)
            target_vertex_degree = self.graph.vertex_degree(target_vertex)

            source_desired_degree = self.vertex_desired_degree[source_vertex]
            target_desired_degree = self.vertex_desired_degree[target_vertex]

            self.graph.delete_halfedge(hidx)

            face_reward = (abs(current_face_degree - self.face_desired_degree) +
                           abs(twin_face_degree - self.face_desired_degree)) - \
                          (abs(current_face_degree + twin_face_degree - 2 - self.face_desired_degree))

            vertex_reward = (abs(source_vertex_degree - source_desired_degree) +
                             abs(target_vertex_degree - target_desired_degree)) - \
                            (abs(source_vertex_degree - 1 - source_desired_degree) +
                             abs(target_vertex_degree - 1 - target_desired_degree))
            self._update_scores_and_reward(face_reward, vertex_reward)

        else:
            self.reward = self.no_action_reward

    def _step_delete_source_vertex(self, hidx):
        if self.graph.is_valid_delete_source_vertex(hidx):
            if self.graph.halfedge_on_boundary(hidx):
                face_idx = self.graph.face(hidx)
                source_vertex = self.graph.source_vertex(hidx, tag=False)

                face_degree = self.graph.face_degree(face_idx)
                source_vertex_degree = self.graph.vertex_degree(source_vertex)
                source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]

                self.graph.delete_source_vertex(hidx)

                face_reward = abs(face_degree - self.face_desired_degree) - (
                    abs(face_degree - 1 - self.face_desired_degree))
                vertex_reward = abs(source_vertex_degree - source_vertex_desired_degree)

            else:
                face_idx = self.graph.face(hidx)
                twin_halfedge = self.graph.twin_halfedge(hidx)
                source_vertex = self.graph.source_vertex(hidx, tag=False)
                twin_face = self.graph.face(twin_halfedge)

                face_degree = self.graph.face_degree(face_idx)
                twin_face_degree = self.graph.face_degree(twin_face)
                source_vertex_degree = self.graph.vertex_degree(source_vertex)
                source_vertex_desired_degree = self.vertex_desired_degree[source_vertex]

                self.graph.delete_source_vertex(hidx)

                face_reward = (abs(face_degree - self.face_desired_degree) +
                               abs(twin_face_degree - self.face_desired_degree)) - \
                              (abs(face_degree - 1 - self.face_desired_degree) +
                               abs(twin_face_degree - 1 - self.face_desired_degree))
                vertex_reward = abs(source_vertex_degree - source_vertex_desired_degree)

            self._update_scores_and_reward(face_reward, vertex_reward)
            self.vertex_desired_degree.pop(source_vertex)
        # if action is invalid, update reward to no_action_reward
        else:
            self.reward = self.no_action_reward

    def _step_halfedge_action(self, halfedge, action):
        assert 0 <= action < self.num_actions_per_halfedge

        self.action_sequence.append((halfedge, action))

        if action < self.max_edge_addition_steps:
            self._step_insert_edge(halfedge, action + 1)
        elif action == self.max_edge_addition_steps:
            self._step_delete_edge(halfedge)
        elif action == self.max_edge_addition_steps + 1:
            self._step_insert_vertex(halfedge)
        else:  # delete vertex
            self._step_delete_source_vertex(halfedge)

    def _update_halfedge_template_center(self):
        # if the previous template center does not exist (i.e. it was deleted in the previous step),
        # randomly select between next/previous halfedge in the current template
        if not self.graph.is_halfedge(self.template_center):
            idx = np.random.choice([1, 2])
            self.template_center = self.index_to_halfedge[idx]
            assert self.graph.is_halfedge(self.template_center)

    def is_terminated(self):
        if self.num_substeps >= self.max_substeps:
            return True

        template_score = sum(self._halfedge_score(hidx) for hidx in self.index_to_halfedge)
        if template_score == 0:
            return True

        if self.exception_occurred:
            return True

        return False

    def step(self, linear_action_index):

        if self.num_substeps >= self.max_substeps:
            print("WARNING : NUM SUBSTEPS > MAX SUBSTEPS!!")  # this should not happen

        halfedge, local_action = self._linear_action_index_to_halfedge_and_action(linear_action_index)

        try:
            self._step_halfedge_action(halfedge, local_action)
            self.num_substeps += 1
            # update the template center after step
            self._update_halfedge_template_center()
            self._build_template()
            terminated = self.is_terminated()
            observation = self._get_obs()
        except Exception as e:
            self.exception_occurred = True
            print("\n\n\tENCOUNTERED ENVIRONMENT EXCPETION\n\n")
            self._log_exception()
            terminated = True
            observation = self._get_obs()


        if not self.incremental_reward:
            if not terminated:
                self.reward = 0
            else:
                self.reward = self.face_reward_weight * (self.initial_face_score - self.face_score) + \
                              self.vertex_reward_weight * (self.initial_vertex_score - self.vertex_score)

        return observation, self.reward, terminated, False, {"score": self.score}

    def _soft_reset(self):
        self.num_steps += 1
        self.template_center = self._select_halfedge_template_center(self.graph.halfedge_list())
        self._build_template()
        self.reward = 0
        self.num_substeps = 0
        obs = self._get_obs()
        return obs, {"score": self.score}

    def _hard_reset(self):
        graph, desired_degree = initialize_graph_and_desired_degree(self.polygon_degree, self.desired_angle)
        self.graph = graph
        self.vertex_desired_degree = desired_degree

        self.initial_graph = deepcopy(self.graph)
        self.action_sequence = []
        self.exception_occurred = False

        self.template_center = self._select_halfedge_template_center(self.graph.halfedge_list())
        self._build_template()

        self.num_substeps = 0
        self.num_steps = 0

        self.face_score = self.global_face_score()
        self.vertex_score = self.global_vertex_score()
        self.score = self.face_score + self.vertex_score
        self.initial_score = self.score
        self.initial_face_score = self.face_score
        self.initial_vertex_score = self.vertex_score

        obs = self._get_obs()
        return obs, {"score": self.score}

    def reset(self, seed=None, options=None):
        if self.exception_occurred:
            return self._hard_reset()

        if self.num_steps >= self.max_steps:
            return self._hard_reset()

        if self.score == 0:
            return self._hard_reset()

        return self._soft_reset()

    def _log_exception(self):
        exception_filename = str(uuid.uuid4()) + ".pkl"
        self.exception_count += 1
        exception_filepath = os.path.join(self.logdir, exception_filename)
        output_data = {"graph": self.initial_graph, "actions": self.action_sequence}
        with open(exception_filepath, "wb") as output_file:
            pickle.dump(output_data, output_file)

        print("\n\n\tLOGGED EXCEPTED ENV TO : ", exception_filepath, "\n\n\t")
