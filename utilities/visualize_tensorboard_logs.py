import tensorboard as tb
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np

root_dir = "experiments/angle-env-with-length/quad/quad-20-angles/PPO_1/"
filename = "events.out.tfevents.1712170698.exp-12-58.807661.0"
filepath = os.path.join(root_dir, filename)


def extract_data(tb_log, key):
    data = tb_log.Scalars(key)
    steps = [d.step for d in data]
    value = [d.value for d in data]
    return steps, value



event_acc = EventAccumulator(filepath)
event_acc.Reload()

mean_reward = event_acc.Scalars("rollout/ep_rew_mean")
pg_loss = event_acc.Scalars("train/policy_gradient_loss")
value_loss = event_acc.Scalars("train/value_loss")
ent_loss = event_acc.Scalars("train/entropy_loss")
loss = event_acc.Scalars("train/loss")
clip_fraction = event_acc.Scalars("train/clip_fraction")
explained_variance = event_acc.Scalars("train/explained_variance")
