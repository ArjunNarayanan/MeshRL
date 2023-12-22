import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os
from optuna.visualization import *

study_name = "quad-10-30"
storage_path = "experiments/random-polygon/quad-10-30-optuna/optuna-journal.log"
storage = JournalStorage(JournalFileStorage(storage_path))
study = optuna.load_study(study_name=study_name, storage=storage)

fig = plot_param_importances(study)
fig.show()

fig = plot_intermediate_values(study)
fig.show()

fig = plot_optimization_history(study)
fig.show()

fig = plot_contour(study, params=["lr", "ent_coef"])
fig.show()

fig = plot_slice(study, params=["lr", "ent_coef"])
fig.show()