import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os

journal_file_path = "experiments/tiler-random-polygon/quad/optuna/quad-convolution/quad-5-50-conv.log"
journal_file = os.path.basename(journal_file_path)
default_study_name = os.path.splitext(journal_file)[0]

study_name = default_study_name
print("Loading study at : ", journal_file_path, "\twith name : ", study_name)

storage = JournalStorage(JournalFileStorage(journal_file_path))

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    load_if_exists=True
)

trials = study.trials
idx = [t.number for t in trials]
vals = [t.value for t in trials]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(idx, vals)