import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os
from optuna.importance import get_param_importances
import matplotlib.pyplot as plt


def plot_parameter_importances(importances, key2name, rotation=40, fontsize=12, figsize=(12,4)):
    fig, ax = plt.subplots(figsize=figsize)
    xlabels = [key2name[k] for k, _ in importances.items()]
    vals = [v for _, v in importances.items()]
    ax.bar(range(len(xlabels)), vals)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=rotation, fontsize=fontsize)
    fig.tight_layout()
    ax.grid()

    return fig


input_folder = "experiments/tiler-random-polygon/quad/optuna/quad-convolution/"
study_name = "quad-5-50-conv"
journal_file_path = os.path.join(input_folder, study_name + ".log")
journal_file = os.path.basename(journal_file_path)

output_folder = os.path.join(input_folder, "figures")
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

storage = JournalStorage(JournalFileStorage(journal_file_path))
study = optuna.load_study(study_name=study_name, storage=storage)

importances = get_param_importances(study)
key2name = dict(
    ent_coef="$c_e$",
    gae_lambda="$\\lambda$",
    vf_coef="$c_v$",
    lr="$\\eta$",
    clip_range="$\\epsilon$",
    max_grad_norm="Gradient\nclip norm",
    ortho_init="Orthogonal\ninitialization",
    feature_extractor_layers="Num. conv.\nlayers",
    feature_extractor_size="Num. conv.\nchannels"
)

fig = plot_parameter_importances(importances, key2name, fontsize=16)
outputfile = os.path.join(output_folder, "param_importance.pdf")
fig.savefig(outputfile)

# fig = plot_intermediate_values(study)
# fig.show()
#
# fig = plot_optimization_history(study)
# fig.show()
#
# fig = plot_contour(study, params=["lr", "ent_coef"])
# fig.show()
#
# fig = plot_slice(study, params=["lr", "ent_coef"])
# fig.show()
