import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os
import matplotlib.pyplot as plt
import argparse

key2name = dict(
    gae_lambda="$\\lambda$",
    ent_coef="$c_e$",
    vf_coef="$c_v$",
    clip_range="$\\epsilon$",
    max_grad_norm="Max.\ngrad norm",
    lr="$\\eta$",
    ortho_init="Ortho.\ninit.",
    feature_extractor_layers="L",
    feature_extractor_size="d"
)


def _plot_parameter_importances(importances, key2name, rotation=0, fontsize=12, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    xlabels = [key2name[k] for k, _ in importances.items()]
    vals = [v for _, v in importances.items()]
    ax.bar(range(len(xlabels)), vals, color="salmon")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=rotation, fontsize=fontsize)
    ax.set_ylabel("Importance", fontsize=fontsize)
    fig.tight_layout()
    ax.grid()

    return fig


def plot_parameter_importances(study, figsize=(12, 4), fontsize=16):
    importances = optuna.importance.get_param_importances(study)
    fig = _plot_parameter_importances(importances, key2name, figsize=figsize, fontsize=fontsize)
    return fig


def plot_intermediate_values(study, figsize=(8, 4), fontsize=16, ylim=(-2, 1)):
    trial_infos = optuna.visualization._intermediate_values._get_intermediate_plot_info(study).trial_infos
    fig, ax = plt.subplots(figsize=figsize)
    for tinfo in trial_infos:
        x = [v[0] for v in tinfo.sorted_intermediate_values]
        y = [v[1] for v in tinfo.sorted_intermediate_values]
        ax.plot(x, y, "-o", alpha=0.5)
    ax.grid()
    ax.set_xlabel("Evaluation step", fontsize=fontsize)
    ax.set_ylabel("Average return", fontsize=fontsize)
    ax.set_ylim(ylim[0], ylim[1])
    fig.tight_layout()
    return fig


def plot_optimization_history(study, fontsize=16, ylim=(-0.5, 1), linewidth=3):
    from optuna.visualization._optimization_history import _ValueState

    info_list = optuna.visualization._optimization_history._get_optimization_history_info_list(study, None,
                                                                                               "Objective Value", False)
    trial_numbers, value_info, best_values_info = info_list[0]
    feasible_trials = [n for n, s in zip(trial_numbers, value_info.states) if s == _ValueState.Feasible]
    feasible_values = value_info.values

    fig, ax = plt.subplots()
    ax.scatter(feasible_trials, feasible_values, color="black")
    ax.plot(best_values_info.values, color="firebrick", linewidth=linewidth)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel("Average return", fontsize=fontsize)
    ax.set_xlabel("Trial", fontsize=fontsize)
    ax.grid()
    fig.tight_layout()
    return fig


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="input directoru", required=True)
    parser.add_argument("-study", help="study name", required=True)
    parser.add_argument("-output", help="output sub-folder", default="figures")
    args = parser.parse_args()

    input_folder = args.input
    study_name = args.study
    journal_file = study_name + ".log"
    journal_file_path = os.path.join(input_folder, journal_file)

    output_folder = os.path.join(input_folder, "figures")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    storage = JournalStorage(JournalFileStorage(journal_file_path))
    study = optuna.load_study(study_name=study_name, storage=storage)

    print("\nPLOTTING PARAMETER IMPORTANCES")
    fig = plot_parameter_importances(study)
    outputfile = os.path.join(output_folder, "param_importance.pdf")
    fig.savefig(outputfile)

    print("\nPLOTTING RETURN TRAJECTORIES")
    fig = plot_intermediate_values(study, figsize=(8, 4))
    filename = "return-trajectory.pdf"
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath)

    print("\nPLOTTING OPTIMIZATION HISTORY")
    fig = plot_optimization_history(study, ylim=(0,1))
    filename = "optimization-history.pdf"
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath)
