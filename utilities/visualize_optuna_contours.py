import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os
import matplotlib.pyplot as plt
import numpy as np
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


def generate_contour_plot(study, params, num_levels=10, fontsize=16, vmin=0, vmax=1):
    assert len(params) == 2
    contour_info = optuna.visualization.matplotlib._contour._get_contour_info(study, params, None, "Objective Value")
    info = contour_info.sub_plot_infos[0][0]
    cmap = plt.get_cmap("magma")

    fig, ax = plt.subplots()
    ax.set_xlabel(key2name[info.xaxis.name], fontsize=fontsize)
    ax.set_ylabel(key2name[info.yaxis.name], fontsize=fontsize)
    (
        xi,
        yi,
        zi,
        x_cat_param_pos,
        x_cat_param_label,
        y_cat_param_pos,
        y_cat_param_label,
        feasible_plot_values,
        infeasible_plot_values,
    ) = optuna.visualization.matplotlib._contour._calculate_griddata(info)
    if info.xaxis.is_log:
        ax.set_xscale("log")
    if info.yaxis.is_log:
        ax.set_yscale("log")

    levels = np.linspace(vmin, vmax, num_levels)
    ax.contour(xi, yi, zi, levels, linewidths=0.5, colors="k")
    cs = ax.contourf(xi, yi, zi, levels, cmap=cmap)
    ax.scatter(
        feasible_plot_values.x,
        feasible_plot_values.y,
        marker="o",
        c="black",
        s=20,
        edgecolors="grey",
        linewidth=2.0,
    )
    # ax.label_outer()
    axcb = fig.colorbar(cs)
    axcb.set_label("Average return", fontsize=fontsize)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="input directoru", required=True)
    parser.add_argument("-study", help="study name", required=True)
    parser.add_argument("-param1", help="first parameter to plot", required=True)
    parser.add_argument("-param2", help="second parameter to plot", required=True)
    parser.add_argument("-levels", help="num. contour levels", default=10, type=int)
    parser.add_argument("-vmin", help="contour min value", default=0.0, type=float)
    parser.add_argument("-vmax", help="contour max value", default=1.0, type=float)
    parser.add_argument("-output", help="output sub-folder", default="figures")
    args = parser.parse_args()

    input_folder = args.input
    study_name = args.study
    journal_file = study_name + ".log"
    journal_file_path = os.path.join(input_folder, journal_file)

    output_folder = os.path.join(input_folder, args.output)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    storage = JournalStorage(JournalFileStorage(journal_file_path))
    study = optuna.load_study(study_name=study_name, storage=storage)

    params = [args.param1, args.param2]
    fig = generate_contour_plot(study, params, num_levels=args.levels, vmin=args.vmin, vmax=args.vmax)

    outputfile = args.param1 + "_vs_" + args.param2 + ".pdf"
    filepath = os.path.join(output_folder, outputfile)
    fig.savefig(filepath)
