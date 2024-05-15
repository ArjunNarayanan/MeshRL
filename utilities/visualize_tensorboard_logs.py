import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import argparse


def extract_data(key):
    data = event_acc.Scalars(key)
    steps = [d.step for d in data]
    value = [d.value for d in data]
    return steps, value


def plot_key(key, xlabel="Training step", ylabel="", ylim=None, fontsize=16, color="black"):
    steps, value = extract_data(key)

    fig, ax = plt.subplots()
    ax.plot(steps, value, color=color)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="input file path", required=True)
    args = parser.parse_args()

    filepath = args.input
    filename = os.path.basename(filepath)

    root_dir = os.path.dirname(filepath)
    output_dir = os.path.join(root_dir, "figures")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    event_acc = EventAccumulator(filepath)
    event_acc.Reload()

    print("Plotting average returns")
    fig = plot_key("rollout/ep_rew_mean", ylim=[-0.5, 1], ylabel="Average returns")
    outputfile = os.path.join(output_dir, "average_returns.pdf")
    fig.savefig(outputfile)

    print("Plotting loss terms")
    fig = plot_key("train/policy_gradient_loss", ylabel="Policy gradient loss")
    outputfile = os.path.join(output_dir, "pg_loss.pdf")
    fig.savefig(outputfile)

    fig = plot_key("train/value_loss", ylabel="Value loss")
    outputfile = os.path.join(output_dir, "value_loss.pdf")
    fig.savefig(outputfile)

    fig = plot_key("train/entropy_loss", ylabel="Entropy loss")
    outputfile = os.path.join(output_dir, "entropy_loss.pdf")
    fig.savefig(outputfile)

    fig = plot_key("train/loss", ylabel="Total loss")
    outputfile = os.path.join(output_dir, "total_loss.pdf")
    fig.savefig(outputfile)

    print("Plotting clip fraction")
    fig = plot_key("train/clip_fraction", ylabel="PPO clip fraction")
    outputfile = os.path.join(output_dir, "clip_fraction.pdf")
    fig.savefig(outputfile)

    print("Plotting explained variance")
    fig = plot_key("train/explained_variance", ylabel="Explained variance")
    outputfile = os.path.join(output_dir, "explained_variance.pdf")
    fig.savefig(outputfile)
