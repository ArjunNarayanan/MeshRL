import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="input csv file", required=True)
    parser.add_argument("-output", help="output file name", default="returns-vs-degree.pdf")
    parser.add_argument("-fontsize", default=16, type=int)
    args = parser.parse_args()

    filename = args.input
    folder = os.path.dirname(filename)
    fontsize = args.fontsize
    output_filename = args.output

    data = pd.read_csv(filename)

    polys = data["polygon degree"]
    returns = data["average returns"]
    dev = data["std. deviation returns"]

    lower = returns - dev
    upper = returns + dev

    fig, ax = plt.subplots()
    ax.plot(polys, returns, "-o", color="black", linewidth=2)
    ax.fill_between(polys, lower, upper, alpha=0.6, color="gray")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Polygon degrees", fontsize=fontsize)
    ax.set_ylabel("Average returns", fontsize=fontsize)
    ax.grid()

    fig.tight_layout()

    outfile = os.path.join(folder, output_filename)
    fig.savefig(outfile)
