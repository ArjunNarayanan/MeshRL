import matplotlib.pyplot as plt
import numpy as np


def plot_returns_vs_poly_degree(poly_degrees, avg_returns, std_returns):
    avg_returns = np.array(avg_returns)
    std_returns = np.array(std_returns)
    lower = np.clip(avg_returns - std_returns, a_min=0, a_max=None)
    upper = np.clip(avg_returns + std_returns, None, a_max=1)

    fig, ax = plt.subplots()
    ax.plot(poly_degrees, avg_returns)
    ax.fill_between(poly_degrees, lower, upper, alpha=0.3)
    ax.grid()
    ax.set_xlabel("Polygon degree")
    ax.set_ylabel("Average returns")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
