"""
Source for generating visualizations
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import List, Tuple


Vec = np.ndarray
Color = Tuple[float, float, float, float]


def find_theta(m: Vec, m_hat: Vec) -> float:
    """
    Return value of theta for given m and m_hat.
    """

    return np.max(np.abs(m - m_hat))


def gh_heatmap(ax: plt.Axes, mat: np.ndarray, xticklabels: List[str], yticklabels: List[str]):
    """
    Plot github style discrete heatmaps
    """

    n_rows, n_cols = mat.shape

    # Set limits
    ax.set_xlim(0, n_cols + 1)
    ax.set_ylim(0, n_rows + 1)
    ax.set_xticks(range(1, n_cols + 1))
    ax.set_xticklabels(xticklabels, rotation="90")
    ax.set_yticks(range(1, n_rows + 1))
    ax.set_yticklabels(yticklabels[::-1])

    padding = 0.2
    offset = 0.5
    for i in range(n_cols):
        for j in range(n_rows):
            r = mpl.patches.FancyBboxPatch(
                (i + padding + offset, j + padding + offset), 1 - 2 * padding, 1 - 2 * padding,
                color=mpl.cm.viridis_r(mat[n_rows - j - 1,i]),
                boxstyle=mpl.patches.BoxStyle("Round", pad=0.1)
            )
            ax.add_patch(r)


def plot_mm(ax: plt.Axes, m: Vec, m_hat: Vec, title=""):
    """
    Place mm plot on the given axes
    """

    lim = 1.2

    # Draw feasible region
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.fill([1,1,0], [0,1,1], "k", alpha=0.05)

    # Draw theta lines
    theta = find_theta(m, m_hat)
    x_line = mpl.lines.Line2D([theta, lim], [0, lim - theta],
                              linestyle="--", alpha=0.7,
                              label=f"$\\theta = {theta:.2f}$")
    y_line = mpl.lines.Line2D([0, lim - theta], [theta, lim], linestyle="--", alpha=0.7)
    ax.add_line(x_line)
    ax.add_line(y_line)

    # Draw middle line
    ax.add_line(mpl.lines.Line2D([0, lim], [0, lim], linestyle="--", color="gray", alpha=0.3))

    # Draw scatter
    colors = [mpl.cm.viridis_r(i / len(m)) for i in range(len(m))]
    sct = ax.scatter(m, m_hat, s=40, alpha=0.5, c=colors)
    ax.set_xlabel("$m$")
    ax.set_ylabel("$\hat{m}$")

    # Draw mean point
    mp = (np.mean(m), np.mean(m_hat))
    ax.add_patch(mpl.patches.Circle(mp, radius=0.015, color="gray", alpha=0.8))

    # Draw mean projections
    mp_x = mpl.lines.Line2D([mp[0], mp[0]], [0, mp[1]], linestyle="dotted", color="gray", alpha=0.3)
    ax.add_line(mp_x)
    mp_y = mpl.lines.Line2D([0, mp[0]], [mp[1], mp[1]], linestyle="dotted", color="gray", alpha=0.3)
    ax.add_line(mp_y)

    # Mean projection annotations
    ax.annotate(f"{mp[0]:.4f}", xy=(mp[0], 0), xytext=(mp[0] - 0.05, 0.05),
                arrowprops=dict(facecolor="gray", headwidth=1, width=1, shrink=0.05),
                horizontalalignment="right")
    ax.annotate(f"{mp[1]:.4f}", xy=(0, mp[1]), xytext=(0.05, mp[1] - 0.05),
                arrowprops=dict(facecolor="gray", headwidth=1, width=1, shrink=0.05),
                horizontalalignment="left")

    ax.legend()
    ax.set_title(title)
