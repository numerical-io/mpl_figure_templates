from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from figure_templates import make_figure, set_axis, style

plt.ioff()
save_dir = Path(".").absolute()


# Print aside
with style("print"):
    fig = make_figure("print_aside")
    ax = fig.axes[0]
    x = np.linspace(0, 10, 101)
    mu, s2 = 3, 1.5
    ax.plot(
        x,
        np.exp(-((x - mu) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2),
        color="#cccccc",
    )
    mu, s2 = 6, 1
    ax.plot(
        x,
        np.exp(-((x - mu) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2),
        color="r",
    )
    ax.set_xlabel("Application time [s]")
    ax.set_ylabel("Pressure [bar]")
    set_axis(ax, "x", major_ticks=[0, 5, 10])
    set_axis(ax, "y", major_ticks=[0, 0.4], minor_ticks=[])
    fig.show()
    fig.savefig(save_dir / "print_aside.png")


# Print aside, multiple
with style("print"):
    fig = make_figure(
        "print_aside",
        num_rows=3,
        fig_height=24,
        ylabel_preset="center",
        ylabel_pad=2,
        margin_top=0.5,
    )
    x = np.linspace(0, 10, 101)
    y1 = np.exp(-0.5 * x)
    y2 = np.concatenate((np.zeros(31), np.sin(50 / x[1:71])))
    y3 = np.concatenate((np.zeros(31), np.exp(x[1:71] / 3) - 1))

    ax = fig.axes[0]
    ax.axvline(x[30], color="#cccccc")
    ax.axhline(y1[30], color="#cccccc")
    ax.plot(x, y1, color="orange")
    set_axis(ax, "y", major_ticks=[0, y1[30], 1], minor_ticks=[])
    ax.set_ylabel("Pressure [bar]", linespacing=1.6)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    ax = fig.axes[1]
    ax.axvline(x[30], color="#cccccc")
    ax.plot(x, y2, color="orange", clip_on=False)
    set_axis(ax, "y", major_ticks=[-1, 0, 1], minor_ticks=[])
    ax.set_ylabel("Air flow [l/s]", linespacing=1.6)

    ax = fig.axes[2]
    ax.axvline(x[30], color="#cccccc")
    ax.plot(x, y3, color="r")
    ax.set_xlabel("Time [s]")
    set_axis(ax, "y", major_ticks=[0, 10], minor_ticks=[5])
    ax.set_ylabel("Concentration [g/l]", linespacing=1.6)

    for ax in fig.axes:
        set_axis(ax, "x", major_ticks=[0, 3, 5, 10], minor_ticks=[])

    fig.savefig(save_dir / "print_aside_multi.png")


# Print textwidth
with style("print"):
    delta = 0.025
    x = np.arange(-3, 3, delta)
    y = np.arange(-2, 2, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2) - Y**2)
    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    fig = make_figure("print_textwidth")
    ax = fig.axes[0]
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True)
    set_axis(ax, "x", major_ticks=[-3, 0, 3], minor_ticks=[-2, -1, 1, 2])
    set_axis(ax, "y", major_ticks=[-2, 0, 2], minor_ticks=[-1, 1])
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    fig.savefig(save_dir / "print_textwidth.png")

# Print fullwidth
with style("print"):
    df = pd.DataFrame(
        [
            [10.0, 8.04, 10.0, 9.14, 10.0, 7.46, 8.0, 6.58],
            [8.0, 6.95, 8.0, 8.14, 8.0, 6.77, 8.0, 5.76],
            [13.0, 7.58, 13.0, 8.74, 13.0, 12.74, 8.0, 7.71],
            [9.0, 8.81, 9.0, 8.77, 9.0, 7.11, 8.0, 8.84],
            [11.0, 8.33, 11.0, 9.26, 11.0, 7.81, 8.0, 8.47],
            [14.0, 9.96, 14.0, 8.10, 14.0, 8.84, 8.0, 7.04],
            [6.0, 7.24, 6.0, 6.13, 6.0, 6.08, 8.0, 5.25],
            [4.0, 4.26, 4.0, 3.10, 4.0, 5.39, 19.0, 12.50],
            [12.0, 10.84, 12.0, 9.13, 12.0, 8.15, 8.0, 5.56],
            [7.0, 4.82, 7.0, 7.26, 7.0, 6.42, 8.0, 7.91],
            [5.0, 5.68, 5.0, 4.74, 5.0, 5.73, 8.0, 6.89],
        ],
        columns=pd.MultiIndex.from_product([range(4), ["x", "y"]]),
    )
    x = [0, 20]

    fig = make_figure("print_fullwidth", num_cols=4)

    for n, ax in enumerate(fig.axes):
        series = df.loc[:, n]
        a = series[["x"]].assign(intercept=1)
        sol, _, _, _ = np.linalg.lstsq(a, series.y)
        y = np.matmul([[x[0], 1], [x[1], 1]], sol)
        ax.plot(x, y, color="#cccccc")
        ax.scatter(series.x, series.y, c="r")
        set_axis(ax, "x", major_ticks=[0, 10, 20], minor_ticks=[5, 15])
        set_axis(ax, "y", major_ticks=[0, 15], minor_ticks=[5, 10])
        ax.set_ylabel(f"Anscombe {n}")
    fig.axes[1].set_xlabel("Abscissa")

    fig.savefig(save_dir / "print_fullwidth.png")


# Slide fullwidth
with style("slide"):
    df = pd.DataFrame(
        [
            [10.0, 8.04, 10.0, 9.14, 10.0, 7.46, 8.0, 6.58],
            [8.0, 6.95, 8.0, 8.14, 8.0, 6.77, 8.0, 5.76],
            [13.0, 7.58, 13.0, 8.74, 13.0, 12.74, 8.0, 7.71],
            [9.0, 8.81, 9.0, 8.77, 9.0, 7.11, 8.0, 8.84],
            [11.0, 8.33, 11.0, 9.26, 11.0, 7.81, 8.0, 8.47],
            [14.0, 9.96, 14.0, 8.10, 14.0, 8.84, 8.0, 7.04],
            [6.0, 7.24, 6.0, 6.13, 6.0, 6.08, 8.0, 5.25],
            [4.0, 4.26, 4.0, 3.10, 4.0, 5.39, 19.0, 12.50],
            [12.0, 10.84, 12.0, 9.13, 12.0, 8.15, 8.0, 5.56],
            [7.0, 4.82, 7.0, 7.26, 7.0, 6.42, 8.0, 7.91],
            [5.0, 5.68, 5.0, 4.74, 5.0, 5.73, 8.0, 6.89],
        ],
        columns=pd.MultiIndex.from_product([range(4), ["x", "y"]]),
    )
    x = [0, 20]

    fig = make_figure("slide_fullwidth", num_cols=3)

    for n, ax in enumerate(fig.axes):
        series = df.loc[:, n]
        a = series[["x"]].assign(intercept=1)
        sol, _, _, _ = np.linalg.lstsq(a, series.y)
        y = np.matmul([[x[0], 1], [x[1], 1]], sol)
        ax.plot(x, y, color="#cccccc")
        ax.scatter(series.x, series.y, c="r")
        set_axis(ax, "x", major_ticks=[0, 10, 20], minor_ticks=[5, 15])
        set_axis(ax, "y", major_ticks=[0, 15], minor_ticks=[5, 10])
        ax.set_ylabel(f"Anscombe {n}")
    fig.axes[-2].set_xlabel("Abscissa")

    fig.savefig(save_dir / "slide_fullwidth_multi.png")


# Slide fullwidth
with style("slide"):
    delta = 0.025
    x = np.arange(-3, 3, delta)
    y = np.arange(-2, 2, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-((X + 0.8) ** 2) - Y**2)
    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    fig = make_figure("slide_fullwidth")
    ax = fig.axes[0]
    CS = ax.contour(X, Y, Z, levels=20)
    ax.clabel(CS, inline=True)
    set_axis(ax, "x", major_ticks=[-3, 0, 3], minor_ticks=[-2, -1, 1, 2])
    set_axis(ax, "y", major_ticks=[-2, 0, 2], minor_ticks=[-1, 1])
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    fig.savefig(save_dir / "slide_fullwidth.png")


# Slide square
with style("slide"):
    x = np.linspace(0, 15, 101)
    y = 15 + (x - 3) ** 2

    points_x = 7 + 3 * np.random.randn(60)
    points_y = 15 + (points_x - 3) ** 2 + 6 * np.random.randn(60)

    fig = make_figure("slide_square")
    ax = fig.axes[0]
    ax.plot(x, y, color="#cccccc")
    ax.scatter(points_x, points_y, color="r")
    set_axis(ax, "x", major_ticks=[0, 5, 10, 15], minor_ticks=[])
    set_axis(ax, "y", major_ticks=[0, 50, 100, 150], minor_ticks=[])
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Speed [m/s]")

    fig.savefig(save_dir / "slide_square.png")


# Print aside, multiple
with style("slide"):
    fig = make_figure("slide_square", num_rows=2)
    x = np.linspace(0, 10, 101)
    y1 = np.exp(-0.5 * x)
    y2 = np.concatenate((np.zeros(31), np.sin(50 / x[1:71])))
    y3 = np.concatenate((np.zeros(31), np.exp(x[1:71] / 3) - 1))

    ax = fig.axes[0]
    ax.axvline(x[30], color="#cccccc")
    ax.plot(x, y2, color="orange", clip_on=False)
    set_axis(ax, "y", major_ticks=[-1, 0, 1], minor_ticks=[])
    ax.set_ylabel("Air flow [l/s]")

    ax = fig.axes[1]
    ax.axvline(x[30], ymax=0.8, color="#cccccc")
    ax.plot(x, y3, color="r")
    ax.set_xlabel("Time [s]")
    set_axis(ax, "y", major_ticks=[0, 10], minor_ticks=[5])
    ax.set_ylabel("Concentration [g/l]")

    for ax in fig.axes:
        set_axis(ax, "x", major_ticks=[0, 3, 5, 10], minor_ticks=[])

    fig.savefig(save_dir / "slide_square_multi.png")


plt.close("all")
