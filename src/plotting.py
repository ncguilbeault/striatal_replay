import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def colored_line(x, y, c, ax, **lc_kwargs):
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)
    lc = ax.add_collection(lc)
    return ax, lc

def plot_all_trajectories(trajectories_df, ports_df, ax = None, normalize_color = True, padding = 10, **lc_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    default_kwargs = {
        "cmap" : "rainbow", 
        "alpha" : 0.1,
        "zorder": 3
    }
    default_kwargs.update(lc_kwargs)
    for i, trial in trajectories_df.groupby("trial_id"):
        x = trial["x_position"].values
        y = trial["y_position"].values
        if not normalize_color:
            steps = 1000
        else:
            steps = len(x)
        t = range(steps)
        lines_kwargs = default_kwargs.copy()
        lines_kwargs["clim"] = (0, len(x))
        ax, lc = colored_line(x, y, t, ax, **default_kwargs)

    ax.plot(ports_df["x_position"], ports_df["y_position"], linewidth=5, c="black", alpha=0.8, zorder=2)
    ax.scatter(ports_df["x_position"], ports_df["y_position"], c="black", s=200, zorder=2)

    min_x = np.min([np.min(trajectories_df["x_position"]), np.min(ports_df["x_position"])])
    max_x = np.max([np.max(trajectories_df["x_position"]), np.max(ports_df["x_position"])])
    ax.set_xlim([min_x - padding, max_x + padding])

    min_y = np.min([np.min(trajectories_df["y_position"]), np.min(ports_df["y_position"])])
    max_y = np.max([np.max(trajectories_df["y_position"]), np.max(ports_df["y_position"])])
    ax.set_ylim([min_y - padding, max_y + padding])

    return ax

def plot_single_trajectory(trajectories_df, ports_df, trial_id = 0, ax = None, normalize_color = True, padding = 10, **lc_kwargs):
    trajectories = trajectories_df[trajectories_df["trial_id"] == trial_id]
    ax = plot_all_trajectories(trajectories, ports_df, ax = ax, alpha = 0.8)
    return ax

def plot_all_trajectories_linear(trajectories_df, ports_df, ax = None, normalize_color = True, padding = 10, x_range = None, y_range = None, **lc_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    default_kwargs = {
        "cmap" : "rainbow", 
        "alpha" : 0.1,
        "zorder": 3
    }
    default_kwargs.update(lc_kwargs)
    for i, trial in trajectories_df.groupby("trial_id"):
        y = trial["linear_position"].values
        x = np.arange(len(y))
        if not normalize_color:
            steps = 1000
        else:
            steps = len(x)
        t = range(steps)
        lines_kwargs = default_kwargs.copy()
        lines_kwargs["clim"] = (0, len(x))
        ax, lc = colored_line(x, y, t, ax, **default_kwargs)

    if x_range is None:
        x_max = trajectories_df.groupby("trial_id").agg("count").iloc[:,0].max()
        x_min = 0
    else:
        x_min = x_range[0]
        x_max = x_range[1]

    if y_range is None:
        y_min = trajectories_df["linear_position"].min()
        y_max = trajectories_df["linear_position"].max()
    else:
        y_min = y_range[0]
        y_max = y_range[1]
        
    ax.hlines(ports_df["linear_position"], x_min, x_max, linewidth=2, colors="black", alpha=0.8, zorder=2, ls="--")

    ax.set_xlim([x_min - padding, x_max + padding])
    ax.set_ylim([y_min - padding, y_max + padding])

    return ax

def plot_single_trajectory_linear(trajectories_df, ports_df, trial_id = 0, ax = None, normalize_color = True, padding = 10, **lc_kwargs):
    trajectories = trajectories_df[trajectories_df["trial_id"] == trial_id]
    ax = plot_all_trajectories_linear(trajectories, ports_df, ax = ax, alpha = 0.8)
    return ax