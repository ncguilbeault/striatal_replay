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

def plot_trajectories(trajectories_df, ax, normalize_color = True, x_col = "x_position", y_col = "y_position", **lc_kwargs):
    default_kwargs = {
        "cmap" : "rainbow", 
        "alpha" : 0.1,
        "zorder": 3
    }
    default_kwargs.update(lc_kwargs)
    for i, trial in trajectories_df.groupby("trial_id"):
        x = trial[x_col].values
        y = trial[y_col].values
        if not normalize_color:
            steps = 1000
        else:
            steps = len(x)
        t = range(steps)
        ax, lc = colored_line(x, y, t, ax, **default_kwargs)
    return ax

def plot_trajectories_linear(trajectories_df, ax, normalize_color = True, steps = 1000, pos_col = "linear_position", **lc_kwargs):
    default_kwargs = {
        "cmap" : "rainbow", 
        "alpha" : 0.1,
        "zorder": 3
    }
    default_kwargs.update(lc_kwargs)
    for i, trial in trajectories_df.groupby("trial_id"):
        y = trial[pos_col].values
        x = np.arange(len(y))
        if normalize_color:
            steps = len(x)
        t = range(steps)
        ax, lc = colored_line(x, y, t, ax, **default_kwargs)
    return ax

def plot_ports(ports_df, ax):
    ax.plot(ports_df["x_position"], ports_df["y_position"], linewidth=5, c="black", alpha=0.8, zorder=2)
    ax.scatter(ports_df["x_position"], ports_df["y_position"], c="black", s=200, zorder=2)
    return ax

def plot_all_trajectories(trajectories_df, ports_df, ax = None, normalize_color = True, padding = 10, x_col = "x_position", y_col = "y_position", **lc_kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax = plot_trajectories(trajectories_df, ax, normalize_color, x_col, y_col, **lc_kwargs)
    ax = plot_ports(ports_df, ax)

    min_x = np.min([np.min(trajectories_df[x_col]), np.min(ports_df["x_position"])])
    max_x = np.max([np.max(trajectories_df[x_col]), np.max(ports_df["x_position"])])
    ax.set_xlim([min_x - padding, max_x + padding])

    min_y = np.min([np.min(trajectories_df[y_col]), np.min(ports_df["y_position"])])
    max_y = np.max([np.max(trajectories_df[y_col]), np.max(ports_df["y_position"])])
    ax.set_ylim([min_y - padding, max_y + padding])

    return ax

def plot_single_trajectory(trajectories_df, ports_df, trial_id = 0, ax = None, alpha = 0.8, **kwargs):
    trajectories = trajectories_df[trajectories_df["trial_id"] == trial_id]
    ax = plot_all_trajectories(trajectories, ports_df, ax = ax, alpha = alpha, **kwargs)
    return ax

def plot_all_trajectories_linear(trajectories_df, ports_df, ax = None, normalize_color = True, padding = 10, pos_col = "linear_position", x_range = None, y_range = None, **lc_kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax = plot_trajectories_linear(trajectories_df, ax, normalize_color, pos_col, **lc_kwargs)

    if x_range is None:
        x_max = trajectories_df.groupby("trial_id").agg("count").iloc[:,0].max()
        x_min = 0
    else:
        x_min = x_range[0]
        x_max = x_range[1]

    if y_range is None:
        y_min = trajectories_df[pos_col].min()
        y_max = trajectories_df[pos_col].max()
    else:
        y_min = y_range[0]
        y_max = y_range[1]
        
    ax.hlines(ports_df[pos_col], x_min, x_max, linewidth=2, colors="black", alpha=0.8, zorder=2, ls="--")

    ax.set_xlim([x_min - padding, x_max + padding])
    ax.set_ylim([y_min - padding, y_max + padding])

    return ax

def plot_single_trajectory_linear(trajectories_df, ports_df, trial_id = 0, ax = None, alpha = 0.8, **kwargs):
    trajectories = trajectories_df[trajectories_df["trial_id"] == trial_id]
    ax = plot_all_trajectories_linear(trajectories, ports_df, ax = ax, alpha = alpha, **kwargs)
    return ax