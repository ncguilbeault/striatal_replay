import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from open_ephys.analysis import Session

def find_correct_seq_indices(behavioral_data, correct_seq=[2, 1, 6, 3, 7]):
    correct_seqs_indices = []
    reward_times_indices = \
        np.where(np.logical_not(np.isnan(behavioral_data["Reward_Times"])))[0]
    for reward_index in reward_times_indices:
        seq_index = 4
        prev_index = 0
        append_seq_indices = True
        correct_seq_indices = []
        while seq_index >= 0:
            while behavioral_data.loc[reward_index-prev_index, "Port"] == \
                    correct_seq[seq_index]:
                correct_seq_indices.append(reward_index-prev_index)
                prev_index += 1
            seq_index -= 1
            if seq_index >= 0:
                if behavioral_data.loc[reward_index-prev_index, "Port"] != \
                    correct_seq[seq_index]:
                        append_seq_indices = False
        if append_seq_indices:
            correct_seqs_indices += correct_seq_indices
    correct_seqs_indices.sort()
    return correct_seqs_indices

def filter_df_on_likelihood(df, threshold = 0.98, interpolate_between=True):
    df = df.copy()
    df.columns = df.columns.droplevel((0,1))
    if not interpolate_between:
        df = df[df["likelihood"] >= threshold]
    else:
        df.loc[df["likelihood"] < threshold] = pd.NA
        if df.iloc[0].isna().all():
            first_vals = df.isna().values.argmin(axis=0)
            for i, val in enumerate(first_vals):
                df.iloc[[0, i]] = df.iloc[[val, i]].copy()
        if df.iloc[-1].isna().all():
            last_vals = len(df) - df.isna().values[::-1].argmin(axis=0) - 1
            for i, val in enumerate(last_vals):
                df.iloc[[-1, i]] = df.iloc[[val, i]].copy()
        df.interpolate(inplace=True)
    return df[["x", "y"]]

def extract_centre_point_using_histogram(df, bins=20):
    hist_counts, hist_edges_x, hist_edges_y = np.histogram2d(df.x, df.y, bins=bins)
    hist_centers_x = np.diff(hist_edges_x) + hist_edges_x[0]
    hist_centers_y = np.diff(hist_edges_y) + hist_edges_y[0]
    idx_x, idx_y = np.unravel_index(np.argmax(hist_counts), hist_counts.shape)
    point = (hist_centers_x[idx_x], hist_centers_y[idx_y])
    return point

def get_starts_ends_of_sequence(df1, df2, start_time_col="backcam_aligned_pokein_times", end_time_col="backcam_aligned_pokeout_times"):
    shared_idxs = df2.index[df2.index.isin(df1.index)]
    starts = df1.loc[shared_idxs][start_time_col]
    ends = df2.loc[shared_idxs][end_time_col]
    starts_ends_times_seqs = pd.concat([starts, ends], axis=1)
    starts_ends_times_seqs.columns = ("start_time", "end_time")
    return starts_ends_times_seqs

def get_tracking_idxs_from_times_df(times_df, keypoint_df, fps = 60):
    timestamps = keypoint_df.index.values / fps
    tracking_idxs = {
        "start_idx": [],
        "end_idx": []
    }
    for i, (start_time, end_time) in times_df.iterrows():
        tracking_idxs["start_idx"].append(np.abs(timestamps - start_time).argmin())
        tracking_idxs["end_idx"].append(np.abs(timestamps - end_time).argmin())
    return pd.DataFrame(tracking_idxs, index=times_df.index)

def get_camera_sync_times(keypoint_df, sync_df, fps = 60):
    first_pokes = sync_df["FirstPoke_EphysTime"].values
    mask = ~np.isnan(first_pokes)
    ephys_times = first_pokes[mask]
    camera_times = sync_df["backcam_aligned_P1In_times"].values[mask]
    camera_frames = np.round(camera_times * fps)
    frame_to_ephys_interpolation = interp1d(camera_frames,ephys_times,fill_value='extrapolate')
    frame_to_camera_interpolation = interp1d(camera_frames,camera_times,fill_value='extrapolate')
    camera_frames_from_movement = keypoint_df.iloc[:,0].index.values
    camera_frames_in_ephys_time = frame_to_ephys_interpolation(camera_frames_from_movement)
    camera_frames_in_camera_time = frame_to_camera_interpolation(camera_frames_from_movement)
    return pd.DataFrame({
        "ephys_time": camera_frames_in_ephys_time,
        "camera_time": camera_frames_in_camera_time
    })

def get_camera_times_from_times_df(times_df, camera_times_df, camera_times_col = "camera_time"):
    tracking_idxs = {
        "start_idx": [],
        "end_idx": []
    }
    for i, (start_time, end_time) in times_df.iterrows():
        tracking_idxs["start_idx"].append((camera_times_df[camera_times_col] - start_time).abs().argmin())
        tracking_idxs["end_idx"].append((camera_times_df[camera_times_col] - end_time).abs().argmin())
    return pd.DataFrame(tracking_idxs, index = times_df.index)

def interpolate_x_y_data(df, t = np.arange(0, 1, 1000)):
    p = np.linspace(0, 1, len(df))
    x_interp = np.interp(t, p, df.x)
    y_interp = np.interp(t, p, df.y)
    return x_interp, y_interp

def calculate_average_keypoint(df, keypoint_cols=["left_ear", "right_ear", "head_centre"], threshold=[0.98, 0.98, 0.98]):
    if isinstance(threshold, float):
        threshold = [threshold] * len(keypoint_cols)
    output = None
    for keypoint_col, thresh in zip(keypoint_cols, threshold):
        keypoint_data = df.loc[:, (slice(None), keypoint_col, ["x", "y", "likelihood"])]
        keypoint_data = filter_df_on_likelihood(keypoint_data, thresh)
        if output is None:
            output = keypoint_data
            counts = pd.DataFrame(np.ones(len(output)), index=output.index)
            counts = pd.concat([counts.T] * 2).T
            counts.columns = ("x", "y")
        else:
            new_idxs = keypoint_data[~keypoint_data.index.isin(output.index)].index
            new_counts = pd.DataFrame(np.ones(len(new_idxs)), index=new_idxs)
            new_counts = pd.concat([new_counts.T] * 2).T
            new_counts.columns = ("x", "y")
            shared_idxs = keypoint_data[keypoint_data.index.isin(output.index)].index
            counts = pd.concat([counts, new_counts]).sort_index()
            keypoint_data.loc[shared_idxs] /= counts.loc[shared_idxs]
            output.loc[shared_idxs] *= (counts.loc[shared_idxs] - 1) / counts.loc[shared_idxs]
            output.loc[shared_idxs] += keypoint_data.loc[shared_idxs]
            output = pd.concat([keypoint_data.loc[new_idxs], output]).sort_index()
    return output

def get_max_trajectory_duration(trajectories_df):
    last = trajectories_df.groupby("trial_id").agg("last")["camera_time"]
    first = trajectories_df.groupby("trial_id").agg("first")["camera_time"]
    return np.max(last - first)

def get_ephys_timestamps(ephys_folder):
    session_data = Session(ephys_folder)
    recording = session_data.recordnodes[0].recordings[0]
    recording.add_sync_line(1, 102, '0', main=True)
    recording.compute_global_timestamps()
    ephys_timestamps = recording.continuous[2].global_timestamps
    return ephys_timestamps

def get_spike_matrix(spike_clusters, cutoff = np.Inf):
    cutoff = np.min([cutoff, len(spike_clusters)])
    unique_clusters = np.unique(spike_clusters)
    clusters_count = len(unique_clusters)
    clusters_mapping = np.arange(np.max(spike_clusters)+1)
    clusters_mapping[unique_clusters] = np.arange(clusters_count)
    spikes = np.zeros((clusters_count, cutoff), dtype=int)
    spikes[clusters_mapping[spike_clusters][:cutoff], np.arange(spikes.shape[1])] = 1
    return spikes