import sys
sys.path.append("/nfs/gatsbystor/nicholasg/replay_trajectory_classification")
from replay_trajectory_classification import SortedSpikesDecoder
from replay_trajectory_classification.state_transition import estimate_movement_var
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time

# from dask.distributed import Client

# Client(n_workers=4,
#     threads_per_worker=2,
#     processes=True,
#     memory_limit='32GB')

processed_data_folder = "/nfs/gatsbystor/nicholasg/striatal_replay/processed_data"
figures_folder = "/nfs/gatsbystor/nicholasg/striatal_replay/figures/decoder_script"

def get_time_mask_where_spikes_occur_in_window(spikes, start, end):
    mask = np.zeros(len(spikes)).astype(bool)
    mask[start:end] = ~((spikes[start:end] == 0).all(axis=1))
    return mask

def plot_all_results(results, position, start, end):
    fig, ax = plt.subplots()
    ax.imshow(results.causal_posterior.T[::-1], cmap="plasma", extent=(0, len(results.causal_posterior), np.min(position[start:end]), np.max(position[start:end])), vmin=0.1, vmax=0.2)
    ax.set_aspect("auto")
    ax.plot(position[start:end], c="lime")
    plt.savefig(os.path.join(figures_folder, f"causal_posterior_start_{start}_end_{end}.png"))
    plt.close(fig)

def plot_results_start_end(results, position, number, start, end):
    fig, ax = plt.subplots()
    ax.imshow(results.causal_posterior.T[::-1, start:end], cmap="plasma", extent=(0, (end - start), np.min(position[start:end]), np.max(position[start:end])), vmin=0.1, vmax=0.2)
    ax.set_aspect("auto")
    ax.plot(position[start:end], c="lime")
    plt.savefig(os.path.join(figures_folder, f"causal_posterior_trial_{number}_start_{start}_end_{end}.png"))
    plt.close(fig)

if __name__ == "__main__":

    start_exec_time = time()
    print(f"Starting time is : {start_exec_time}s.")

    Fs = 30000.0
    print("Loading data.")
    spikes = np.load(os.path.join(processed_data_folder, "spikes.npy")).T.astype(np.float64)
    position = np.load(os.path.join(processed_data_folder, "position.npy")).astype(np.float64)
    
    idxs = np.where(np.diff(position) < -500)[0]
    idxs = np.insert(idxs, 0, 0)
    movement_var = estimate_movement_var(position, 60)

    decoder = SortedSpikesDecoder(movement_var=movement_var,
                                    replay_speed=1,
                                    spike_model_penalty=0.5,
                                    place_bin_size=np.sqrt(movement_var))
    
    start = 0
    # trial_count = int(len(idxs) * 0.9)
    trial_count = 1
    end = idxs[trial_count]
    time_mask = get_time_mask_where_spikes_occur_in_window(spikes, start, end)
    print("Fitting decoder.")
    decoder.fit(position[time_mask], spikes[time_mask])

    prediction_folder = os.path.join(figures_folder, f"s{start}_e{end}")
    os.mkdir(prediction_folder)

    for i, (trial_start, trial_end) in enumerate(zip(idxs[:-1], idxs[1:])):
        print(f"Predicting trial number: {i + 1} trial start: {trial_start} trial end: {trial_end}", end="\r")
        trial_number = i + 1
        results = decoder.predict(spikes[trial_start:trial_end], time=np.arange(trial_end-trial_start)/Fs)
        fig, ax = plt.subplots()
        ax.imshow(results.causal_posterior.T[::-1], cmap="plasma", extent=(0, len(results.causal_posterior), np.min(position[trial_start:trial_end]), np.max(position[trial_start:trial_end])), vmin=0.1, vmax=0.22)
        ax.set_aspect("auto")
        ax.plot(position[trial_start:trial_end], c="lime")
        output_filename = os.path.join(prediction_folder, f"results_fit_s{start}_e{end}_trial{trial_number}_s{trial_start}_e{trial_end}.png")
        plt.savefig(output_filename)
        plt.close(fig)
    
    print("")
    end_exec_time = time()
    print(f"Ending time is : {end_exec_time}s.")
    print(f"Total execution time is : {end_exec_time - start_exec_time}s.")
    print("Finished.")