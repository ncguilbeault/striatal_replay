import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kalman_filter import KalmanFilter
from src.data_processing import *
from src.plotting import *
import pandas as pd
import numpy as np
from time import time

organized_data_folder = "/ceph/sjones/projects/sequence_squad/organised_data/animals/EJT178_implant1/recording7_30-03-2022/"
processed_data_folder = "/nfs/gatsbystor/nicholasg/striatal_replay/processed_data"

if __name__ == "__main__":

    start_exec_time = time()
    print(f"Starting time is : {start_exec_time}s.")

    print("Loading data.")
    back_pos_df = pd.read_hdf(os.path.join(organized_data_folder, "video", "tracking", "2_task", "back_2022-03-30T15_02_32DLC_resnet50_task-tracking_backviewApr6shuffle1_800000.h5"), "df_with_missing")

    print("Calculating keypoint position.")
    keypoints = calculate_average_keypoint(back_pos_df, keypoint_cols=["head_centre"])
    y = keypoints.values.T

    print("Initializing Kalman Filter.")
    kf = KalmanFilter(pos_x0=y[0,0], pos_y0=y[1,0])

    print("Optimizing parameters.")
    training = int(len(keypoints) * 0.01)
    kf.optimize(y[:training], max_iter = 5, disp = True)

    print("Smoothing observations.")
    means, std_devs = kf.smooth(y)

    print("Writing to csv.")
    keypoints["smoothed_x_position"] = means[0, 0, :]
    keypoints["smoothed_y_position"] = means[3, 0, :]
    keypoints["velocity"] = np.sqrt(means[1, 0, :] ** 2 + means[4, 0, :] ** 2)
    keypoints["acceleration"] = np.sqrt(means[2, 0, :] ** 2 + means[5, 0, :] ** 2)
    keypoints.to_csv(os.path.join(processed_data_folder, "keypoints.csv"))
    
    end_exec_time = time()
    print(f"Ending time is : {end_exec_time}s.")
    print(f"Total execution time is : {end_exec_time - start_exec_time}s.")
    print("Finished.")