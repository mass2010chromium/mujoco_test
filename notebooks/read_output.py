import os

import pickle
import numpy as np
import mediapy

show = False
if show:
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(0)

corrupted = []

i = 0
name_template = "output/franka_stacking.{0}"
while True:
    base_name = name_template.format(i)
    data_filename = base_name + '.pkl'
    if not os.path.exists(data_filename):
        break
    with open(data_filename, 'rb') as data_file:
        data = pickle.load(data_file)
    video_data = mediapy.read_video(base_name+'.mp4')

    # Detect if over 50% of pixels are black
    # Technically not gray but good enough for me
    frame_grays = np.max(video_data, axis=3)
    black_frames = np.median(frame_grays, axis=(1, 2))

    corrupted_video = np.any(black_frames == 0)
    if show:
        print("Showing run", i)
        if corrupted_video:
            print("Corrupted video!")
        for subtask, status, snapshot in zip(data['subtasks'], data['statuses'], data['snapshots']):
            plt.figure(0)
            plt.clf()
            plt.imshow(snapshot)
            plt.pause(0.5)
            print(subtask)
            print(status)
            if input().strip().lower() == 'q':
                print("Skipping")
                break
    if corrupted_video:
        corrupted.append(i)

    i += 1
print("Corrupted videos:", corrupted)
