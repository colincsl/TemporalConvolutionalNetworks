
"""
Requires scipy, pandas, difflib
sudo pip install difflib
"""

import numpy as np
import pandas as pd
import os.path as path
import os
from pylab import *
import difflib
from scipy.io import savemat, loadmat
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree as KDTree

# --------User params---------------
save_features = [False, True][0]
viz_timelines = [False, True][0]
viz_video = [False, True][1]
# Don't create new splits unless there is good reason. 
create_new_test_splits = [False, True][0]
# -----------------------

if viz_video:
    import skvideo.io

DATASET = path.expanduser("~/data/50Salads/")
label_types_all = ["low", "mid", "eval", "high"]

# Base folders
uri_accel = DATASET+"raw/acc-sync/accelerometers/"
uri_sync = DATASET+"raw/acc-sync/synchronization/"
uri_videos = DATASET+"raw/rgb/"
uri_timestamps = DATASET+"raw/ann-ts/timestamps/"
uri_sequences = DATASET+"labels/sequences/"
uri_actions = DATASET+"labels/"
uri_viz = DATASET+"viz/"
uri_save = DATASET+"features/accel/"
uri_videos_out = DATASET+"viz/timelines_new/"

# Get files in these folders
uris_accel = os.listdir(uri_accel)
uris_sync = os.listdir(uri_sync)
uris_labels = os.listdir(uri_actions)

# Only keep the trial names (e.g., 16-1 from 16-1-sync.txt)
trials_sync = [f[:4] for f in uris_sync]
trials_accel = [f[:4] for f in uris_accel]
trials_labels = [f[:4] for f in uris_accel]
# The 0th user should be removed (no sync data); there are 52 files but 50 usable trials
trials = sorted([t for t in trials_sync if t in trials_accel and t in trials_labels])

# ------------ Helper functions --------------
class SyncParams:
    a1, a2 = None, None
    o1, o2 = None, None

    def __init__(self, a1, a2, o1, o2):
        self.a1, self.a2 = float(a1), float(a2)
        self.o1, self.o2 = float(o1), float(o2)

def video2accel_time(vid_time, sync_params):
    """ Interpolate between video/accel times """
    a1, a2 = sync_params.a1, sync_params.a2
    o1, o2 = sync_params.o1, sync_params.o2

    acc_time = np.zeros_like(vid_time, np.float)
    # If before a1
    acc_time += (vid_time <= a1).astype(np.float) * (vid_time/1000 - o1)
    # If after a2
    acc_time += (vid_time >= a2) * (vid_time/1000 - o2)
    # If between
    l = (vid_time/1000 - a1) / (a2-a1)
    acc_time += ((vid_time > a1) * (vid_time < a2)) * (vid_time/1000 - (1-l)*o1 - l*o2)

    return acc_time

def time2accel(time_struct):
    t = time_struct.split()[1]
    hours, minutes, secs = t.split(":")
    secs, millis = secs.split(".")
    hours, minutes, secs, millis = map(int, [hours, minutes, secs, millis])

    return millis + 1000*(secs + 60*(minutes + 60*hours))

def load_sync_data(filename):
    header = ["c1", "c2"]
    data = pd.read_csv(filename, names=header)
    
    # Video time to accelerometer time
    a1, o1 = data["c1"][0], data["c2"][0]
    a2, o2 = data["c1"][1], data["c2"][1]
    sync_params = SyncParams(a1, a2, o1, o2)

    idxs = data["c1"][2:].values
    items = data["c2"][2:].values
    idx2item = {i:o for i,o in zip(idxs, items)}

    return sync_params, idx2item

def load_image_timestamps(filename):
    header = ["time", "img"]
    data = pd.read_csv(filename, delimiter=" ", names=header)
    return data

# Load label names
def load_labels(filename):
    return list(map(str.strip, open(uri_actions+filename+".txt").readlines()))

def load_high_level_labels(): return load_labels("high_level_actions")
def load_mid_level_labels(): return load_labels("mid_level_actions")
def load_low_level_labels(): return load_labels("low_level_actions")
def load_ingredient_labels(): return load_labels("ingredient_actions")
def load_abstract_labels(): return load_labels("abstract_actions")
def load_eval_labels(): return load_labels("eval_actions")
def load_ingredients(): return load_labels("tools")

def load_sequence_labels(filename, labels2idx, timestamps):
    data = pd.read_csv(filename, delimiter=" ", names=["start", "end", "action"]).values
    labels = list(labels2idx.keys())

    # Only keep the labels for this granularity
    data = np.array([d for d in data if any([l in d[2] for l in labels])])

    # Check if it's the high level. There is an overlap with cut_and_mix and cut_[ingredient]
    if "serve_salad" in labels:
        data = np.array([d for d in data if "serve" not in d[2] or "serve_salad"==d[2] ])
    else:
        data = np.array([d for d in data if "serve_salad"!=d[2] and "cut_and_mix_ingredients"!=d[2]])

    # Get closest matching label/action
    starts, ends, actions = data[:,0], data[:,1], data[:,2]    
    for i in range(len(actions)):
        labels_tmp = [l for l in labels if l[:3]==actions[i][:3]]
        matches = difflib.get_close_matches(actions[i], labels_tmp, cutoff=0)[0]
        # print(actions[i], matches)
        actions[i] = matches

    # Go through each timestep and add label
    n_frames = timestamps.shape[0]    
    labels_dense = np.zeros(n_frames, np.int)
    for i in range(n_frames):
        t = timestamps[i]
        idx_action = np.nonzero((starts <= t) * (t < ends))[0]

        if len(idx_action) > 0:
            labels_dense[i] = labels2idx[actions[idx_action[-1]]]

    return labels_dense


# ------------ Load data --------------
ingredients = load_ingredients()
ingredient2idx = {l:i for i,l in enumerate(ingredients)}
idx2ingredient = {i:l for i,l in enumerate(ingredients)}
n_sensors = len(ingredients)

def load_accel_data(filename, idx2item, timestamps_acc, n_sensors=n_sensors):
    header = ["junk", "time", "id", "seq", "x_acc", "y_acc", "z_acc"]
    data = pd.read_csv(filename, names=header)
    times = np.array([time2accel(d) for d in data["time"]])
    ids = data["id"].values
    acc_raw = np.vstack([data["x_acc"].values, data["y_acc"].values, data["z_acc"].values]).T
    
    n_frames = timestamps_acc.shape[0]

    accs_all = np.zeros([n_sensors, n_frames, 3], np.float)
    for i, ingred in idx2item.items():
        accs = np.zeros([n_frames, 3], np.float64)
        times_per_sensor = times[ids==i]
        acc_per_sensor = acc_raw[ids==i]

        if len(times_per_sensor) > 0:
            tree = KDTree(times_per_sensor[:,None])
            dist, idxs_closest = tree.query(timestamps_acc[:,None])

            # Only add if within X millis of actual time
            valid = dist < 250
            accs[valid,:] = acc_per_sensor[idxs_closest[valid]]

        global_ingredient_idx = ingredient2idx[ingred]
        accs_all[global_ingredient_idx] = accs
        
    accs_all_ = np.hstack(accs_all)

    return accs_all_

# ------------ Visualization ------------
label_fcns = {"low":load_low_level_labels,
                "mid":load_mid_level_labels,
                "high":load_high_level_labels,
                "abstract":load_abstract_labels,
                "tools":load_ingredient_labels,
                "eval":load_eval_labels}

# ------------ Save features and visualize ------------
# labels = load_mid_level_labels()
# label_type = ["low", "mid", "high", "abstract", "ingredient", "eval", "tools"][-2]

for trial in trials:
    # Get appropriate filename
    fid_sync = uri_sync + trial + "-synchronization.txt"
    fid_accel = uri_accel + trial + "-accelerometer.csv"
    fid_timestamps = uri_timestamps + "timestamps-"+trial+".txt"
    fid_sequences = uri_sequences + trial+"-activityAnnotation.txt"

    # Get video/accel params and time synchronization
    # idx2item is improtant to correspond which accelerometer is on which tool between trials.
    sync_params, idx2item = load_sync_data(fid_sync)
    timestamps = load_image_timestamps(fid_timestamps)["time"].values
    n_frames = len(timestamps)
    # print(idx2item.values())

    # Get sequence labels
    labels_all = []
    for label_type in label_types_all:
        labels = label_fcns[label_type]()
        labels2idx = {l:i+1 for i,l in enumerate(labels)}
        seq_labels = load_sequence_labels(fid_sequences, labels2idx, timestamps)
        labels_all += [seq_labels]
    labels_all = np.vstack(labels_all).T

    # Trim the start and end of the trial
    seq_labels = labels_all.sum(1) > 0
    start_frame = seq_labels.shape[0] - np.nonzero(seq_labels[::-1] > 0)[0][-1]
    end_frame = np.nonzero(seq_labels > 0)[0][-1]+1
    labels_all = labels_all[start_frame:end_frame]
    timestamps = timestamps[start_frame:end_frame]
    n_frames = timestamps.shape

    # Fill in short background. (reference the lowest layer; 'eval' uses bg class in an odd way)
    if 0:
        unfilled_idxs = np.nonzero(labels_all[:,0] == 0)[0]
        if len(unfilled_idxs) > 0:
            filled_idxs = np.nonzero(labels_all[:,0] != 0)[0]
            tree = KDTree(filled_idxs[:,None])
            idxs_closest = filled_idxs[tree.query(unfilled_idxs[:,None])[1]]
            for i_un, i_fill in zip(unfilled_idxs, idxs_closest):
                labels_all[i_un] = labels_all[i_fill]


    #  ------------ Get accelerometer data --------------
    timestamps_acc = video2accel_time(timestamps, sync_params)
    accs_raw = load_accel_data(fid_accel, idx2item, timestamps_acc)

    # Take absolute value of raw data
    accs_abs = np.abs(accs_raw)

    # # Compute norm of accelerometer signals
    # accs_norm = []
    # for i in range(n_sensors):
    #     accs_norm += [np.linalg.norm(np.abs(accs_raw)[:,i*3:(i+1)*3], 2, 1)]
    # accs_norm = np.abs(np.vstack(accs_norm))
    # accs_norm = 1*(accs_norm > 0)

    # Get framestamps
    video_file = uri_videos+"rgb-{}.avi".format(trial)
    framestamps = np.arange(start_frame, end_frame)

    if viz_timelines:
        plt.figure(trial, figsize=(10,10))
        plt.subplot(2,1,1); 
        plt.imshow(accs_abs.T, interpolation="nearest"); plt.axis("tight")
        # subplot(2,1,1); imshow_(accs_norm_old.T)
        # subplot(2,1,1); imshow_(accs_abs.T>0)
        # yticks(range(accs_norm.shape[1]), ingredients)
        plt.subplot(2,1,2); 
        plt.imshow(labels_all[:,::-1].T/labels_all[:,::-1].max(0)[:,None], interpolation="nearest")
        plt.axis("tight")
        # savefig(uri_viz+"accel/"+trial+".png")

    # ----------------------- Save data --------------------
    x = accs_abs.T

    if save_features:
        # Save each label granularity independently
        for i,label_type in enumerate(label_types_all):
            # output_name_ = output_name + "_" + label_type
            uri_save = path.expanduser("~/Data/50Salads/features/accel_raw/"+label_type+"/")
            if not path.isdir(uri_save):
                os.mkdir(uri_save)
            savemat(uri_save + trial, {"X":x, "Y":labels_all[:,i], 
                                       "file":video_file, "T":framestamps})

        uri_save = path.expanduser("~/Data/50Salads/features/accel_raw/"+"all"+"/")
        if not path.isdir(uri_save):
            os.mkdir(uri_save)
        savemat(uri_save + trial, {"X":x, "Y":labels_all.T})

    if viz_video:
        # Filenames
        video_file = uri_videos+"rgb-{}.avi".format(trial)
        video_file_out = uri_videos_out+"rgb-{}.mov".format(trial)

        if os.path.exists(video_file_out):
            continue

        # Load video
        vid = skvideo.io.vreader(video_file)
        meta = skvideo.io.ffprobe(video_file)['video']

        # Create timeline for labels
        w = int(meta['@width'])
        h = int(meta['@height'])
        # n_frames = ending - beginning
        # h, w, d = vid[0].shape
        n_valid = labels_all.shape[0]
        timeline_height = 50
        # timeline_height = 120
        timeline = np.zeros([timeline_height, w, 3])
        idxs = np.linspace(0, n_valid, w, False).astype(np.int)
        # timeline[:30] = cm.jet(labels_all[:,3][idxs] / 3.)[:,:3]
        # timeline[30:60] = cm.jet(labels_all[:,2][idxs] / 9.)[:,:3]
        # timeline[60:90] = cm.jet(labels_all[:,1][idxs] / 17.)[:,:3]
        # timeline[90:] = cm.jet(labels_all[:,0][idxs] / 51.)[:,:3]
        timeline[:] = cm.jet(labels_all[:,1][idxs] / 17.)[:,:3]
        timeline = (timeline*255).astype(np.uint8)

        # writer = skvideo.io.FFmpegWriter(video_file_out, (n_valid, h+90, w, 3))
        sample_rate = 3
        save_frames = np.empty([n_valid//sample_rate, h+timeline_height, w, 3], np.uint8)

        # Skip beginning frames (w/o label)
        for i_frame, im in enumerate(vid):
            if i_frame >= start_frame-1:
                break

        # Go through each image and add timeline
        ii_frame = 0
        for i_frame, im in enumerate(vid):
            if i_frame >= save_frames.shape[0]:
                break

            if not (i_frame % sample_rate) == 0:
                continue
            timeline_loc = int((i_frame) * (w/float(n_valid)))
            # timeline_loc = int((i_frame-beginning) * (w/float(n_valid)))
            timeline_loc = np.clip(timeline_loc, 1, w-2)
            timeline_ = timeline.copy()
            gesture_color = timeline[-1, timeline_loc]
            timeline_[:, timeline_loc-1:timeline_loc+2] = 255
            # im = vid[i_frame][:,:,[2,1,0]]
            im[:5] = gesture_color
            im[-5:] = gesture_color
            im[:,:5] = gesture_color
            im[:,-5:] = gesture_color
            im = np.vstack([im, timeline_])


            if (i_frame % sample_rate) == 0:
                # writer.writeFrame(im)
                save_frames[ii_frame] = im
                ii_frame += 1

            cv2.imshow("im", im[:,:,[2,1,0]])
            cv2.waitKey(1)
            print(i_frame, "of", n_valid)
        # save_frames[::3]
        skvideo.io.vwrite(video_file_out, save_frames)


if create_new_test_splits:
    uri_splits = os.path.expanduser(DATASET+"splits/sequences/")

    # Create test setups
    np.random.seed(0)
    users_all = np.unique([t.split("-")[0] for t in trials])
    users_left = users_all.tolist()
    for idx_iter in range(1,6):
        users_test = np.random.choice(users_left, 5, replace=False)
        users_train = [i for i in users_all if i not in users_test]

        [users_left.remove(u) for u in users_test]
        # users = np.unique([t.split("-") for t in trials])
        # idx_train = np.random.choice(range(len(users)), 21, replace=False)
        # users_train = [users[i] for i in range(len(users)) if i in idx_train]
        # users_test = [users[i] for i in range(len(users)) if i not in idx_train]

        trials_train = []
        for u in users_train:
            for i in ["-1", "-2"]:
                trial = u+i
                if trial in trials:
                    trials_train +=  [trial]
        trials_test = []
        for u in users_test:
            for i in ["-1", "-2"]:
                trial = u+i
                if trial in trials:
                    trials_test +=  [trial]
        folder_save = uri_splits+str(idx_iter)+"/"
        if not os.path.exists(folder_save):
            os.mkdir(folder_save)

        with open(folder_save+"train.txt", "w") as fid:
            for t in trials_train:
                fid.write(t+"\n")
        with open(folder_save+"test.txt", "w") as fid:
            for t in trials_test:
                fid.write(t+"\n")
