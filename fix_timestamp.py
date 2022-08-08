import glob
import json
import numpy as np
import sys

from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
from natsort import natsorted  # For sorting string values considered as int

ms, us = 1.e-3, 1.e-6

def preprocess(path):
    result = []
    index = ['0', '1']
    for i in range(1, 5):
        # get json files' path for each sensor
        json_path = path + "sensor_{}/time/".format(i)
        for j in index:
            # Sorting filepaths and merging the files into one
            for file in natsorted(glob.glob(json_path + j + '*.json')):
                with open(file, 'rb') as infile:
                    result.append(json.load(infile))

            with open(path + "sensor_{}/{}_merged.json".format(i, j), "w") as outfile:
                json.dump(result, outfile, indent=4)
            result = []


def sorting(path):
    # Open merged file for sorting
    index = ["0", "1"]

    # we have 4 sensors starting from 1 to 4
    for i in range(1, 5):
        merged_json_path = path + "sensor_{}/".format(i)

        for j in index:
            merged_list = []

            with open(merged_json_path + "{}_merged.json".format(j), "rb") as infile:
                result = json.load(infile)

            for dictionary in result:
                for s in natsorted(dictionary.items()):
                    if s[0].startswith("buffer"):
                        merged_list.append(s[1])

            # Overwrite with merged
            with open(merged_json_path + "{}_merged.json".format(j), 'w') as outfile:
                json.dump(merged_list, outfile, indent=4)


def analyze_timestamps(ts, title, fps, n=10, bins=1000, debug=True, verbose=False, plot=True, savefigs=False):
    title, min_t = title.capitalize(), np.min(ts)
    ts = ts - min_t  # Work with relative values

    p_ref, dt = 1 / fps, np.diff(ts)
    # ps = dt[(0.99 * p_ref < dt) & (dt < 1.01 * p_ref)]
    # per = np.mean(ps)  # Inaccurate - use larger step

    p_ref_n, dt_n = n * p_ref, np.diff(ts[n::n])
    ps_n = dt_n[(0.95 * p_ref_n < dt_n) & (dt_n < 1.05 * p_ref_n)]
    per_n = np.mean(ps_n)  # Much better
    per = per_n / n

    # Compute median offset
    ofs = ts - np.round(ts / per) * per
    off = np.median(ofs)

    # Compute frame ids
    ids = np.round((ts - off) / per).astype(np.int)

    # Check for duplicates after rounding due to the frame acquisition delays
    for it in range(100):
        idx = np.nonzero(np.diff(ids) <= 0)[0]

        # Repeat until all consecutive delays are resolved (up to 100 iterations)
        if idx.shape[0] == 0:
            break

        for i in reversed(idx):
            ids[i] = ids[i + 1] - 1

    # Remaining gaps (lost frames)
    gaps = np.nonzero(np.diff(ids) > 1)[0]
    lost = ids[gaps + 1] - ids[gaps] - 1

    if debug:
        print("\nAnalyzing %s timestamps ..." % title)
        print("fps:", fps)
        print("period:", per / ms, "ms")
        print("offset:", off / ms, "ms")

        if verbose:
            print("\tmin_t:", min_t, "sec")
            print("\trange:", [np.min(dt) / ms, np.max(dt) / ms], "ms")
            # print("ids:", (ts - off) / per)
            print("\tidx:", idx)

        print(len(gaps), "gaps:", ids[gaps] + 1)
        if len(lost):
            print(len(lost), "lost:", lost)

    if plot:
        plt.figure("Timeline - " + title, (16, 9))
        plt.plot(ids * per, ts, ".-", label="Received")
        plt.plot(ids * per, ids * per, "r-", label="Expected")
        plt.xlabel("Frame time, sec")
        plt.ylabel("Timestamp, sec")
        plt.title(title + " Timeline")
        plt.legend()
        plt.tight_layout()

        plt.figure("Periods Hist - " + title, (16, 9))
        plt.hist(ps_n / ms / n, bins=2 * bins // n)
        plt.plot([per_n / ms / n, per_n / ms / n], [0, 100], "r-")
        # plt.semilogy()
        plt.xlabel("Period, ms")
        plt.ylabel("Counts")
        plt.title(title + " Periods" + " (avg = %.3f ms)" % (per_n / ms / n))
        plt.tight_layout()

        plt.figure("Periods All - " + title, (16, 9))
        plt.plot(dt / ms)
        plt.xlabel("Period #")
        plt.ylabel("Period, ms")
        plt.tight_layout()

        plt.figure("Delays - " + title, (16, 9))
        delays = (ts - ids * per) / ms
        plt.plot(ids, delays, "-", label="Delays")
        plt.plot(ids[[0, -1]], [off / ms, off / ms], "k--", label="Offset")

        for i, gap in enumerate(gaps):
            f0, t0, f1, t1 = ids[gap], delays[gap], ids[gap + 1], delays[gap + 1]
            dt = 0.5 * (t1 - t0) / (f1 - f0)
            plt.plot([f0 + 0.5, f1 - 0.5], [t0 + dt, t1 - dt], "m-", label="Gaps" if i == 0 else None)
            x = np.linspace(f0 + 1, f1 - 1, f1 - f0 - 1)
            y = np.linspace(t0 + 2 * dt, t1 - 2 * dt, f1 - f0 - 1)
            plt.plot(x, y, "r.", label="Lost" if i == 0 else None)

        plt.xlabel("Frame ID")
        plt.ylabel("Delay, ms")
        plt.title(title + " Delays")
        plt.legend()
        plt.tight_layout()

        if savefigs:
            pass

    return ids * per
    # return ids * per, ids, (gaps, lost), (per, off)


def correlate(x, y, ax=None, xlabel="x", ylabel="y"):
    mb = np.polyfit(x, y, 1)  # fit a line

    if ax is not None:
        ax.plot(x, y, ".", label="Data points")
        ax.plot(x, np.poly1d(mb)(x), "r-", label="Line fit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("y=%f * x + %f" % (mb[0], mb[1]))
        ax.legend()
        return mb[0], mb[1]


if __name__ == "__main__":
    path = "/home/summer/software/aug2/session_1/"

    # process the json files and merge them into one big file
    # preprocess(path)
    # sorting(path)
    # print("Done merging")

    cam_names, radio_freq = ["0", "1"], 1200
    fixed_gt = []

    # iterate for each sensor
    for s in range(1, 5):

        print("Working on sensor {}".format(s))
        cam_ts = {}

        for name in cam_names:
            with open(path + "sensor_{}/{}_merged.json".format(s, name), "r") as infile:
                # remember to change sessions when you are using different path
                data = json.load(infile)

            t_names = ["python_timestamp", "global_timestamp", "gstreamer_timestamp"]

            cam_ts[name] = np.array([[meta[t_name] for t_name in t_names] for meta in data])

            print(name, cam_ts[name].shape)

        cam_min = np.array([min([np.min(cam_ts[name][:, i]) for name in cam_names]) for i in range(2)])
        cam_max = np.array([max([np.max(cam_ts[name][:, i]) for name in cam_names]) for i in range(2)])

        print(cam_min, cam_max)

        for name in cam_names:
            # relative value of python_timestamp and global timestamp
            cam_ts[name][:, :2] -= cam_min
            cam_ts[name][:, 2] -= np.min(cam_ts[name][:, 2])  # Camera specific clock origin doesn't matter

            cam_ts[name][:, 1] /= radio_freq

        gs_0 = analyze_timestamps(cam_ts["0"][:, 2], "Camera_0", 14.4, verbose=True, plot=False)
        gs_1 = analyze_timestamps(cam_ts["1"][:, 2], "Camera_1", 14.4, verbose=True, plot=False)

        # store the regression parameters
        timestamp_parameters = []
        for i, cam_name in enumerate(cam_names):
            t_names = ["python_timestamp", "global_timestamp", "gstreamer_timestamp"]

            for j, (xi, yi) in enumerate([(2, 0), (0, 1)]):
                xl, yl = t_names[xi], t_names[yi]
                ax = plt.subplot(2, 2, i * 2 + j + 1, title="%s vs %s (%s)" % (yl, xl, cam_name))
                x, y = correlate(cam_ts[cam_name][:, xi], cam_ts[cam_name][:, yi], ax=ax, xlabel=xl, ylabel=yl)
                timestamp_parameters.append((x, y))

        # for the first cam
        # gs to python_s
        python_t_0 = gs_0 * timestamp_parameters[0][0] + timestamp_parameters[0][1]
        # pythons s to gt
        global_t_0 = python_t_0 * timestamp_parameters[1][0] + timestamp_parameters[1][1]
        fixed_global_0 = (global_t_0 * 1200 + cam_min[1]).astype(int)
        fixed_gt.append(fixed_global_0)
        # store the file
        with open(path + "s{}_cam0_gt.json".format(s), 'w') as outfile:
            json.dump(fixed_global_0.tolist(), outfile, indent=4)

        # for the second cam
        python_t_1 = gs_1 * timestamp_parameters[2][0] + timestamp_parameters[2][1]
        global_t_1 = python_t_1 * timestamp_parameters[3][0] + timestamp_parameters[3][1]
        fixed_global_1 = (global_t_1 * 1200 + cam_min[1]).astype(int)
        fixed_gt.append(fixed_global_1)

        with open(path + "s{}_cam1_gt.json".format(s), 'w') as outfile:
            json.dump(fixed_global_1.tolist(), outfile, indent=4)


    # start to generate the master timeline
    end = fixed_gt[0][-1]
    start = fixed_gt[0][0]
    diff = []
    for fix in fixed_gt:
        diff.append(np.mean(np.diff(fix)))
        # take the max of the mins of the gts
        # and min of the max of the gts
        # so that we have a start and end point of the timeline
        start = max(fix[0], start)
        end = min(fix[-1], end)

    # average difference between frames in scale of gt
    avg_diff = np.mean(diff)

    master_gt = np.arange(start, end, avg_diff, dtype=int)

    with open(path + "master_ss1_gt.json", 'w') as outfile:
        json.dump(master_gt.tolist(), outfile, indent=4)







