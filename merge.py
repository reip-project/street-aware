import re
import os
import glob
import json
import joblib
import logging
import subprocess
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from gst_video import *

W, H, FPS = 2592, 1944, 15
ALL_BITRATES = (100, 65, 40, 25, 15, 10, 6.5, 4, 2.5, 1.5, 1, 0.65, 0.4, 0.25, 0.15, 0.1)  # Mbps
DEFAULT_BITRATE = 10000  # kbps


def find_video_segments(path, save=True, verbose=True):
    files = sorted(glob.glob(path + "*_*_0.avi"))
    if verbose:
        print("\n%d segments in" % len(files), path)

    segments = []
    for file in files:
        prefix = file[:-6]
        chunks = glob.glob(prefix + "_*.avi")
        ids = sorted([int(chunk[chunk.rfind("_")+1:chunk.rfind(".")]) for chunk in chunks])

        good = []
        for i, ch in enumerate(ids):
            if i != ch:
                break
            good.append(i)

        skip, lost = ids[len(good):], []
        for i in range(len(good), ids[-1]+1):
            if i not in skip:
                lost.append(i)

        if verbose:
            print("\n", prefix, "with", len(chunks), "chunks:")
            print(" all:", ids)
            print("good:", good)
            print("skip:", skip)
            print("lost:", lost)

        prefix = prefix[prefix.rfind("/")+1:]
        segments.append({"pattern": prefix + "_%d.avi", "prefix": prefix,
                         "all": ids, "good": good,
                         "skip": skip, "lost": lost})
    if save:
        json.dump(segments, open(path + "segments.json", "w"), indent=4)

    return segments


def merge_single_video(pattern, filename, width=W, height=H, fps=FPS, divider=1,
                       bitrate=DEFAULT_BITRATE, variable=True, codec="h265", gpu=0, overwrite=True):
    print("\nBitrate", bitrate, "kbps with", codec, "codec for", filename)
    if os.path.exists(filename) and not overwrite:
        print(filename, "already exists. Skipping...")
        return

    gstream = "multifilesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! " \
              "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
              "appsink" % (pattern, width, height, fps)

    reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
    # total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))  # returns -1 because of chunked format
    writer = GstVideo(filename, width // divider, height // divider, fps, format="BGR",
                      bitrate=bitrate, variable=variable, codec=codec, gpu=gpu)

    ret, frame = reader.read()
    i = 1
    while ret:
        writer.write(frame[::divider, ::divider, :])
        if i % 10 == 0:
            print(i, "done")
        ret, frame = reader.read()
        i += 1

    writer.close()


def merge_single_video_fast(pattern, filename, width=W, height=H, fps=FPS, divider=1,
                            bitrate=DEFAULT_BITRATE, variable=True, codec="h265", gpu=0, overwrite=True,
                            verbose=False, save_log=None):
    print("\nBitrate", bitrate, "kbps with", codec, "codec for", filename)
    if os.path.exists(filename) and not overwrite:
        print(filename, "already exists. Skipping...")
        return

    assert codec in ["h264", "h265"], "Unsupported codec %d" % codec

    cmd = "multifilesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! videoconvert ! " \
          "videoscale ! video/x-raw, width=%d, height=%d ! " \
          "nv%senc preset=hq bitrate=%d rc-mode=%s gop-size=45 cuda-device-id=%d ! %sparse ! matroskamux ! filesink location=%s" \
          % (pattern, width, height, fps, width // divider, height // divider, codec, bitrate, "vbr" if variable else "cbr", gpu, codec, filename)

    print(cmd)

    if save_log is True:
        save_log = filename[:-3] + "log"
    elif save_log is False:
        save_log = None

    os.environ.update({"GST_DEBUG": "2,filesink:4,GST_EVENT:4" if verbose else "2,filesink:4"})#,
                       # "GST_DEBUG_FILE": "" if save_log is None else save_log + "_gst"})
    proc = subprocess.Popen(["gst-launch-1.0", *(cmd.split(" "))], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    if type(save_log) is str:
        save_log = open(save_log, "w")

    if save_log is not None:
        save_log.write(cmd + "\n\n")

    for line in proc.stdout:
        print(line[:-1])
        if save_log is not None:
            line = re.sub(r'\x1b\[[\d;]+m', '', line)
            save_log.write(line)

    if save_log is not None:
        save_log.close()


def count_frames(source):
    if type(source) is str:
        reader = cv2.VideoCapture(source)
    else:
        reader = source

    n = 0
    ret, frame = reader.read()
    while ret:
        n += 1
        ret, frame = reader.read()

    return n


def diff(filename, max_frames, bitrate, codec, width=W, height=H, fps=FPS, debug=False):
    gstream = "filesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! " \
              "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
              "appsink" % (filename, width, height, fps)
    original = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)  # original .avi chunk without critical warnings
    # original = cv2.VideoCapture(filename)  # cv2.CAP_PROP_FRAME_COUNT might not be set properly
    # original_frames = int(original.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # OpenCV bug - overestimates by 1 (but not always)
    # original_frames = count_frames(filename)

    out_name = filename[:-6] + ("_%s/" % codec) + str(bitrate) + "_Mbps" + (".mp4" if codec == "h264" else ".mkv")
    encoded = cv2.VideoCapture(out_name)  # re-encoded .mp4 or .mkv
    # encoded_frames = int(encoded.get(cv2.CAP_PROP_FRAME_COUNT))  # Works well with matroska container
    # print(encoded_frames)
    # encoded_frames = count_frames(out_name)
    # print(encoded_frames)

    # if original_frames != encoded_frames:
    #     print("!!! Number of frames mismatch:", original_frames, "vs", encoded_frames)

    errors = []
    for i in range(max_frames):
        ref_good, ref = original.read()
        enc_good, enc = encoded.read()
        if not ref_good or not enc_good:
            break

        if debug:
            # ref[ref < 16] = 0
            # ref[ref > 235] = 255
            ch = 2
            plt.figure("ref")
            plt.imshow(ref[:, :, ch])
            plt.figure("enc")
            plt.imshow(enc[:, :, ch])
            plt.figure("diff")
            plt.imshow(ref[:, :, ch].astype(np.float64) - enc[:, :, ch].astype(np.float64), cmap="bwr")
            plt.colorbar()

        ref16 = ref.ravel()[::20].astype(np.int16)
        diff = np.abs(ref16 - enc.ravel()[::20])
        rel = diff / (ref16 + 1)
        avg_abs, avg_rel = np.average(diff), np.average(rel)
        hist, edges = np.histogram(diff, bins=256, range=(-0.5, 255.5))
        errors.append({"avgs": [float(avg_abs), float(avg_rel)], "hist": hist.tolist()})

        if i % 50 == 0:
            print("diff for", i, "of", max_frames, "in", out_name, "=", [avg_abs, avg_rel * 100])

    return errors


def load_sweep(filename):
    sweep = json.load(open(filename, "r"))
    errors, hists = {}, {}
    for bitrate in sweep.keys():
        errors[bitrate] = np.array([errs["avgs"] for errs in sweep[bitrate]])
        hists[bitrate] = np.array([errs["hist"] for errs in sweep[bitrate]])
    return errors, hists


def bitrate_sweep(filename, width=W, height=H, fps=FPS, max_frames=150, bitrates=ALL_BITRATES, variable=False, codec="h264", gpu=0):
    dir_name = filename[:-6] + ("_%s/" % codec)

    for bitrate in bitrates:
        print("\nBitrate", bitrate, "Mbps with", codec, "codec for", filename)
        out_name = dir_name + str(bitrate) + "_Mbps" + (".mp4" if codec == "h264" else ".mkv")

        if os.path.exists(out_name):
            print(out_name, "already exists. Skipping...")
            continue

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        cmd = "filesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! identity eos-after=%d ! videoconvert ! " \
          "nv%senc preset=hq bitrate=%d rc-mode=%s gop-size=45 cuda-device-id=%d ! %sparse ! matroskamux ! filesink location=%s" \
          % (filename, width, height, fps, max_frames + 2, codec, bitrate * 1000, "vbr" if variable else "cbr", gpu, codec, out_name)

        print(cmd)
        os.environ.update({"GST_DEBUG": "2,filesink:4"})
        proc = subprocess.Popen(["gst-launch-1.0", *(cmd.split(" "))], stdout=subprocess.PIPE, universal_newlines=True)

        for line in proc.stdout:
            print(line[:-1])

    json_file = dir_name + "sweep.json"
    if not os.path.exists(json_file):
        jobs = [joblib.delayed(diff)(filename, max_frames, bitrate, codec) for bitrate in bitrates]
        results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=len(bitrates), pre_dispatch='all', backend="multiprocessing")(jobs)

        all_errors = {str(bitrate): results[i] for i, bitrate in enumerate(bitrates)}
        json.dump(all_errors, open(json_file, "w"), indent=4)

    # sweep = json.load(open(json_file, "r"))
    all_errors, all_hists = load_sweep(json_file)

    for err_type in ["Absolute", "Relative"]:
        n, m = 5, 0
        plt.figure(filename + " - " + err_type, (24, 13.5))
        plt.gcf().clear()
        for j, bitrate in enumerate(reversed(bitrates)):
            # errors = np.array([errs["avgs"] for errs in sweep[str(bitrate)]])
            # hists = np.array([errs["hist"] for errs in sweep[str(bitrate)]])
            errors, hists = all_errors[str(bitrate)], all_hists[str(bitrate)]
            m = errors.shape[0] - n
            errors = errors[:, 0] if err_type == "Absolute" else errors[:, 1] * 100
            plt.plot(np.convolve(errors, np.ones(n), 'valid') / n, "" if j == 0 or ((j-1) % 5 == 4) else "--",
                                                                   label=str(bitrate) + " Mbps")
        plt.plot([0, m], [0, 0], "--k")
        plt.title(err_type + " encoding errors for " + filename[filename.rfind("/")+1:] + " vs bitrate (%d frames moving average)" % n)
        plt.xlabel("Frame ID")
        if err_type == "Absolute":
            plt.ylabel("Average difference, 8-bit brightness")
        else:
            plt.ylabel("Relative error, %")
        plt.xlim([-1, m+1])
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(json_file[:-5] + ("_abs" if err_type == "Absolute" else "_rel") + ".png", dpi=160)

    # plt.show()


if __name__ == '__main__':
    num_gpus, n = 2, 3  # Maximum of n jobs to be scheduled at the same time (limited to 3 per GPU by the driver)
    data_path = '../data/'
    # merge_single_video(data_path + "chase_2/sensor_1/video/0_1659466350_%d.avi", data_path + "0_1659466350.mp4", divider=2)
    # merge_single_video_fast(data_path + "chase_2/sensor_1/video/0_1659466350_%d.avi", data_path + "0_1659466350.mp4", divider=2, save_log=True)
    # diff(data_path + "chase_2/sensor_1/video/0_1659466350_0.avi", 1, 100, "h264", debug=True)
    # plt.show()
    # exit(0)

    #####################################################################################
    # First pass - do a bitrate sweep by reencoding the first chunk of each video segment
    #####################################################################################

    for session in sorted(glob.glob(data_path + "*/")):
        # break
        print("\n*******")
        print("Session", session)
        print("*******\n")

        for i in range(4):
            sensor = session + "sensor_%d/" % (i+1)
            video = sensor + "video/"

            segments = find_video_segments(video, save=True, verbose=True)

            for codec in ["h265", "h264"]:
                # Sequencial processing - supports plotting
                old_cam = -1
                for seg in segments:
                    cam = int(seg["prefix"][0])
                    if cam == old_cam:
                        continue  # sweep first segment only for each camera
                    old_cam = cam
                    bitrate_sweep(video + seg["pattern"] % 0, codec=codec)
                # plt.show()

                # Parallel processing - joblib nesting is not supported (and would be useless with <= 16 cores)
                # jobs = [joblib.delayed(bitrate_sweep)(video + seg["pattern"] % 0, codec=codec,
                #                                       gpu=int(seg["prefix"][0]) % num_gpus) for seg in segments]
                # joblib.Parallel(verbose=15, n_jobs=n, batch_size=n, pre_dispatch=n, backend="threading")(jobs)

    #####################################################################################
    # User interaction - choose optimal bitrates
    #####################################################################################

    exit(0)
    print("\nPress enter to continue with merging...")
    _ = input()

    #####################################################################################
    # Second pass - re-encode all video segments using optimal bitrate settings
    #####################################################################################

    for session in sorted(glob.glob(data_path + "*/")):
        print("\n*******")
        print("Session", session)
        print("*******\n")

        renders = json.load(open(session + "meta/video_qualities.json", "r"))

        for i in range(4):
            sensor = session + "sensor_%d/" % (i+1)
            video = sensor + "video/"

            segments = json.load(open(video + "segments.json", "r"))

            print("\n%d segments in" % len(segments), video)
            for seg in segments:
                print("\t", len(seg["good"]), "out of", len(seg["all"]), "for", seg["pattern"], "are good. Lost:", seg["lost"])
            print()

            for render in renders:
                codec, suffix, divider = render["codec"], render["suffix"], render["divider"]
                qualities = render["qualities"]["sensor_" + str(i+1)]
                qmap = {"low": 0, "mid": 1, "high": 2}
                bitrates = [render["bitrates"][qmap[q]] for q in qualities]

                # Sequencial processing
                # for seg in segments:
                #     cam = int(seg["prefix"][0])
                #     merge_single_video_fast(video + seg["pattern"], video + seg["prefix"] + suffix + (".mp4" if codec == "h264" else ".mkv"),
                #                             bitrate=bitrates[cam], codec=codec, divider=divider, save_log=codec=="h265", gpu=cam % num_gpus, overwrite=False)

                # Parallel processing - faster with multiple GPUs
                jobs = [joblib.delayed(merge_single_video_fast)(video + seg["pattern"], video + seg["prefix"] + suffix + (".mp4" if codec == "h264" else ".mkv"),
                                                                bitrate=bitrates[int(seg["prefix"][0])], codec=codec, divider=divider, save_log=codec=="h265",
                                                                gpu=int(seg["prefix"][0]) % num_gpus, overwrite=False) for seg in segments]
                joblib.Parallel(verbose=15, n_jobs=n, batch_size=n, pre_dispatch=n, backend="multiprocessing")(jobs)
