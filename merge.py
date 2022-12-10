import os
import glob
import json
import joblib
import logging
import subprocess
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from gst_video import *


def find_video_segments(path, default_bitrate=7500, save=True, verbose=True):
    files = sorted(glob.glob(path + "video/*_*_0.avi"))
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
                         # "bitrate, kbps": default_bitrate,
                         "all": ids, "good": good,
                         "skip": skip, "lost": lost})
    if save:
        json.dump(segments, open(path + "video/segments.json", "w"), indent=4)

    return segments


def merge_single_video(pattern, filename, width=2592, height=1944, fps=15, bitrate=20000, codec="h264", gpu=0, overwrite=True, variable=True):
    print("Bitrate", bitrate, "with", codec, "codec for", filename)
    if os.path.exists(filename) and not overwrite:
        print(filename, "already exists. Skipping...")
        return

    gstream = "multifilesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! " \
              "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
              "appsink" % (pattern, width, height, fps)

    reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
    # total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))  # return -1 because of chunked format
    writer = GstVideo(filename, fps, width, height, bitrate=bitrate, variable=variable, gpu=gpu, format="BGR", codec=codec)

    ret, frame = reader.read()
    i = 1
    while ret:
        writer.write(frame)
        if i % 10 == 0:
            print(i, "done")
        ret, frame = reader.read()
        i += 1

    writer.close()


def merge_single_video_fast(pattern, filename, width=2592, height=1944, fps=15, bitrate=20000, codec="h264", variable=True, gpu=0, overwrite=True, verbose=False):
    print("Bitrate", bitrate, "with", codec, "codec for", filename)
    if os.path.exists(filename) and not overwrite:
        print(filename, "already exists. Skipping...")
        return

    print(filename)
    if codec == "h264":
        assert filename[-3:] == "mp4"
    elif codec == "h265":
        assert filename[-3:] == "mkv"
    else:
        raise ValueError("Unsupported codec %d" % codec)

    cmd = "multifilesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! videoconvert ! " \
          "nv%senc preset=hq bitrate=%d rc-mode=%s gop-size=45 cuda-device-id=%d ! %sparse ! matroskamux ! filesink location=%s" \
          % (pattern, width, height, fps, codec, bitrate, "vbr" if variable else "cbr", gpu, codec, filename)

    os.environ.update({"GST_DEBUG": "2,filesink:4,GST_EVENT:4" if verbose else "2,filesink:4"})
    proc = subprocess.Popen(["gst-launch-1.0", *(cmd.split(" "))], stdout=subprocess.PIPE, universal_newlines=True)

    for line in proc.stdout:
        print(line[:-1])


def diff(filename, bitrate, max_frames, codec, width=2592, height=1944, fps=15):
    gstream = "filesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! " \
              "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
              "appsink" % (filename, width, height, fps)
    original = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
    # original = cv2.VideoCapture(filename)  # .avi
    # original_frames = int(original.get(cv2.CAP_PROP_FRAME_COUNT)) - 1  # OpenCV bug - overestimates by 1 (but not always)

    # n = 0
    # ret, frame = original.read()
    # while ret:
    #     n += 1
    #     ret, frame = original.read()
    #
    # print("real total frames:", n)
    # original = cv2.VideoCapture(filename)

    # base_name = filename[filename.rfind("/"):-6]
    dir_name = filename[:-6] + ("_%s/" % codec)
    out_name = dir_name + str(bitrate) + "_Mbps" + (".mp4" if codec == "h264" else ".mkv")
    encoded = cv2.VideoCapture(out_name)  # .mp4 or .mkv
    # encoded_frames = int(encoded.get(cv2.CAP_PROP_FRAME_COUNT))

    # if original_frames != encoded_frames:
    #     print("!!! Number of frames mismatch:", original_frames, "vs", encoded_frames)

    # errors, n = [], min(original_frames, encoded_frames, max_frames)
    errors = []
    for i in range(max_frames):
        ref_good, ref = original.read()
        enc_good, enc = encoded.read()
        if not ref_good or not enc_good:
            break

        # ref[ref < 16] = 0
        # ref[ref > 235] = 255
        # ch = 2
        # plt.figure("ref")
        # plt.imshow(ref[:, :, ch])
        # plt.figure("enc")
        # plt.imshow(enc[:, :, ch])
        # plt.figure("diff")
        # plt.imshow(ref[:, :, ch].astype(np.float) - enc[:, :, ch].astype(np.float), cmap="bwr")
        # plt.colorbar()

        ref16 = ref.ravel()[::20].astype(np.int16)
        diff = np.abs(ref16 - enc.ravel()[::20])
        rel = diff / (ref16 + 1)
        avg_abs, avg_rel = np.average(diff), np.average(rel)
        hist, edges = np.histogram(diff, bins=256, range=(-0.5, 255.5))
        errors.append({"avgs": [float(avg_abs), float(avg_rel)], "hist": hist.tolist()})

        if i % 50 == 0:
            print("diff for", i, "of", max_frames, "in", out_name, "=", [avg_abs, avg_rel * 100])

    return errors


def bitrate_sweep(filename, width=2592, height=1944, fps=15, max_frames=150, gpu=0, codec="h264",
                  bitrates=(100, 65, 40, 25, 15, 10, 6.5, 4, 2.5, 1.5, 1, 0.65, 0.4, 0.25, 0.15, 0.1), variable=False):
    # base_name = filename[filename.rfind("/"):-6]
    dir_name = filename[:-6] + ("_%s/" % codec)
    for bitrate in bitrates:
        print("Bitrate", bitrate, "with", codec, "codec for", filename)
        out_name = dir_name + str(bitrate) + "_Mbps" + (".mp4" if codec == "h264" else ".mkv")

        if os.path.exists(out_name):
            print(out_name, "already exists. Skipping...")
            continue

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        cmd = "filesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! jpegdec ! identity eos-after=151 ! videoconvert ! " \
          "nv%senc preset=hq bitrate=%d rc-mode=%s gop-size=45 cuda-device-id=%d ! %sparse ! matroskamux ! filesink location=%s" \
          % (filename, width, height, fps, codec, bitrate * 1000, "vbr" if variable else "cbr", gpu, codec, out_name)

        print(cmd)
        os.environ.update({"GST_DEBUG": "2,filesink:4"})
        proc = subprocess.Popen(["gst-launch-1.0", *(cmd.split(" "))], stdout=subprocess.PIPE, universal_newlines=True)

        for line in proc.stdout:
            print(line[:-1])

    json_file = dir_name + "sweep.json"
    if not os.path.exists(json_file):
        jobs = [joblib.delayed(diff)(filename, bitrate, max_frames, codec) for bitrate in bitrates]
        results = joblib.Parallel(verbose=15, n_jobs=-1, batch_size=len(bitrates), pre_dispatch='all', backend="multiprocessing")(jobs)

        all_errors = {str(bitrate): results[i] for i, bitrate in enumerate(bitrates)}
        json.dump(all_errors, open(json_file, "w"), indent=4)

    sweep = json.load(open(json_file, "r"))

    for err_type in ["Absolute", "Relative"]:
        n, m = 5, 0
        plt.figure(filename + " - " + err_type, (24, 13.5))
        plt.gcf().clear()
        for j, bitrate in enumerate(reversed(bitrates)):
            errors = np.array([errs["avgs"] for errs in sweep[str(bitrate)]])
            hists = np.array([errs["hist"] for errs in sweep[str(bitrate)]])
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
        plt.legend()
        plt.tight_layout()
        plt.savefig(json_file[:-5] + ("_abs" if err_type == "Absolute" else "_rel") + ".png", dpi=160)

    # plt.show()


if __name__ == '__main__':
    num_gpus, n = 1, 3  # Maximum of n jobs to be scheduled at the same time (limited to 3 per GPU by the driver)
    data_path = '../data/'
    # merge_single_video(data_path + "chase_2/sensor_1/video/0_1659466350_%d.avi", data_path + "0_1659466350.mp4")
    # merge_single_video_fast(data_path + "chase_2/sensor_1/video/0_1659466350_%d.avi", data_path + "0_1659466350.mp4")
    # diff(data_path + "chase_2/sensor_1/video/0_1659466350_0.avi", 100, 10, "h264")
    # plt.show()
    # exit(0)

    for session in sorted(glob.glob(data_path + "*/")):
        print("\n*******")
        print("Session", session)
        print("*******\n")

        bitrates = json.load(open(session + "meta/video_qualities.json", "r"))

        for i in range(4):
            sensor = session + "sensor_%d/" % (i+1)
            video, time, audio = sensor + "video/", sensor + "time/", sensor + "audio/"

            if not os.path.exists(video + "segments.json"):
                segments = find_video_segments(sensor, verbose=False)
            else:
                segments = json.load(open(video + "segments.json", "r"))

            print("\n%d segments in" % len(segments), video)
            for seg in segments:
                print("\t", len(seg["good"]), "out of", len(seg["all"]), "for", seg["pattern"], "are good. Lost:", seg["lost"])
            print()

            for codec in ["h265", "h264"]:
                #####################################################################################
                # First pass - do a bitrate sweep by reencoding the first chunk of each video segment
                #####################################################################################

                # Sequencial processing - supports plotting
                for seg in segments:
                    bitrate_sweep(video + seg["pattern"] % 0, codec=codec)
                # plt.show()

                # Parallel processing - joblib nesting is not supported (and would be useless with <= 16 cores)
                # jobs = [joblib.delayed(bitrate_sweep)(video + seg["pattern"] % 0, codec=codec, gpu=int(seg["prefix"][0]) % num_gpus) for seg in segments]
                # joblib.Parallel(verbose=15, n_jobs=n, batch_size=n, pre_dispatch=n, backend="threading")(jobs)

                #####################################################################################
                # User Interaction - adjusting default bitrates
                #####################################################################################

                # print("\n*****")
                # print("Enter \'c\' to continue after adjusting bitrate settings in " + video + "segments.json")
                # print("*****\n")
                # c = ""
                # while c != "c":
                #     c = input()
                # segments = json.load(open(video + "segments.json", "r"))
                bitrate = bitrates[codec]["sensor_%d" % (i+1)]

                #####################################################################################
                # Second pass - re-encode all video segments using optimal bitrate setting
                #####################################################################################

                # Sequencial processing
                # for seg in segments:
                #     cam = int(seg["prefix"][0])
                #     merge_single_video_fast(video + seg["pattern"], video + seg["prefix"] + (".mp4" if codec == "h264" else ".mkv"),
                #                             bitrate=bitrate[cam], codec=codec, gpu=cam % num_gpus)

                # Parallel processing - faster with multiple GPUs
                jobs = [joblib.delayed(merge_single_video_fast)(video + seg["pattern"], video + seg["prefix"] + (".mp4" if codec == "h264" else ".mkv"),
                                                                bitrate=bitrate[int(seg["prefix"][0])], codec=codec,
                                                                gpu=int(seg["prefix"][0]) % num_gpus, overwrite=False) for seg in segments]
                joblib.Parallel(verbose=15, n_jobs=n, batch_size=n, pre_dispatch=n, backend="multiprocessing")(jobs)
