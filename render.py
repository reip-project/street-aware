import gc

from merge import *
from hrnet_pose import *


def find_time_span(session):
    l, r, per, n = 10e+9, 0, 0, 0

    for i in range(4):
        video = session + "sensor_%d/" % (i + 1) + "video/"
        segments = json.load(open(video + "segments.json", "r"))
        for segment in segments:
            data = json.load(open(video + segment['prefix'] + "_sync.json"))
            per += data["average_period"]
            l = min(l, np.min(data["global_timestamps"]))
            r = max(r, np.max(data["global_timestamps"]))
            n += 1

    assert n > 0, "No segments found!"
    per /= n
    return l, r, per


class SegmentReader:
    def __init__(self, path, prefix, tolerance=1.0, anonymize=True):
        # self.path, self.segment = path, segment
        self.tolerance, self.anonymize = tolerance, anonymize
        self.reader = cv2.VideoCapture(path + prefix + ".mkv")
        self.timestamps = json.load(open(path + prefix + "_sync.json"))["global_timestamps"]
        self.meta = json.load(open(path + prefix + "_poses.json"))
        self.num_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.num_frames != len(self.meta):
            print("Metadata mismatch:", self.num_frames, len(self.meta))
        self.num_frames = min(self.num_frames, len(self.meta), len(self.timestamps))
        print(self.num_frames, "frames total in", path + prefix + ".mkv")

        self.i = 0
        ret, self.lframe = self.reader.read()
        assert ret
        ret, self.rframe = self.reader.read()
        assert ret
        self.lt = self.timestamps[self.i]
        self.rt = self.timestamps[self.i + 1]
        self.lmeta = self.meta[self.i]
        self.rmeta = self.meta[self.i + 1]
        if self.anonymize:
            self._anonymize(self.lframe, self.lmeta)
            self._anonymize(self.rframe, self.rmeta)

    def _anonymize(self, frame, meta):
        if meta is not None:
            if len(meta["poses"]) == 0:
                return
            blurred = cv2.GaussianBlur(frame, (50, 50), cv2.BORDER_DEFAULT)
            mask = np.zeros_like(frame)
            for pose in meta["poses"]:
                pose = np.array(pose)
                f_min = np.min(pose[:5, :], axis=0)
                f_max = np.max(pose[:5, :], axis=0)
                f_c = (f_min + f_max) // 2
                f_s = (f_max - f_min) // 2
                f_s[0] = np.round(f_s[0] * 1.2)
                f_s[1] = np.round(f_s[0] * 1.5)
                # cv2.rectangle(frame, (f_c[0] - f_s[0], f_c[1] - f_s[1]), (f_c[0] + f_s[0], f_c[1] + f_s[1]), (0, 255, 0), 2)
                # cv2.ellipse(frame, (f_c[0], f_c[1]), (f_s[0], f_s[1]), 0, 0, 360, (0, 255, 0), -1)
                cv2.ellipse(mask, (f_c[0], f_c[1]), (f_s[0], f_s[1]), 0, 0, 360, (255, 255, 255), -1)
            idx = np.nonzero(mask)
            frame[idx] = blurred[idx]

    def _advance(self):
        ret, new_frame = self.reader.read()
        if ret and self.i < self.num_frames-2:
            self.lframe = self.rframe
            self.rframe = new_frame
            self.i += 1
            self.lt = self.timestamps[self.i]
            self.rt = self.timestamps[self.i + 1]
            self.lmeta = self.meta[self.i]
            self.rmeta = self.meta[self.i + 1]
            if self.anonymize:
                self._anonymize(self.rframe, self.rmeta)
            return True
        elif self.rt < 10e+9:
            self.lframe = self.rframe
            self.rframe = np.zeros_like(self.lframe)
            self.i += 1
            self.lt = self.timestamps[self.i]
            self.rt = 10e+9
            self.lmeta = self.meta[self.i]
            self.rmeta = None
            return True
        return False

    def _forward_seek(self, t):
        if t > self.rt:
            while self._advance():
                if t <= self.rt:
                    break

    def get_frame(self, t):
        self._forward_seek(t)
        mid_t = (self.lt + self.rt) / 2
        if t < mid_t:
            if abs(t - self.lt) < self.tolerance:
                return self.lframe, self.lmeta
            else:
                return np.zeros_like(self.lframe), None
        else:
            if abs(t - self.rt) < self.tolerance:
                return self.rframe, self.rmeta
            else:
                return np.zeros_like(self.rframe), None

    def release(self):
        self.reader.release()


class SessionReader:
    def __init__(self, path, segments, tolerance=1.0, anonymize=True, cam_id=None):
        assert len(segments) > 0, "Not enough segments!"
        self.prefixes = sorted([segment["prefix"] for segment in segments])
        if cam_id is not None:
            self.prefixes = [prefix for prefix in self.prefixes if prefix.startswith(str(cam_id))]
        print("SessionReader:", path, "+", self.prefixes)
        self.readers = [SegmentReader(path, prefixe, tolerance, anonymize) for prefixe in self.prefixes]
        self.ri = 0
        self.current = self.readers[self.ri]

    def get_frame(self, t):
        frame, meta = self.current.get_frame(t)
        if self.ri < len(self.readers) - 1:
            if self.current.rt == 10e+9 and (t - self.current.lt) > self.current.tolerance:
                self.ri += 1
                self.current = self.readers[self.ri]
                frame, meta = self.current.get_frame(t)
        return frame, meta

    def release(self):
        for reader in self.readers:
            reader.release()


def render_single(path, segments, timing, cam_id, filename, bitrate, use_gpu=0, max_frames=None, overwrite=False):
    full_filename = path + "../" + filename + ".mp4"
    if os.path.exists(full_filename) and not overwrite:
        print(full_filename, "already exists. Skipping...")
        meta_filename = path + "../" + filename + "_meta.json"
        if os.path.exists(meta_filename):
            print("Renaming meta:", meta_filename)
            os.rename(meta_filename, path + "../" + filename + "_poses.json")
        return

    reader = SessionReader(path, segments, tolerance=1.1*timing[2], anonymize=True, cam_id=cam_id)
    writer = GstVideo(full_filename, 2592, 1944, 1200 / timing[2], format="BGR",
                      bitrate=bitrate, variable=True, codec='h264', gpu=use_gpu)

    all_metas = []
    for i, t in enumerate(np.arange(timing[0], timing[1] + timing[2], timing[2])):
        if max_frames and i >= max_frames:
            break
        if i > 0 and i % 100 == 0:
            print(i, "frames rendered of", max_frames, "in", path + "../" + filename + ".mp4")
            gc.collect()

        frame, meta = reader.get_frame(t)
        writer.write(frame)
        all_metas.append(meta)

    with open(path + "../" + filename + "_poses.json", "w") as f:
        json.dump(all_metas, f, indent=None, cls=NumpyEncoder)

    with open(path + "../" + filename + ".json", "w") as f:
        json.dump({"start": timing[0], "stop": timing[1],
                   "period": timing[2], "frequency, Hz": 1200,
                   "max_frames": max_frames}, f, indent=4, cls=NumpyEncoder)
    writer.close()
    reader.release()


def render_all(sessions, use_gpu=0, skip_cameras=True, max_frames=None, prep_jobs_only=False, overwrite=False):
    jobs = []

    for session in browse_sessions(sessions):
        renders = json.load(open(session + "meta/video_qualities.json", "r"))
        orders = json.load(open(session + "meta/camera_orders.json", "r"))
        quality = renders[0]
        assert quality["codec"] == "h265"

        l, r, per = find_time_span(session)
        print(l, r, (r-l)/1200, per)

        for sensor_i in range(4):
            video = session + "sensor_%d/" % (sensor_i + 1) + "video/"
            segments = json.load(open(video + "segments.json", "r"))

            for cam_id in range(2):
                if skip_cameras and cam_id != use_gpu:
                    print("Skipping camera", cam_id, "in", video)
                    continue

                q = quality["qualities"]["sensor_%d" % (sensor_i+1)][cam_id]
                q = {"low": 0, "mid": 1, "high": 2}[q]
                bitrate = quality["bitrates"][q]
                filename = orders["sensor_%d" % (sensor_i+1)][cam_id]

                if prep_jobs_only:
                    jobs.append(joblib.delayed(render_single)(video, segments, (l, r, per), cam_id, filename, bitrate, use_gpu=use_gpu, max_frames=max_frames, overwrite=overwrite))
                else:
                    render_single(video, segments, (l, r, per), cam_id, filename, bitrate, use_gpu=use_gpu, max_frames=max_frames, overwrite=overwrite)
    return jobs


def subsample_single(filename, suffix, divider=1, bitrate=DEFAULT_BITRATE, variable=True, codec="h264", gpu=0, overwrite=True):
    p = filename.rfind('.')
    new_filename = filename[:p] + suffix + filename[p:]
    print("\nBitrate", bitrate, "kbps with", codec, "codec for", new_filename)
    if os.path.exists(new_filename) and not overwrite:
        print(new_filename, "already exists. Skipping...")
        return

    assert codec in ["h264", "h265"], "Unsupported codec %d" % codec

    cmd = "filesrc location=%s ! decodebin ! videoconvert ! " \
          "videoscale ! video/x-raw, width=%d, height=%d ! " \
          "nv%senc preset=hq bitrate=%d rc-mode=%s gop-size=45 cuda-device-id=%d ! %sparse ! matroskamux ! filesink location=%s" \
          % (filename, W // divider, H // divider, codec, bitrate, "vbr" if variable else "cbr", gpu, codec, new_filename)

    print(cmd)
    # return

    os.environ.update({"GST_DEBUG": "2,filesink:4,GST_EVENT:3"})
    proc = subprocess.Popen(["gst-launch-1.0", *(cmd.split(" "))], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    for line in proc.stdout:
        print(line[:-1])


def subsample_all(sessions, num_gpus=2):
    jobs = []

    for session in browse_sessions(sessions):
        renders = json.load(open(session + "meta/video_qualities.json", "r"))
        orders = json.load(open(session + "meta/camera_orders.json", "r"))

        for render in renders:
            for i in range(4):
                codec, suffix, divider = render["codec"], render["suffix"], render["divider"]
                qualities = render["qualities"]["sensor_" + str(i + 1)]
                qmap = {"low": 0, "mid": 1, "high": 2}
                bitrates = [render["bitrates"][qmap[q]] for q in qualities]

                if codec != 'h264':
                    continue

                for cam_id in range(2):
                    cam_name = orders["sensor_%d" % (i+1)][cam_id]
                    jobs.append(joblib.delayed(subsample_single)(session + "sensor_%d/" % (i+1) + cam_name + ".mp4", render["suffix"],
                                                                 divider=render["divider"], bitrate=bitrates[cam_id], gpu=cam_id % num_gpus, overwrite=False))
    return jobs


def render_mosaic(session, scale=3, max_frames=None, save_images=True, save_every=1000, use_gpu=0, overwrite=False):
    if os.path.exists(session + "mosaic.mp4") and not overwrite:
        print(session + "mosaic.mp4", "already exists. Skipping...")
        return

    readers = []
    for sensor_id in range(4):
        for cam_id in ["left", "right"]:
            filename = session + "sensor_%d/" % (sensor_id + 1) + cam_id + ".mp4"
            readers.append(cv2.VideoCapture(filename))

    meta = json.load(open(session + "sensor_1/left.json"))
    fps = meta["frequency, Hz"] / meta["period"]
    w, h = W // scale, H // scale
    writer = GstVideo(session + "mosaic.mp4", 4*w, 2*h, fps, format="BGR",
                      bitrate=DEFAULT_BITRATE, variable=True, codec='h264', gpu=use_gpu)
    if save_images:
        if not os.path.exists(session + "mosaic/"):
            os.mkdir(session + "mosaic/")

    tot_frames = 0
    while True:
        frames = []
        for reader in readers:
            ret, frame = reader.read()
            if not ret:
                frames = None
                break
            else:
                frames.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))

        if frames is None:
            break

        big_frame = np.zeros((2*h, 4*w, 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            r, c = i // 4, i % 4
            big_frame[r*h:(r*h+h), c*w:(c*w+w), :] = frame

        if save_images and (tot_frames % save_every) < 21:
            cv2.imwrite(session + "mosaic/%d.jpg" % tot_frames, big_frame)

        writer.write(big_frame)
        tot_frames += 1

        if tot_frames % 100 == 0:
            print(tot_frames, "frames rendered of", max_frames, "in", session)
            gc.collect()

        if max_frames is not None and tot_frames >= max_frames:
            break

    for reader in readers:
        reader.release()

    writer.close()


def pad_mosaic(filename):
    img = cv2.imread(filename)
    scale = 3
    w, h = 2592 // scale, 1944 // scale
    pad, off = 100, 150

    img[h:2*h, 2*w:3*w, :] = 0

    pimg = np.ones((h*2+pad+2*off, w*4+pad*3+2*off, 3), np.uint8) * 255
    for i in range(2):
        for j in range(4):
            pimg[off+i*(h+pad):off+i*(h+pad)+h, off+j*(w+pad):off+j*(w+pad)+w, :] = img[i*(h+0):i*(h+0)+h, j*(w+0):j*(w+0)+w, :]

    H, W, d = pimg.shape[0], pimg.shape[1], off - pad
    pimg[:, :W//2, :] = pimg[:, d:d+W//2, :]
    pimg[:, W//2:, :] = pimg[:, W//2-d:W-d, :]
    pimg[H//2:, :, :] = pimg[H//2-3*d//2:H-3*d//2, :, :]
    pimg[H//2-d//2:H//2+d//2, :, :] = 255

    p2 = 25
    for i in range(2):
        for j in range(2):
            l, t = off-d + j*(2*w+2*pad+2*d) - p2, off + i*(h+pad+3*d//2) - p2
            r, b = l + 2*w+pad + 2*p2, t + h + 2*p2
            cv2.rectangle(pimg, (l, t), (r, b), (0, 165, 255), 7)
            cv2.putText(pimg, "Sensor %d" % (i*2 + j + 1), ((l+r)//2 - 150, t-p2-5), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (0, 0, 0), 4, cv2.LINE_AA)

    cv2.imwrite(filename[:-4] + "_padded.jpg", pimg)


def extract_frames_with_poses(filename, start_frame=5000, max_frame=5500):
    reader = cv2.VideoCapture(filename)
    metas = json.load(open(filename[:-4] + "_poses.json"))
    if not os.path.exists(filename[:-4]):
        os.mkdir(filename[:-4])

    i = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break

        if i >= start_frame:
            meta = metas[i]
            if meta is not None:
                boxes, poses = meta["boxes"], meta["poses"]
                for j in range(len(boxes)):
                    b, p = boxes[j], poses[j]
                    # print(i, j, boxes[j])
                    cv2.rectangle(frame, (b[0][0]-1, b[0][1]-1), (b[1][0]+1, b[1][1]+1), (0, 250, 0), 3)
                    draw_pose(np.array(p), frame)

            cv2.imwrite(filename[:-4] + "/%d.jpg" % i, frame)
            if i in [0, 5031, 5041, 5108, 5110, 5111, 5112, 5113, 5175]:
                path = filename[:filename.rfind("/")+1]
                s = int(path[-2])
                cv2.imwrite(path + "../%d_%d.jpg" % (s, i), frame)

        if i % 100 == 0:
            print("Extracted", i, "from", filename)

        i += 1
        if i > max_frame:
            break

    reader.release()


if __name__ == '__main__':
    pad_mosaic("mosaic_sample.jpg")
    # extract_frames_with_poses('../all_data/dumbo_2/sensor_1/right.mp4', start_frame=0, max_frame=100)
    extract_frames_with_poses('../all_data/dumbo_2/sensor_1/right.mp4', start_frame=5000, max_frame=5200)
    extract_frames_with_poses('../all_data/dumbo_2/sensor_4/right.mp4', start_frame=5000, max_frame=5200)
    exit(0)


    n = 3  # Maximum of n jobs to be scheduled at the same time (limited to 3 per GPU by the driver)
    use_gpu = 0
    if len(sys.argv) > 1:
        use_gpu = int(sys.argv[1])
        sys.argv = [sys.argv[0]]

    data_path = '../data/'
    sessions = glob.glob(data_path + "*/")  # glob.glob behaves differently outside of __main__ (i.e. inside functions)

    # Single threaded
    # render_all(sessions, use_gpu=use_gpu, skip_cameras=False, max_frames=200, prep_jobs_only=False, overwrite=True)
    # Parallel
    # jobs = render_all(sessions, use_gpu=use_gpu, skip_cameras=False, max_frames=None, prep_jobs_only=True)
    # joblib.Parallel(verbose=15, n_jobs=n, batch_size=n, pre_dispatch=n, backend="multiprocessing")(jobs)

    # jobs = subsample_all(sessions, num_gpus=2)

    # Single threaded
    # for session in browse_sessions(sessions):
    #     render_mosaic(session, max_frames=None, save_images=True, save_every=910, use_gpu=use_gpu, overwrite=True)
    # Parallel
    jobs = [joblib.delayed(render_mosaic)(session, max_frames=None, save_images=True, use_gpu=use_gpu) for session in browse_sessions(sessions)]

    joblib.Parallel(verbose=15, n_jobs=n, batch_size=n, pre_dispatch=n, backend="multiprocessing")(jobs)

    # for session in browse_sessions(sessions):
    #     renders = json.load(open(session + "meta/video_qualities.json", "r"))
    #     orders = json.load(open(session + "meta/camera_orders.json", "r"))
    #     quality = renders[0]
    #     assert quality["codec"] == "h265"
    #
    #     l, r, per = find_time_span(session)
    #     print(l, r, (r-l)/1200, per)
    #
    #     for sensor_i in range(4):
    #         video = session + "sensor_%d/" % (sensor_i + 1) + "video/"
    #         segments = json.load(open(video + "segments.json", "r"))
    #
    #         for cam_id in range(2):
    #             q = quality["qualities"]["sensor_%d" % (sensor_i+1)][cam_id]
    #             q = {"low": 0, "mid": 1, "high": 2}[q]
    #             bitrate = quality["bitrates"][q]
    #             filename = orders["sensor_%d" % (sensor_i+1)][cam_id]
    #             render_single(video, segments, (l, r, per), cam_id, filename, bitrate, use_gpu=use_gpu, max_frames=100)
    #
    #         # # for segment in segments:
    #         # #     reader = SegmentReader(video, segment["prefix"], tolerance=per, anonymize=True)
    #         # for cam_id in range(2):
    #         #     reader = SessionReader(video, segments, tolerance=1.1*per, anonymize=True, cam_id=cam_id)
    #         #     print(reader.get_frame(l)[0].shape)
    #         #     for i in range(200):
    #         #         frame, meta = reader.get_frame(l + (i) * per)
    #         #         # if meta is not None:
    #         #         #     for pose in meta["poses"]:
    #         #         #         draw_pose(np.array(pose), frame)
    #         #
    #         #         cv2.imshow("test", frame)
    #         #         cv2.waitKey(50)
