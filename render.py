import numpy as np

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
        assert self.num_frames == len(self.meta), "Metadata mismatch"
        self.num_frames = min(self.num_frames, len(self.timestamps))
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
            blurred = cv2.GaussianBlur(frame, (15, 15), cv2.BORDER_DEFAULT)
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


def render_single(path, segments, timing, cam_id, filename, bitrate, use_gpu=0, max_frames=None):
    reader = SessionReader(path, segments, tolerance=1.1*timing[2], anonymize=True, cam_id=cam_id)
    writer = GstVideo(path + "../" + filename + ".mp4", 2592, 1944, 1200 / timing[2], format="BGR",
                      bitrate=bitrate, variable=True, codec='h264', gpu=use_gpu)

    all_metas = []
    for i, t in enumerate(np.arange(timing[0], timing[1] + timing[2], timing[2])):
        if max_frames and i >= max_frames:
            break
        if i > 0 and i % 100 == 0:
            print(i, "frames rendered of", max_frames, "in", path + "../" + filename + ".mp4")
        frame, meta = reader.get_frame(t)
        writer.write(frame)
        all_metas.append(meta)

    with open(path + "../" + filename + "_meta.json", "w") as f:
        json.dump(all_metas, f, indent=None, cls=NumpyEncoder)

    with open(path + "../" + filename + ".json", "w") as f:
        json.dump({"start": timing[0], "stop": timing[1],
                   "period": timing[2], "frequency, Hz": 1200,
                   "max_frames": max_frames}, f, indent=4, cls=NumpyEncoder)
    writer.close()
    reader.release()


def render_all(sessions, use_gpu=0, skip_cameras=True, max_frames=None, prep_jobs_only=False):
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
                    jobs.append(joblib.delayed(render_single)(video, segments, (l, r, per), cam_id, filename, bitrate, use_gpu=use_gpu, max_frames=max_frames))
                else:
                    render_single(video, segments, (l, r, per), cam_id, filename, bitrate, use_gpu=use_gpu, max_frames=max_frames)
    return jobs


if __name__ == '__main__':
    use_gpu = 0
    if len(sys.argv) > 1:
        use_gpu = int(sys.argv[1])
        sys.argv = [sys.argv[0]]

    data_path = '../data/'
    sessions = glob.glob(data_path + "*/")  # glob.glob behaves differently outside of __main__ (i.e. inside functions)

    # render_all(sessions, use_gpu=use_gpu, skip_cameras=False, max_frames=200, prep_jobs_only=False)
    jobs = render_all(sessions, use_gpu=use_gpu, skip_cameras=True, max_frames=None, prep_jobs_only=True)
    n = 3  # Maximum of n jobs to be scheduled at the same time (limited to 3 per GPU by the driver)
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
