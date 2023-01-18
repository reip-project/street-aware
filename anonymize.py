import sys

from merge import *
from hrnet_pose import *


def process_segment(path, segment, detector, max_frames=None):
    filename = path + segment['prefix']
    print("Processing", filename)
    vid_in = cv2.VideoCapture(filename + ".mkv")
    vid_out = cv2.VideoWriter(filename + "_poses.avi", cv2.VideoWriter_fourcc(*'XVID'), 3, (2592, 1944))
    tot_frames = 0
    all_detections = []
    while True:
        ret, image_bgr = vid_in.read()
        if ret:
            write = tot_frames % 5 == 0
            detections = detector.detect(image_bgr, draw=write)
            all_detections.append(detections)
            if write:
                vid_out.write(image_bgr)
            tot_frames += 1
            if tot_frames % 15 == 0:
                print(tot_frames, "frames done out of", max_frames)
        else:
            break
        if max_frames is not None and tot_frames >= max_frames:
            break

    vid_in.release()
    vid_out.release()

    with open(filename + "_poses.json", "w") as f:
        json.dump(all_detections, f, indent=None, cls=NumpyEncoder)


def main(sessions, use_gpu=0, skip_segments=True):
    print("Using GPU", use_gpu)
    detector = PoseDetector(gpu_id=0)  # torch/nn/parallel/data_paralle.py bugs out on line 153 when trying to use second GPU

    for session in browse_sessions(sessions):
        for i in range(4):
            video = session + "sensor_%d/" % (i+1) + "video/"
            segments = json.load(open(video + "segments.json", "r"))

            print("\n%d segments in" % len(segments), video)
            for seg in segments:
                print("\t", len(seg["good"]), "out of", len(seg["all"]), "for", seg["pattern"], "are good. Lost:", seg["lost"])
            print()

            for segment in segments:
                print(segment)
                if int(segment['prefix'][0]) != use_gpu and skip_segments:
                    print("Segment %s is not for this gpu (use_gpu=%d). Skipping..." % (segment['prefix'], use_gpu))
                    continue

                process_segment(video, segment, detector, max_frames=None)


if __name__ == '__main__':
    data_path = '../data/'
    sessions = glob.glob(data_path + "*/")  # glob.glob behaves differently outside of __main__ (i.e. inside functions)

    use_gpu = 0
    if len(sys.argv) > 1:
        use_gpu = int(sys.argv[1])
        sys.argv = [sys.argv[0]]

    main(sessions, use_gpu=use_gpu)
