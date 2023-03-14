import sys

from merge import *
from hrnet_pose import *
from centerface import CenterFace


def process_segment(path, segment, face_detector=None, pose_detector=None, max_frames=None):
    filename = path + segment['prefix']
    print("Processing", filename)
    vid_in = cv2.VideoCapture(filename + ".mkv")
    tot_frames = 0

    if face_detector is not None:
        face_vid_out = cv2.VideoWriter(filename + "_faces.avi", cv2.VideoWriter_fourcc(*'XVID'), 3, (2592, 1944))
        all_faces = []
    else:
        face_vid_out, all_faces = None, None

    if pose_detector is not None:
        pose_vid_out = cv2.VideoWriter(filename + "_poses.avi", cv2.VideoWriter_fourcc(*'XVID'), 3, (2592, 1944))
        all_poses = []
    else:
        pose_vid_out, all_poses = None, None

    while True:
        ret, image_bgr = vid_in.read()
        if ret:
            write = tot_frames % 5 == 0

            if face_detector is not None:
                # img2 = cv2.resize(image_bgr[:, :, ::-1], (image_bgr.shape[1]//2, image_bgr.shape[0]//2))
                # dets, lms = face_detector(img2, threshold=0.2)
                # faces = [[int(round(2*x)) for x in det[:4]] for det in dets]

                dets, lms = face_detector(image_bgr[:, :, ::-1], threshold=0.3)
                faces = [[int(round(x)) for x in det[:4]] for det in dets]

                scores = [det[4] for det in dets]
                all_faces.append({"faces": faces, "scores": scores})
                if write:
                    for face in faces:
                        cv2.rectangle(image_bgr, (face[0], face[1]), (face[2], face[3]), (2, 255, 0), 2)
                    face_vid_out.write(image_bgr)

            if pose_detector is not None:
                poses = pose_detector.detect(image_bgr, draw=write)
                all_poses.append(poses)
                if write:
                    pose_vid_out.write(image_bgr)

            tot_frames += 1
            if tot_frames % 15 == 0:
                print(tot_frames, "frames done out of", max_frames)
        else:
            break
        if max_frames is not None and tot_frames >= max_frames:
            break

    vid_in.release()

    if face_detector is not None:
        face_vid_out.release()
        with open(filename + "_faces.json", "w") as f:
            json.dump(all_faces, f, indent=None, cls=NumpyEncoder)

    if pose_detector is not None:
        pose_vid_out.release()
        with open(filename + "_poses.json", "w") as f:
            json.dump(all_poses, f, indent=None, cls=NumpyEncoder)


def main(sessions, use_gpu=0, skip_segments=True, faces_only=True):
    print("Using GPU", use_gpu)
    face_detector = CenterFace()

    if not faces_only:
        pose_detector = PoseDetector(gpu_id=0)  # torch/nn/parallel/data_paralle.py bugs out on line 153 when trying to use second GPU
    else:
        pose_detector = None

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

                process_segment(video, segment, face_detector=face_detector, pose_detector=pose_detector, max_frames=None)


if __name__ == '__main__':
    data_path = '../data/'
    sessions = glob.glob(data_path + "*/")  # glob.glob behaves differently outside of __main__ (i.e. inside functions)

    use_gpu = 0
    if len(sys.argv) > 1:
        use_gpu = int(sys.argv[1])
        sys.argv = [sys.argv[0]]

    main(sessions, use_gpu=use_gpu)
