from utils import *

sensors = [1, 2, 3, 4]
cameras = ["left", "right"]


def load_people(filename, frame_ids=None):
    if os.path.exists(filename + ".pkl"):
        return pickle.load(open(filename + ".pkl", "rb"))
    else:
        data = json.load(open(filename + ".json"))

        for i in range(len(data)):
            if data[i] is not None:
                if frame_ids is not None and i not in frame_ids:
                    data[i] = None
                else:
                    data[i]["boxes"] = np.array(data[i]["boxes"])
                    data[i]["poses"] = np.array(data[i]["poses"])

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(data, f)

        return data


def load_data(data_path, data=None, calib_path=None, frame_ids=None):
    data = data or {s: {c: {} for c in cameras} for s in sensors}
    print("frame_ids:", frame_ids)

    for s in sensors:
        for c in cameras:
            data[s][c]["img"] = cv2.imread(data_path + "%d_%s.png" % (s, c))[:, :, ::-1].copy()
            data[s][c]["people"] = load_people(data_path + "%d_%s_people" % (s, c), frame_ids=frame_ids)

            if calib_path is not None:
                calib = json.load(open(calib_path + "%s_%d.json" % (c, s)))
                data[s][c]["K"] = np.array(calib["mtx"])

                pos = json.load(open(calib_path + "positions_%d.json" % s))
                data[s][c]["box_T"] = np.array(pos[c]["cam_T"])
                data[s][c]["box_R"] = np.array(pos[c]["cam_R"])

    return data


def load_annotations(files, views=None, data=None):
    views = views or [(sensor, camera) for sensor in sensors for camera in cameras]
    print("views:", views)
    all_annot = []

    for file in files:
        raw = json.load(open(file))
        frame_id = raw["points"]["frame-number"]

        # Points
        for i in range(len(raw["points"]["sensors"][0]["aligned-points"])):
            visibility, annot = [False for _ in views], [None for _ in views]

            for j, view in enumerate(views):
                for view_data in raw["points"]["sensors"]:
                    if int(view_data["sensor"][-1]) == view[0] and view_data["side"] == view[1]:
                        annot[j] = view_data["aligned-points"][i]
                        visibility[j] = len(annot[j]) > 0

            if False not in visibility:
                all_annot.append(annot)

        # Poses
        # joints = list(range(len(SKELETON)))
        joints = [5, 7, 6, 8, 11, 13, 12, 14]  # hips, shoulders, knees, and elbows
        for i in range(len(raw["boxes"][0]["aligned-boxes"])):
            if data is None:
                break
            visibility, annots = [False for _ in views], [[None for _ in views] for _ in range(len(joints))]

            for j, view in enumerate(views):
                for view_data in raw["boxes"]:
                    if int(view_data["sensor"][-1]) == view[0] and view_data["side"] == view[1]:
                        pose_id = view_data["box-indices"][i]
                        if pose_id >= 0:
                            people = data[view[0]][view[1]]["people"]
                            poses = people[frame_id]["poses"]
                            pose = poses[pose_id, ...]
                            for k, joint in enumerate(joints):
                                annots[k][j] = pose[joint, :]
                            visibility[j] = True

            if False not in visibility:
                all_annot.extend(annots)

    return np.array(all_annot)


def preview(data, views, annot, view_id=0, frame_id=11200):
    s, c = views[view_id]
    img = data[s][c]["img"]
    people = data[s][c]["people"][frame_id]

    if people is not None:
        poses = people["poses"]

        if len(poses.shape) > 1:
            draw_pose_2d(poses[0, ...], img, small=True)

    plt.figure("Preview", (13, 10))
    plt.imshow(img)
    plt.plot(annot[:, view_id, 0], annot[:, view_id, 1], "r.")
    plt.tight_layout()


if __name__ == '__main__':
    data_path = './chase_1/'

    files = [file for file in glob.glob(data_path + "*.json") if "poses" in file or "points" in file]
    frame_ids = sorted(list({json.load(open(file))["points"]["frame-number"] for file in files}))

    data = load_data(data_path, calib_path='./calibration_data/', frame_ids=frame_ids)

    # Metro
    views = [(3, "left"), (4, "right")]
    # views = [(3, "right"), (4, "right")]  # blur

    # Chase
    # views = [(2, "left"), (1, "left")]
    # views = [(2, "right"), (1, "left")]  # tilt

    # Across
    # views = [(3, "right"), (1, "left")]  # worse
    # views = [(3, "left"), (1, "left")]
    # views = [(4, "right"), (2, "right")]
    # views = [(4, "right"), (2, "left")]  # worse

    annot = load_annotations(files, views=views, data=data)
    print("annot:", annot.shape)

    preview(data, views, annot, view_id=0, frame_id=11200)

    ps1, ps2 = annot[:, 0, :], annot[:, 1, :]
    K1, K2 = data[views[0][0]][views[0][1]]["K"], data[views[1][0]][views[1][1]]["K"]
    img1, img2 = data[views[0][0]][views[0][1]]["img"], data[views[1][0]][views[1][1]]["img"]

    T, R = find_relative(ps1, K1, ps2, K2, thr=1.5, img1=img1, img2=img2, plot=True, every=5)
    p3d = triangulate_relative(ps1, K1, ps2, K2, T, R)

    ax = plot_3d("Reconstruction", plot_basis=True)
    basis(ax, T, R)
    scatter(ax, p3d)
    axis_equal_3d(ax)

    plt.show()
