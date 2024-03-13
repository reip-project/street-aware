import utils
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


def load_data(data_path, data=None, calib_path=None, frame_ids=None, old=False):
    data = data or {s: {c: {} for c in [*cameras, "array"]} for s in sensors}
    print("frame_ids:", frame_ids, "\n")

    for s in sensors:
        for c in cameras:
            if old:
                filename = data_path + "%d_%s.png" % (s, c)
            else:
                filename = data_path + "frames/park_3_sensor_%d_%s.jpg" % (s, c)
            print(s, c, filename)
            data[s][c]["img"] = cv2.imread(filename)[:, :, ::-1].copy()
            if frame_ids is not None:
                data[s][c]["people"] = load_people(data_path + "%d_%s_people" % (s, c), frame_ids=frame_ids)

            if calib_path is not None:
                calib = json.load(open(calib_path + "%s_%d.json" % (c, s)))
                data[s][c]["K"] = np.array(calib["mtx"])

                pos = json.load(open(calib_path + "positions_%d.json" % s))
                data[s][c]["box_T"] = np.array(pos[c]["cam_T"]) / 1000.0
                data[s][c]["box_R"] = np.array(pos[c]["cam_R"])

                if c == cameras[0]:
                    data[s]["array"]["step"] = [pos["array"][a + "_stride"] / 1000.0 for a in ["x", "y"]]
                    data[s]["array"]["size"] = [pos["array"]["num_" + a] for a in ["x", "y"]]

    return data


def load_annotations(files, views=None, data=None, old=False, with_poses=False, all_joints=False):
    views = views or [(sensor, camera) for sensor in sensors for camera in cameras]
    print("views:", views)
    all_annot = []

    for file in files:
        raw = json.load(open(file))
        frame_id = raw["points"]["frame-number"] if with_poses else 0

        # Points
        if old:
            n = len(raw["points"]["sensors"][0]["aligned-points"])
        else:
            n = len(raw["sensors"][0]["aligned-points"])
        for i in range(n):
            visibility, annot = [False for _ in views], [None for _ in views]

            for j, view in enumerate(views):
                for view_data in raw["points"]["sensors"] if old else raw["sensors"]:
                    if int(view_data["sensor"][-1]) == view[0] and view_data["side"] == view[1]:
                        annot[j] = view_data["aligned-points"][i]
                        visibility[j] = len(annot[j]) > 0

            if False not in visibility:
                all_annot.append(annot)
        if with_poses:
            # Poses
            joints = [5, 7, 6, 8, 11, 13, 12, 14]  # hips, shoulders, knees, and elbows
            if all_joints:
                joints = list(range(len(SKELETON)))

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

    if frame_id is not None:
        people = data[s][c]["people"][frame_id]

        if people is not None:
            poses = people["poses"]

            if len(poses.shape) > 1:
                draw_pose_2d(poses[0, ...], img, small=True)

    plt.figure("Preview", (12, 9))
    plt.imshow(img)
    plt.plot(annot[:, view_id, 0], annot[:, view_id, 1], "r.")
    plt.tight_layout()


def reconstruct_pair(data, files, views, plot=False):
    assert len(views) == 2

    annot = load_annotations(files, views=views, data=data)

    ps1, ps2 = annot[:, 0, :], annot[:, 1, :]
    K1, K2 = data[views[0][0]][views[0][1]]["K"], data[views[1][0]][views[1][1]]["K"]
    img1, img2 = data[views[0][0]][views[0][1]]["img"], data[views[1][0]][views[1][1]]["img"]

    T, R, mask = find_relative(ps1, K1, ps2, K2, thr=3, img1=img1, img2=img2, plot=plot, every=5)
    p3d = triangulate_relative(ps1, K1, ps2, K2, T, R)

    if plot:
        ax = plot_3d("Reconstruction", plot_basis=True)
        basis(ax, T, R)
        scatter(ax, p3d[mask])
        scatter(ax, p3d, "b", s=2)
        axis_equal_3d(ax)

    return T, R


def triangulate_pair(data, annot, views, view_ids, T, R, plot=False):
    v0, v1 = view_ids
    ps1, ps2 = annot[:, v0, :], annot[:, v1, :]
    cam1, cam2 = data[views[v0][0]][views[v0][1]], data[views[v1][0]][views[v1][1]]

    p3d = triangulate_relative(ps1, cam1["K"], ps2, cam2["K"], T, R)

    if plot:
        ax = plot_3d("Triangulation", plot_basis=True)
        basis(ax, T, R)
        scatter(ax, p3d)
        axis_equal_3d(ax)

    return p3d, cam1, cam2, ax if plot else None


def explore_pair(data, files, side="across"):
    # if side == "chase":
    #     views = [(2, "left"), (1, "left")]
    #     # views = [(2, "right"), (1, "left")]  # tilt
    #
    # if side == "metro":
    #     views = [(3, "left"), (4, "right")]
    #     # views = [(3, "right"), (4, "right")]  # blur
    #
    # if side == "across":
    #     views = [(3, "left"), (1, "left")]
    #     # views = [(3, "right"), (1, "left")]  # worse
    #
    # if side == "across_2":
    #     views = [(4, "right"), (2, "right")]
    #     # views = [(4, "right"), (2, "left")]  # worse

    if side == "school":
        views = [(2, "right"), (3, "left")]

    if side == "bridge":
        views = [(1, "right"), (4, "right")]

    if side == "across":
        views = [(2, "right"), (4, "right")]

    if side == "across_2":
        views = [(3, "left"), (4, "right")]

    annot = load_annotations(files, views=views, data=data)
    print("annot:", annot.shape)

    preview(data, views, annot, view_id=0, frame_id=None)

    reconstruct_pair(data, files, views, plot=True)


def calibrate_pair(data, files, side, calib_file, scale_file, plot=False):
    # calib_views = [(2, "left"), (1, "left"), (3, "left"), (4, "right"), (2, "right")]
    calib_views = [(1, "right"), (2, "right"), (3, "left"), (4, "right")]
    calib_annot = load_annotations([calib_file], views=calib_views, data=data)
    print("calib_annot:", calib_annot.shape)

    # if side == "chase":
    #     view_ids = [0, 1]
    # if side == "metro":
    #     view_ids = [2, 3]
    # if side == "across":
    #     view_ids = [2, 1]
    # if side == "across_2":
    #     view_ids = [3, 4]

    if side == "school":
        view_ids = [1, 2]
    if side == "bridge":
        view_ids = [0, 3]
    if side == "across":
        view_ids = [1, 3]
    if side == "across_2":
        view_ids = [2, 3]

    views = [calib_views[i] for i in view_ids]

    if plot:
        preview(data, calib_views, calib_annot, view_id=view_ids[0], frame_id=None)

    # Cam2 in Cam1's frame of reference
    T, R = reconstruct_pair(data, files, views, plot=False)

    p3d, cam1, cam2, ax = triangulate_pair(data, calib_annot, calib_views, view_ids, T, R, plot=plot)

    # Recover the basis
    pca = PCA(n_components=3)
    p2 = pca.fit_transform(p3d)
    W_T, W_R = pca.mean_, pca.components_

    # Ensure Z is up
    if np.dot(np.array([0, 0, 1]), W_R[2, :]) > 0:
        W_R[2, :] = -W_R[2, :]

    # Y along the zebra lines
    y = np.average(p3d[1::2, :], axis=0) - np.average(p3d[::2, :], axis=0)
    y -= W_R[2, :] * np.dot(W_R[2, :], y)
    W_R[1, :] = y / np.linalg.norm(y)

    # Force right handedness
    W_R[0, :] = np.cross(W_R[1, :], W_R[2, :])

    # World in Cam1's frame of reference
    if plot:
        utils.plot(ax, p3d, ":")
        basis(ax, W_T, W_R)

    # Recover the scale
    ref_scales = np.array(json.load(open(scale_file)))
    ref_scales = np.repeat(ref_scales, 2) * 0.0254
    scales = np.linalg.norm(p3d[1::2, :] - p3d[::2, :], axis=1)
    factor = np.average(ref_scales / scales)
    print("Scale:", factor, "\n")

    # Apply scaling
    T, W_T = T * factor, W_T * factor

    # Place Cam1 in World's frame of reference
    T1, R1 = np.matmul(W_R, -W_T).ravel(), W_R.T

    # Place Cam2 in World's frame of reference
    T2 = np.matmul(W_R, -(W_T - T)).ravel()
    R2 = np.matmul(W_R, R.T).T

    calibs = [(*views[0], T1, R1, cam1["K"]), (*views[1], T2, R2, cam2["K"])]
    ref_calibs = calibs.copy()  # avoid duplicate array entries
    calibs.extend([second_eye(data, calib) for calib in ref_calibs])
    calibs.extend([mic_array(data, calib) for calib in ref_calibs])

    return calibs


def second_eye(data, calib):
    s, c, T, R, K = calib
    assert len(cameras) == 2
    c2 = cameras[0] if c == cameras[1] else cameras[1]

    cam1, cam2 = data[s][c], data[s][c2]

    # Place Cam2 in Cam1's frame of reference
    dT = np.matmul(cam1["box_R"], cam2["box_T"] - cam1["box_T"]).ravel()
    dR = np.matmul(cam1["box_R"], cam2["box_R"].T).T

    # Place Cam2 in World's frame of reference
    T2 = T + np.matmul(R.T, dT).ravel()
    R2 = np.matmul(R.T, dR.T).T

    return s, c2, T2, R2, cam2["K"]


def mic_array(data, calib):
    s, c, T, R, K = calib
    cT, cR = data[s][c]["box_T"], data[s][c]["box_R"]

    # Place Array in Cam1's frame of reference
    aT, aR = np.matmul(cR, -cT).ravel(), cR.T

    # Place Array in World's frame of reference
    T_A = T + np.matmul(R.T, aT).ravel()
    R_A = np.matmul(R.T, aR.T).T

    return s, "array", T_A, R_A, (*data[s]["array"]["size"], *data[s]["array"]["step"])


def calibrate_all(data, all_files, sides, calib_file, scale_file, plot=True):
    calibs = []

    for side in sides:
        calibs.extend(calibrate_pair(data, all_files, side, calib_file, scale_file))

    if plot:
        ax = plot_3d("Calibration", plot_basis=False)
        basis(ax, np.zeros(3), np.eye(3), length=3)

        for s, c, T, R, K in calibs:
            utils.plot(ax, np.array([[0, 0, 0], [T[0], T[1], 0], T]), "k--")
            basis(ax, T, R, length=1.5 if c == "array" else 1)

        utils.plot(ax, np.array([[5, 2, 0], [-5, 2, 0], [-5, -2, 0], [5, -2, 0], [5, 2, 0]]))
        axis_equal_3d(ax)

    return calibs


def position_errors(calibs):
    errors = []
    for i, calib in enumerate(calibs):
        s, c, T, R, K = calib
        for j, calib2 in enumerate(calibs):
            s2, c2, T2, R2, K2 = calib2
            if j > i and i != j and s == s2 and c == c2:
                errors.append((s, c, np.linalg.norm(T-T2)))

    if len(errors) > 0:
        print("Average position error:", np.average([err[2] for err in errors]), "meters")
        [print(" ", err) for err in errors]


def load_calibs(filename):
    calibs = {s: {c: {} for c in [*cameras, "array"]} for s in sensors}

    for calib in json.load(open(filename)):
        s, c = calib[:2]
        calibs[s][c]["T"] = np.array(calib[2])
        calibs[s][c]["R"] = np.array(calib[3])

        if c == "array":
            calibs[s][c]["dims"] = calib[4]
        else:
            calibs[s][c]["K"] = np.array(calib[4])

    return calibs


if __name__ == '__main__':
    data_path = './park_3/'
    calib_file, scale_file = data_path + "zebra_park_3_new.json", data_path + "dist_lines_park_3_new.json"

    all_files = [file for file in glob.glob(data_path + "*.json") if "poses" in file or "points" in file]
    # frame_ids = sorted(list({json.load(open(file))["frame-number"] for file in all_files}))
    # frame_ids.extend(list(range(11600, 11621)))  # cache event poses as well
    frame_ids = None

    data = load_data(data_path, calib_path='./calibration_data/', frame_ids=frame_ids)

    # sides, side_id = ["chase", "metro", "across", "across_2"], 2
    sides, side_id = ["school", "bridge", "across", "across_2"], 1

    # explore_pair(data, all_files, sides[side_id])
    # calibrate_pair(data, all_files, sides[side_id], calib_file, scale_file, plot=True)
    #
    # plt.show()
    # exit(0)

    utils.UP_AXIS = "Z"

    calibs = calibrate_all(data, all_files, sides, calib_file, scale_file)
    position_errors(calibs)

    calibs = calibrate_all(data, all_files, ["school", "bridge"], calib_file, scale_file)
    with open(data_path + "park_3_calibration.json", "w") as f:
        json.dump(calibs, f, indent=4, cls=NumpyEncoder)
        plt.savefig(data_path + "park_3_calibration.png", dpi=200)

    plt.show()
