from camera_positions import *
from scipy import optimize
import joblib

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

def draw_pose(keypoints, img, small=False):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 3 if small else 7, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 3 if small else 7, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 1 if small else 3)

def draw_3d_pose(ax, keypoints):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 3)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        a = keypoints[kpt_a][:]
        b = keypoints[kpt_b][:]
        # cv2.circle(img, (int(x_a), int(y_a)), 5, CocoColors[i], -1)
        # cv2.circle(img, (int(x_b), int(y_b)), 5, CocoColors[i], -1)
        # cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
        line(ax, a, b)


print(T, "\n", R)
data_path = "D:/Dropbox/work/cvpr/"
times_1 = json.load(open(data_path + "2_fscam.json"))
times_2 = json.load(open(data_path + "3_fscam.json"))
times_1z = json.load(open(data_path + "2_flir.json"))
times_2z = json.load(open(data_path + "3_flir.json"))
gt1 = np.array([t["global_timestamp"] for t in times_1["all_metas"]])
gt2 = np.array([t["global_timestamp"] for t in times_2["all_metas"]])
gt1z = np.array([t["global_timestamp"] for t in times_1z["all_metas"]])
gt2z = np.array([t["global_timestamp"] for t in times_2z["all_metas"]])
all_poses_1 = json.load(open(data_path + "2_fscam_full_poses.json"))
all_poses_2 = json.load(open(data_path + "3_fscam_full_poses.json"))
all_poses_1z = json.load(open(data_path + "2_flir_full_poses.json"))
all_poses_2z = json.load(open(data_path + "3_flir_full_poses.json"))

K1z = np.array([[13000, 0, 960],
                [0, 13000, 600],
                [0, 0, 1]])

K2z = np.array([[13000, 0, 960],
                [0, 13000, 600],
                [0, 0, 1]])

target = np.array([0.4, 0, 1.18])
tn = target / np.linalg.norm(target)
y = np.array([0, 1, 0])
tx = np.cross(y, tn)
Tz = np.array([np.dot(tx, T), np.dot(y, T), np.dot(tn, T)])
tz = target - T
z = np.array([np.dot(tx, tz), np.dot(y, tz), np.dot(tn, tz)])
z /= np.linalg.norm(z)
x = np.cross(y, z)
Rz = np.array([x, y, z])
print("\nTz:\n", Tz)
print("\nRz:\n", Rz)

poses_1 = []
for dets in all_poses_1:
    best_pose = None
    sides = [box[1][0] for box in dets["boxes"]]

    if len(sides) > 0:
        i = np.argmax(np.array(sides))
        best_pose = np.array(dets["poses"][i], dtype=np.float32)

        best_pose = cv2.undistortPoints(best_pose, np.array(calib_1["mtx"]), np.array(calib_1["dist"]), P=K1).reshape((-1, 2))

    poses_1.append(best_pose)

poses_2 = []
for dets in all_poses_2:
    best_pose = None
    areas = [(box[1][0] - box[0][0]) * (box[1][1] - box[0][1]) for box in dets["boxes"]]

    if len(areas) > 0:
        i = np.argmax(np.array(areas))
        best_pose = np.array(dets["poses"][i], dtype=np.float32)

        best_pose = cv2.undistortPoints(best_pose, np.array(calib_2["mtx"]), np.array(calib_2["dist"]), P=K2).reshape((-1, 2))

    poses_2.append(best_pose)


poses_1z = []
for dets in all_poses_1z:
    best_pose = None
    areas = [(box[1][0] - box[0][0]) * (box[1][1] - box[0][1]) for box in dets["boxes"]]

    if len(areas) > 0:
        i = np.argmax(np.array(areas))
        best_pose = np.array(dets["poses"][i], dtype=np.float32)

    poses_1z.append(best_pose)


poses_2z = []
for dets in all_poses_2z:
    best_pose = None
    areas = [(box[1][0] - box[0][0]) * (box[1][1] - box[0][1]) for box in dets["boxes"]]

    if len(areas) > 0:
        i = np.argmax(np.array(areas))
        best_pose = np.array(dets["poses"][i], dtype=np.float32)

    poses_2z.append(best_pose)

# 15400, 15700
for i in range(1140, 1182):
# for i in range(7545, 7580):
    t = gt1[i]
    j = np.argmin(np.abs(t-gt2))
    pose_1 = poses_1[i]
    pose_2 = poses_2[j]

    iz = np.argmin(np.abs(t-gt1z)) - 38
    # jz = np.argmin(np.abs(t-gt2z))
    jz = iz
    pose_1z = poses_1z[iz]
    pose_2z = poses_2z[jz]

    print(i, j, iz, jz)

    if pose_1 is not None and pose_2 is not None and pose_1z is not None and pose_2z is not None:
        pose_3d = triangulate(pose_1, K1, pose_2, K2, T, R)
        pose_3dz = triangulate(pose_1z, K1z, pose_2z, K2z, Tz, Rz)

        img = cv2.imread(data_path + "2_fscam_frames/undistorted/%d.jpg" % i)
        draw_pose(pose_1, img, small=True)
        # img = cv2.imread(data_path + "2_flir_frames/%d.jpg" % iz)
        # draw_pose(pose_1z, img, small=False)
        # plt.figure("img")
        # plt.clf()
        # plt.imshow(img[800:1200, 2900:3400, ::-1])
        # plt.imshow(img[:, 300:1800, ::-1])
        cv2.imwrite(data_path + "pitch_1_wide_frames/%d.png" % i, img[800:1200, 2900:3400, :])
        # cv2.imwrite(data_path + "pitch_1_zoom_frames/%d.png" % i, img[:, 300:1800, :])

        ax = plot_3d(None, None, figure_name="3D Pose from Wide View Camera", plot_basis=False)
        # ax = plot_3d(None, None, figure_name="3D Pose from Zoomed View Camera", plot_basis=False)
        # basis(ax, T, R.T)
        # basis(ax, Tz, Rz.T)

        scatter(ax, pose_3d)
        draw_3d_pose(ax, pose_3d)
        ax.set_xlim([0.37, 0.43])
        ax.set_zlim([0.0, 0.04])
        ax.set_ylim([1.14, 1.22])
        ax.view_init(elev=30, azim=-50)
        plt.savefig(data_path + "pitch_1_wide_poses/%d.png" % i, dpi=200)

        # scatter(ax, pose_3dz)
        # draw_3d_pose(ax, pose_3dz)
        # ax.set_xlim([-0.02, 0.04])
        # ax.set_zlim([-0.03, 0.03])
        # ax.set_ylim([1.18, 1.25])
        # ax.view_init(elev=30, azim=-50)
        # plt.savefig(data_path + "pitch_1_zoom_poses/%d.png" % iz, dpi=200)

        # axis_equal_3d(ax)

        # plt.show()


def mls(pos, fps=150, n=15, order=4):
    assert n % 2 == 1
    l = pos.shape[0]
    # print(l)
    pos_mls = []
    vel_mls = []

    def err_func(model, t, ps):
        model = model.reshape((-1, 2))
        dist = np.zeros_like(ps)

        for dim in range(2):
            p_model = np.polyval(model[:, dim], t)
            dist[:, dim] = ps[:, dim] - p_model

        return dist.ravel()

    for i in range(n//2, l-n//2):
        p = pos[i - n//2:i + n//2 + 1]
        t = np.arange(-n//2 + 1, n//2 + 1) / fps
        model_0 = np.zeros((order + 1, 2))
        model_0[-1, :] = pos[i, :]

        # print(i)
        # print(model_0)
        # print(err_func(model_0, t, p).reshape((-1, 2)))
        # print()

        res = optimize.least_squares(err_func, model_0.ravel(), args=(t, p))
        model = res['x'].reshape((-1, 2))
        pos_mls.append(model[-1, :])
        vel_mls.append(model[-2, :])

        # print(model)
        # print(err_func(model, t, p).reshape((-1, 2)))
        # print()

    return np.array(pos_mls) #, np.array(vel_mls)


# n = 15
# many_poses_1z = np.array(poses_1z[15300-(n//2):15751 + (n//2)])
# many_poses_2z = np.array(poses_2z[15300-(n//2):15751 + (n//2)])
# mls_poses_1z = np.array(poses_1z[15300:15751])
# mls_poses_2z = np.array(poses_2z[15300:15751])
#
# jobs_1 = []
# jobs_2 = []
# for joint in range(many_poses_1z.shape[1]):
#     jobs_1.append(joblib.delayed(mls)(many_poses_1z[:, joint, :]))
#     jobs_2.append(joblib.delayed(mls)(many_poses_2z[:, joint, :]))
#
# results_1 = joblib.Parallel(verbose=15, n_jobs=-1, pre_dispatch="all")(jobs_1)
# results_2 = joblib.Parallel(verbose=15, n_jobs=-1, pre_dispatch="all")(jobs_2)
#
# for joint in range(many_poses_1z.shape[1]):
#     mls_poses_1z[:, joint, :] = results_1[joint]
#     mls_poses_2z[:, joint, :] = results_2[joint]
#
# for iz in range(15300, 15751):
#     # pose_1z = poses_1z[iz]
#     # pose_2z = poses_2z[iz]
#     pose_1z = mls_poses_1z[iz-15300]
#     pose_2z = mls_poses_2z[iz-15300]
#
#     print(iz)
#
#     if pose_1z is not None and pose_2z is not None:
#         pose_3dz = triangulate(pose_1z, K1z, pose_2z, K2z, Tz, Rz)
#
#         img = cv2.imread(data_path + "2_flir_frames/%d.jpg" % iz)
#         draw_pose(pose_1z, img, small=False)
#         cv2.imwrite(data_path + "pitch_1_zoom_frames/%d.png" % iz, img[:, 300:1800, :])
#
#         ax = plot_3d(None, None, figure_name="3D Pose from Zoomed View Camera", plot_basis=False)
#
#         scatter(ax, pose_3dz)
#         draw_3d_pose(ax, pose_3dz)
#         ax.set_xlim([-0.02, 0.04])
#         ax.set_zlim([-0.03, 0.03])
#         ax.set_ylim([1.18, 1.25])
#         ax.view_init(elev=30, azim=-50)
#         plt.savefig(data_path + "pitch_1_zoom_poses/%d.png" % iz, dpi=200)
