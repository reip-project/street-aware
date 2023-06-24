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


print("\nT:\n", T, "\nR:\n", R)

all_poses_1 = json.load(open(path + "%s_%s_people.json" % (s1[0], s1[1])))
all_poses_2 = json.load(open(path + "%s_%s_people.json" % (s2[0], s2[1])))

poses_1 = []
for dets in all_poses_1:
    best_pose = None
    if dets is None:
        poses_1.append(best_pose)
        continue

    areas = [(box[1][0] - box[0][0]) * (box[1][1] - box[0][1]) for box in dets["boxes"]]

    if len(areas) > 0:
        i = np.argmax(np.array(areas))
        best_pose = np.array(dets["poses"][i], dtype=np.float32)

    poses_1.append(best_pose)

poses_2 = []
for dets in all_poses_2:
    best_pose = None
    if dets is None:
        poses_1.append(best_pose)
        continue

    areas = [(box[1][0] - box[0][0]) * (box[1][1] - box[0][1]) for box in dets["boxes"]]

    if len(areas) > 0:
        i = np.argmax(np.array(areas))
        best_pose = np.array(dets["poses"][i], dtype=np.float32)

    poses_2.append(best_pose)


vid_1 = cv2.VideoCapture(path + "%s_%s.mp4" % (s1[0], s1[1]))
vid_2 = cv2.VideoCapture(path + "%s_%s.mp4" % (s2[0], s2[1]))

for i in range(1600):
    pose_1 = poses_1[i]
    pose_2 = poses_2[i]
    # _, img_1 = vid_1.read()
    # _, img_2 = vid_2.read()

    if i % 10:
        print(i)
    if i < 1500:
        continue

    if pose_1 is not None and pose_2 is not None:
        pose_3d = triangulate(pose_1, K1, pose_2, K2, T, R)

        # draw_pose(pose_1, img_1, small=False)
        # draw_pose(pose_2, img_2, small=False)

        ax = plot_3d(None, None, figure_name="3D pose of the biggest person", plot_basis=False)

        scatter(ax, pose_3d)
        draw_3d_pose(ax, pose_3d)
        ax.set_xlim([0.4, 0.7])
        ax.set_zlim([0.0, 0.3])
        ax.set_ylim([0.6, 0.9])
        # axis_equal_3d(ax)

        # plt.figure("Imgs", (18, 9))
        # plt.subplot(121)
        # plt.imshow(img_1[:, :, ::-1])
        #
        # plt.subplot(122)
        # plt.imshow(img_2[:, :, ::-1])
        # plt.tight_layout()

        plt.show()

