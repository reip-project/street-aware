import os
import cv2
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import optimize

SKELETON = [[1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17
UP_AXIS = "Y"


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def draw_pose_2d(keypoints, img, small=False):
    assert keypoints.shape == (NUM_KPTS, 2)

    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]

        cv2.circle(img, (int(x_a), int(y_a)), 5 if small else 7, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 5 if small else 7, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2 if small else 3)

    return img


# img1 - image on which we draw the epilines for the points in img2 lines - corresponding epilines
def draw_epipolar(img1, img2, lines, pts1, pts2, colors=None, every=1):
    r, c, _ = img1.shape
    if colors is None:
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(lines))]

    for i, (r, color, pt1, pt2) in enumerate(zip(lines, colors, pts1.astype(np.int32), pts2.astype(np.int32))):
        if not i % every == 0:
            continue
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


def plot(ax, p, *args, **kwargs):
    if UP_AXIS == "Y":
        if len(p.shape) > 1:
            ax.plot(p[:, 0], p[:, 2], -p[:, 1], *args, **kwargs)
        else:
            ax.plot(p[0], p[2], -p[1], *args, **kwargs)
    else:
        if len(p.shape) > 1:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], *args, **kwargs)
        else:
            ax.plot(p[0], p[1], p[2], *args, **kwargs)


def line(ax, p1, p2, *args, **kwargs):
    if UP_AXIS == "Y":
        ax.plot(np.array([p1[0], p2[0]]), np.array([p1[2], p2[2]]), np.array([-p1[1], -p2[1]]), *args, **kwargs)
    else:
        ax.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), np.array([p1[2], p2[2]]), *args, **kwargs)


def basis(ax, T, R, *args, length=0.25, **kwargs):
    line(ax, T, T + length * R[0, :], "r", **kwargs)
    line(ax, T, T + length * R[1, :], "g", **kwargs)
    line(ax, T, T + length * R[2, :], "b", **kwargs)


def scatter(ax, p, *args, **kwargs):
    if UP_AXIS == "Y":
        if len(p.shape) > 1:
            ax.scatter(p[:, 0], p[:, 2], -p[:, 1], *args, **kwargs)
        else:
            ax.scatter(p[0], p[2], -p[1], **kwargs)
    else:
        if len(p.shape) > 1:
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], *args, **kwargs)
        else:
            ax.scatter(p[0], p[1], p[2], **kwargs)


def plot_3d_pose(ax, keypoints):
    assert keypoints.shape == (NUM_KPTS, 3)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        line(ax, keypoints[kpt_a][:], keypoints[kpt_b][:])


def plot_3d(figure_name="3d", size=(10, 9), title=None, plot_basis=False):
    plt.figure(figure_name, size)
    plt.clf()
    ax = plt.subplot(111, projection='3d', proj_type='ortho')
    ax.set_title(title if title else figure_name)

    if plot_basis:
        scatter(ax, np.zeros(3), s=16)
        basis(ax, np.zeros(3), np.eye(3))

    if UP_AXIS == "Y":
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("-y")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()

    return ax


def axis_equal_3d(ax, zoom=1):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r/zoom, ctr + r/zoom)
    plt.tight_layout()


def img_to_ray(ps, K):
    xy = cv2.undistortPoints(ps.astype(np.float32).reshape((-1, 1, 2)), K, None).reshape((-1, 2))
    return np.concatenate([xy, np.ones((xy.shape[0], 1))], axis=1)


def triangulate_relative(ps1, K1, ps2, K2, T, R):
    ref_rays, sec_rays = img_to_ray(ps1, K1), img_to_ray(ps2, K2)
    sec_rays = np.matmul(R.T, sec_rays.T).T

    v12 = np.sum(np.multiply(ref_rays, sec_rays), axis=1)
    v1, v2 = np.linalg.norm(ref_rays, axis=1)**2, np.linalg.norm(sec_rays, axis=1)**2
    L = (np.matmul(ref_rays, T) * v2 + np.matmul(sec_rays, -T) * v12) / (v1 * v2 - v12**2)

    return ref_rays * L[:, None]


def find_relative(ps1, K1, ps2, K2, thr=5, img1=None, img2=None, plot=False, every=2):
    F, mask = cv2.findFundamentalMat(ps1, ps2, cv2.FM_RANSAC, thr, confidence=0.999, maxIters=1000)
    mask = mask.ravel() == 1
    nps1, nps2 = ps1[mask], ps2[mask]
    print(nps1.shape[0], "of", ps1.shape[0])

    E = np.matmul(K2.T, np.matmul(F, K1))
    retval, R, t, mask2 = cv2.recoverPose(E, nps1, nps2, (K1 + K2) / 2)
    T = np.matmul(R.T, -t).ravel()
    # print("T:\n", T, "\nR:\n", R)

    if img1 is not None and img2 is not None:
        lines1 = cv2.computeCorrespondEpilines(nps2.reshape(-1, 1, 2), 2, F)
        lines2 = cv2.computeCorrespondEpilines(nps1.reshape(-1, 1, 2), 1, F)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for i in range(nps1.shape[0])]
        draw_epipolar(img1, img2, lines1.reshape(-1, 3), nps1, nps2, colors=colors, every=every)
        draw_epipolar(img2, img1, lines2.reshape(-1, 3), nps2, nps1, colors=colors, every=every)

        if plot:
            plt.figure("img1", (10, 8))
            plt.imshow(img1)
            plt.tight_layout()

            plt.figure("img2", (10, 8))
            plt.imshow(img2)
            plt.tight_layout()

    return T, R, mask


# ps = [np.array([x, y, x]), ...] in crosswalk's coordinate system
# dirs = [np.array([nx, ny, nx]), ...] in crosswalk's coordinate system (norm(dirs[i]) == 1)
def intersect_lines(ps, dirs):
    As = [np.outer(d, d) - np.eye(3) for d in dirs]
    Bs = [np.matmul(A, p).ravel() for A, p in zip(As, ps)]

    A = np.sum(np.stack(As, axis=2), axis=2)
    B = np.sum(np.stack(Bs, axis=1), axis=1)

    return np.linalg.inv(A) @ B


# points[view_id, :] = np.array([x, y]) in corresponding camera/view's image
# views = [(1, "left"), (2, "right), ...] counting sensors starting from 1
# calibs[sensor][camera] = {"K": np.array, ...} obtained using calibrate.load_calibs(filename)
def triangulate_multiview(points, views, calibs):
    ps, dirs = [], []

    for i, view in enumerate(views):
        calib = calibs[view[0]][view[1]]
        r = img_to_ray(points[i, :], calib["K"]).ravel()
        r /= np.linalg.norm(r)
        r = np.matmul(calib["R"].T, r).ravel()
        ps.append(calib["T"])
        dirs.append(r)

    return intersect_lines(ps, dirs)


# annot[point_id, view_id, :] = np.array([x, y]) in corresponding camera/view's image
# views = [(1, "left"), (2, "right), ...] counting sensors starting from 1
# calibs[sensor][camera] = {"K": np.array, ...} obtained using calibrate.load_calibs(filename)
def triangulate_multiview_many(annot, views, calibs):
    return np.array([triangulate_multiview(annot[i, :, :], views, calibs) for i in range(annot.shape[0])])


# p = np.array([x, y, x]) in crosswalk's coordinate system
def project_point(p, calib):
    r = p - calib["T"]
    r = np.matmul(calib["R"], r)
    r /= r[2]
    return np.matmul(calib["K"], r)[:2]


def locate_audio(ps, ts, p_guess=None):
    def loss(x):
        p, t = x[:3], x[3]
        dl = np.linalg.norm(ps - p[None, :], axis=1)
        dt = np.abs(ts - t) / 48000
        d = dl - dt * 343
        return np.sum(d*d)

    tm = np.min(ts)
    t0 = tm - 500
    p0 = np.average(ps, axis=0) if p_guess is None else p_guess
    print("\n", p0, tm)

    bbox = [(-10, 10), (-10, 10), (-5, 5)]
    res = optimize.minimize(loss, np.concatenate([p0, [t0]]), bounds=[*bbox, (tm - 10_000, tm)])
    print(res["success"], res["x"], res["x"][3] - tm, res["x"][2] - p0[2])

    return res["x"], res["success"]

