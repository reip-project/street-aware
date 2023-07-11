import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def undistort():
    imgs, calibs = [], []
    for sensor in range(1, 5):
        for side in ["left", "right"]:
            imgs.append(cv2.imread("sensor%d_%s.png" % (sensor, side)))
            calibs.append(np.array(json.load(open("./calibration_data/%s_%d.json" % (side, sensor)))["mtx"]))

    K = np.average(np.stack(calibs, axis=0), axis=0)
    with open("./undistorted/K.json", "w") as f:
        json.dump({"K": K.tolist()}, f, indent=4)
    print(K)

    for i, (img, calib) in enumerate(zip(imgs, calibs)):
        uimg = cv2.undistort(img, calib, None, K, None)
        cv2.imwrite("./undistorted/%d.png" % i, uimg)

# undistort()
# exit(0)


def triangulate(ps1, K1, ps2, K2, T, R):
    u_ref_xy = cv2.undistortPoints(ps1.astype(np.float32).reshape((-1, 1, 2)), K1, None).reshape((-1, 2))
    ref_rays = np.concatenate([u_ref_xy, np.ones((u_ref_xy.shape[0], 1))], axis=1)

    u_sec_xy = cv2.undistortPoints(ps2.astype(np.float32).reshape((-1, 1, 2)), K2, None).reshape((-1, 2))
    sec_rays = np.concatenate([u_sec_xy, np.ones((u_sec_xy.shape[0], 1))], axis=1)

    sec_rays = np.matmul(R.T, sec_rays.T).T

    v12 = np.sum(np.multiply(ref_rays, sec_rays), axis=1)
    v1, v2 = np.linalg.norm(ref_rays, axis=1)**2, np.linalg.norm(sec_rays, axis=1)**2
    L = (np.matmul(ref_rays, T) * v2 + np.matmul(sec_rays, -T) * v12) / (v1 * v2 - v12**2)

    return ref_rays * L[:, None]


def line(ax, p1, p2, *args, **kwargs):
    ax.plot(np.array([p1[0], p2[0]]), np.array([p1[2], p2[2]]), np.array([-p1[1], -p2[1]]), *args, **kwargs)

def basis(ax, T, R, *args, length=0.5, **kwargs):
    line(ax, T, T + length * R[:, 0], "r")
    line(ax, T, T + length * R[:, 1], "g")
    line(ax, T, T + length * R[:, 2], "b")

def scatter(ax, p, *args, **kwargs):
    if len(p.shape) > 1:
        ax.scatter(p[:, 0], p[:, 2], -p[:, 1], *args, **kwargs)
    else:
        ax.scatter(p[0], p[2], -p[1], **kwargs)

def axis_equal_3d(ax, zoom=1):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r/zoom, ctr + r/zoom)


def plot_3d(R, T, figure_name="3d", title=None, size=(9, 8), axis_equal=True, plot_basis=True, save_as=None, **kw):
    plt.figure(figure_name, size)
    plt.clf()
    ax = plt.subplot(111, projection='3d', proj_type='ortho')
    ax.set_title(title if title else figure_name)

    if plot_basis:
        scatter(ax, np.zeros(3), s=15, **kw)
        basis(ax, np.zeros(3), np.eye(3))
    # print(T, R)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")
    plt.tight_layout()
    # if axis_equal:
    #     axis_equal_3d(ax)

    if save_as is not None:
        plt.savefig(save_as + ".png", dpi=160)

    return ax


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, _ = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        # print(img1.shape, img1.dtype)
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1, img2


# path = "./"
# files = ["zebra.json", "random.json", "volume.json"]
# # files = ["random.json"]

path = "./chase_2/"
files = ["zebra.json", "signs.json", "metro_cafe.json", "garden_middle.json", "cars.json", "chipotle_side.json"]


# path = "D:/Dropbox/work/cvpr/annotation_files/"
# files = ["background1_2.json", "building.json", "metal_structure.json", "joao_3x.json", "yurii_3x.json"]
# files = ["background2_3.json", "building_people_2_3.json", "joao_3x.json", "yurii_3x.json"]

# s1, s2 = ("sensor1", "center"), ("sensor2", "center")
# s1, s2 = ("sensor2", "center"), ("sensor3", "center")

s1, s2 = ("sensor2", "left"), ("sensor1", "left")
# s1, s2 = ("sensor2", "right"), ("sensor1", "left")
# s1, s2 = ("sensor2", "left"), ("sensor2", "right")
# s1, s2 = ("sensor1", "left"), ("sensor3", "left")
# s1, s2 = ("sensor4", "right"), ("sensor3", "left")
# s1, s2 = ("sensor3", "left"), ("sensor3", "right")
# s1, s2 = ("sensor3", "right"), ("sensor3", "left")
# s1, s2 = ("sensor1", "left"), ("sensor1", "right")

ps1, ps2 = [], []

for file in files:
    all_annot = json.load(open(path + file))

    g1, g2 = None, None
    for group in all_annot:
        # if group["id"] == s1[0]:
        #     g1 = group["frozenAnnotations"]
        # if group["id"] == s2[0]:
        #     g2 = group["frozenAnnotations"]
        if group["id"] == s1[0] and group["side"] == s1[1]:
            g1 = group["frozenAnnotations"]
        if group["id"] == s2[0] and group["side"] == s2[1]:
            g2 = group["frozenAnnotations"]

    print(g1, "\n", g2)
    for p1, p2 in zip(g1, g2):
        if len(p1) == 2 and len(p2) == 2:
            ps1.append(p1)
            ps2.append(p2)
    # continue
    # for point in all_annot:
    #     p1, p2 = None, None
    #
    #     for pos in point:
    #         if pos["id"] + pos["side"] == s1[0] + s1[1] and pos["x"] is not None:
    #             p1 = (pos["x"], pos["y"])
    #         if pos["id"] + pos["side"] == s2[0] + s2[1] and pos["x"] is not None:
    #             p2 = (pos["x"], pos["y"])
    #
    #     if (p1 is not None) and (p2 is not None):
    #         ps1.append(p1)
    #         ps2.append(p2)

ps1, ps2 = np.array(ps1, dtype=np.float32), np.array(ps2, dtype=np.float32)
# ps1, ps2 = np.array(ps1, dtype=np.int32), np.array(ps2, dtype=np.int32)

# filename_template = "./%s_%s.png"
# img1 = cv2.imread(filename_template % (s1[0], s1[1]))[:, :, ::-1]
# img2 = cv2.imread(filename_template % (s2[0], s2[1]))[:, :, ::-1]
#
# filename_template = "./calibration_data/%s_%s.json"
# K1 = np.array(json.load(open(filename_template % (s1[1], s1[0][-1])))["mtx"])
# K2 = np.array(json.load(open(filename_template % (s2[1], s2[0][-1])))["mtx"])

filename_template = "./chase_2/%s_%s.png"
img1 = cv2.imread(filename_template % (s1[0], s1[1]))[:, :, ::-1]
img2 = cv2.imread(filename_template % (s2[0], s2[1]))[:, :, ::-1]

filename_template = "./calibration_data/%s_%s.json"
K1 = np.array(json.load(open(filename_template % (s1[1], s1[0][-1])))["mtx"])
K2 = np.array(json.load(open(filename_template % (s2[1], s2[0][-1])))["mtx"])

# filename_template = "D:/Dropbox/work/cvpr/%s_fscam_frames/undistorted/500.jpg"
# img1 = cv2.imread(filename_template % s1[0][-1])[:, :, ::-1]
# img2 = cv2.imread(filename_template % s2[0][-1])[:, :, ::-1]
#
# filename_template = "D:/Dropbox/work/cvpr/%s_calib_frames/calibrated/geometry.json"
# calib_1 = json.load(open(filename_template % s1[0][-1]))
# calib_2 = json.load(open(filename_template % s1[0][-1]))
# K1 = np.array(calib_1["new_mtx"])
# K2 = np.array(calib_2["new_mtx"])
#
# ps1 = cv2.undistortPoints(ps1, np.array(calib_1["mtx"]), np.array(calib_1["dist"]), P=K1).reshape((-1, 2))
# ps2 = cv2.undistortPoints(ps2, np.array(calib_2["mtx"]), np.array(calib_2["dist"]), P=K2).reshape((-1, 2))

print(K1)
print(K2)

F, mask = cv2.findFundamentalMat(ps1, ps2, cv2.FM_RANSAC, ransacReprojThreshold=10, confidence=0.999, maxIters=1000)
print("\n", F, "\n", mask)
# We select only inlier points
nps1 = ps1[mask.ravel()==1]
nps2 = ps2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(nps2.reshape(-1,1,2), 2, F)
img1l, img2l = drawlines(img1.copy(),img2.copy(),lines1.reshape(-1,3),nps1,nps2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(nps1.reshape(-1,1,2), 1, F)
img2r, img1r = drawlines(img2.copy(),img1.copy(),lines2.reshape(-1,3),nps2,nps1)

E = np.matmul(K2.T, np.matmul(F, K1))
R1, R2, T = cv2.decomposeEssentialMat(E)
print(R1.shape, T.shape)
print(R1, "\n", R2, "\n", T)

ups1 = cv2.undistortPoints(ps1, K1, None)
ups2 = cv2.undistortPoints(ps2, K2, None)
newMtx = np.eye(3, dtype=np.float32)

newE, newMask = cv2.findEssentialMat(ups1, ups2, newMtx)
# print(newMask)
ups1 = ups1[newMask.ravel()==1]
ups2 = ups2[newMask.ravel()==1]

# retval, R, t, mask = cv2.recoverPose(newE, ups1, ups2, newMtx)
retval, R, t, mask = cv2.recoverPose(E, ps1, ps2, (K1 + K2)/2)
print(retval, "of", ps1.shape[0], R, t)

print("\n", E)
print("\n", newE)

# plt.figure("mask", (24, 16))
# plt.imshow(mask)
# plt.tight_layout()

T = -np.matmul(R.T, t).ravel()
print("\n", T, "\n", R)

plot = False

if plot:
    plt.figure("img1", (9, 8))
    plt.imshow(img1l)
    plt.plot(ps1[:, 0], ps1[:, 1], "ro", markersize=5, mfc='none')
    # plt.plot(nps1[:, 0], nps1[:, 1], "g.")
    plt.tight_layout()

    plt.figure("img2", (9, 8))
    plt.imshow(img2r)
    plt.plot(ps2[:, 0], ps2[:, 1], "ro", markersize=5, mfc='none')
    # plt.plot(nps2[:, 0], nps2[:, 1], "g.")
    plt.tight_layout()

    ax = plot_3d(None, None)

    p3d = triangulate(ps1, K1, ps2, K2, T, R)
    scatter(ax, p3d)

    basis(ax, T, R.T)
    # basis(ax, t.ravel(), R.T)
    # basis(ax, T.ravel(), R1.T)
    # basis(ax, -T.ravel(), R1.T)
    # basis(ax, T.ravel(), R2.T)
    # basis(ax, -T.ravel(), R2.T)
    axis_equal_3d(ax)

    plt.show()
