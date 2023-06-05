import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r/zoom, ctr + r/zoom)


def plot_3d(R, T, figure_name="3d", title=None, size=(16, 9), axis_equal=True, save_as=None, **kw):
    plt.figure(figure_name, size)
    plt.clf()
    ax = plt.subplot(111, projection='3d', proj_type='ortho')
    ax.set_title(title if title else figure_name)

    scatter(ax, np.zeros(3), s=15, **kw)
    basis(ax, np.zeros(3), np.eye(3))
    # print(T, R)

    ax.set_xlabel("x, mm")
    ax.set_ylabel("z, mm")
    ax.set_zlabel("-y, mm")
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


path = "./"
files = ["zebra.json", "random.json"]

s1, s2 = ("sensor1", "left"), ("sensor3", "left")

ps1, ps2 = [], []

for file in files:
    all_annot = json.load(open(path + file))

    for point in all_annot:
        p1, p2 = None, None
        
        for pos in point:
            if pos["id"] + pos["side"] == s1[0] + s1[1] and pos["x"] is not None:
                p1 = (pos["x"], pos["y"])
            if pos["id"] + pos["side"] == s2[0] + s2[1] and pos["x"] is not None:
                p2 = (pos["x"], pos["y"])
        
        if (p1 is not None) and (p2 is not None):
            ps1.append(p1)
            ps2.append(p2)


filename_template = "./%s_%s.png"
img1 = cv2.imread(filename_template % (s1[0], s1[1]))[:, :, ::-1]
img2 = cv2.imread(filename_template % (s2[0], s2[1]))[:, :, ::-1]

ps1, ps2 = np.array(ps1, dtype=np.int32), np.array(ps2, dtype=np.int32)

K1 = np.array(json.load(open("./calibration_data/left_1.json"))["mtx"])
K2 = np.array(json.load(open("./calibration_data/left_3.json"))["mtx"])

print(K1)
print(K2)

F, mask = cv2.findFundamentalMat(ps1, ps2, cv2.FM_LMEDS)
print("\n", F, "\n")
# We select only inlier points
nps1 = ps1[mask.ravel()==1]
nps2 = ps2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(nps2.reshape(-1,1,2), 2, F)
img1l, _ = drawlines(img1.copy(),img2.copy(),lines1.reshape(-1,3),nps1,nps2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(nps1.reshape(-1,1,2), 1, F)
img2l, _ = drawlines(img2.copy(),img1.copy(),lines2.reshape(-1,3),nps2,nps1)

E = np.matmul(K2.T, np.matmul(F, K1))
R1, R2, T = cv2.decomposeEssentialMat(E)
print(R1.shape, T.shape)
print(R1, "\n", R2, "\n", T)

# plt.figure("mask", (24, 16))
# plt.imshow(mask)
# plt.tight_layout()

plt.figure("img1", (16, 9))
plt.imshow(img1l)
# plt.plot(ps1[:, 0], ps1[:, 1], "r.")
# plt.plot(nps1[:, 0], nps1[:, 1], "g.")
plt.tight_layout()

plt.figure("img2", (16, 9))
plt.imshow(img2l)
# plt.plot(ps2[:, 0], ps2[:, 1], "r.")
# plt.plot(nps2[:, 0], nps2[:, 1], "g.")
plt.tight_layout()



ax = plot_3d(None, None)

basis(ax, T.ravel(), R1.T)
basis(ax, T.ravel(), R2.T)
axis_equal_3d(ax)

plt.show()
