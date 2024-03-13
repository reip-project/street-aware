from calibrate import *
utils.UP_AXIS = "Z"


if __name__ == '__main__':
    data_path = './chase_1/'
    event_frame = 11619

    calibs = load_calibs(data_path + "calibration.json")

    files = [file for file in glob.glob(data_path + "*.json") if "11200" in file]
    ground_files = [f for f in files if "polls" not in f and "back" not in f]
    views = [(2, "left"), (1, "left"), (3, "left"), (4, "right"), (2, "right")]

    data = load_data(data_path, frame_ids=[event_frame], old=True)
    world_annot = load_annotations(files, views=views, data=data, old=True)
    ground_annot = load_annotations(ground_files, views=views, data=data, old=True)
    event_filename = data_path + "event_%d_poses.json" % event_frame
    event_annot = load_annotations([event_filename], views=views, data=data, old=True, with_poses=True, all_joints=True)
    event_times = np.array(json.load(open(data_path + "event_11602_times.json")))
    event_times = event_times.T.reshape((-1, 3, 4))  # 4x3 microphones array

    # preview(data, views, world_annot, view_id=1, frame_id=event_frame)

    ax = plot_3d("Reconstruction", plot_basis=False)
    basis(ax, np.zeros(3), np.eye(3), length=2)

    for s in sensors:
        T, R = calibs[s]["array"]["T"], calibs[s]["array"]["R"]
        basis(ax, T, R, length=1)

        n, m, w, h = calibs[s]["array"]["dims"]
        x, y = np.meshgrid(np.linspace(-(n-1)*w/2, (n-1)*w/2, n), np.linspace(-(m-1)*h/2, (m-1)*h/2, m))
        p = np.stack([x, y, np.zeros_like(x)], axis=2).reshape((-1, 3))
        p3 = T + np.matmul(R.T, p.T).T
        scatter(ax, T + np.matmul(R.T, 3 * p.T).T, s=1)  # plot scaled up

        ts, c = event_times[s-1, :, :], 343.0 / 48000
        dtx, dty = -np.average(np.diff(ts, axis=1)), -np.average(np.diff(ts, axis=0))
        cx, cy = max(-1,  min(dtx*c/w, 1)), max(-1, min(dty*c/h, 1))
        sx, sy = np.sqrt(1 - cx*cx), np.sqrt(1 - cy*cy)
        dir = np.array([cx, sx * cy, sx * sy])
        dir = np.matmul(R.T, dir).ravel()
        line(ax, T, T + (7 if s == 1 else 13) * dir)

        data[s]["times"] = ts.ravel()
        data[s]["mics"] = p3
        data[s]["pos"] = T
        data[s]["dir"] = dir

    ps = [data[s]["pos"] for s in sensors]
    dirs = [data[s]["dir"] for s in sensors]

    O = intersect_lines(ps, dirs)
    utils.plot(ax, O, "g*", markersize=9, label="Beam forming")

    ps = np.concatenate([data[s]["mics"] for s in sensors], axis=0)
    ts = np.concatenate([data[s]["times"] for s in sensors])

    O2, ret = locate_audio(ps, ts, p_guess=None)
    O2, t = O2[:3], O2[3]
    utils.plot(ax, O2, "b*", markersize=9, label="Time delay")
    utils.plot(ax, (O+O2)/2, "m*", markersize=9, label="Average audio")

    print("Beam forming:", O)
    print("Time delay:", O2)
    print("Distance =", np.linalg.norm(O2-O), "meters")
    print("Average audio:", (O+O2)/2)

    O, n = np.array([0, 0, 0]), np.array([0, 0, 1])
    plane = triangulate_multiview_many(ground_annot, views, calibs, O=O, n=n)
    lines = triangulate_multiview_many(ground_annot, views, calibs)
    dists = np.linalg.norm(plane - lines, axis=1)
    scatter(ax, plane, c="m", s=4, label="Plane X")

    scatter(ax, triangulate_multiview_many(world_annot, views, calibs), c="r", s=4, label="Lines X")
    scatter(ax, triangulate_multiview_many(event_annot, views, calibs), c="b", s=6, label="Lines X")
    plot_3d_pose(ax, triangulate_multiview_many(event_annot, views, calibs))

    utils.plot(ax, np.array([[4.6, 2.3, 0], [-4.6, 2.3, 0], [-4.6, -2.3, 0],
                             [4.6, -2.3, 0], [4.6, 2.3, 0]]), "c--", label="Reference")
    axis_equal_3d(ax, zoom=1.2)
    plt.legend()
    ax.view_init(azim=-120, elev=30)
    plt.savefig(data_path + "reconstruction.png", dpi=200)

    plt.figure("Distances", (12, 9))
    plt.hist(dists, bins=50)
    plt.title("Distances between triangulated points (avg = %.2f m)" % np.average(dists))
    plt.xlabel("Distance, meters")
    plt.ylabel("Counts")
    plt.tight_layout()

    view_id = 2
    s, c = views[view_id]
    preview(data, views, world_annot, view_id=view_id, frame_id=event_frame)

    ps_lines = np.array([project_point(p, calibs[s][c]) for p in lines])
    ps_plane = np.array([project_point(p, calibs[s][c]) for p in plane])
    plt.plot(ps_lines[:, 0], ps_lines[:, 1], "bx", label="Reprojected lines X")
    plt.plot(ps_plane[:, 0], ps_plane[:, 1], "m+", label="Reprojected plane X")

    p = project_point(O, calibs[s][c])
    p2 = project_point(O2, calibs[s][c])
    pa = project_point((O+O2)/2, calibs[s][c])

    plt.plot(p[0], p[1], "g*", markersize=9, label="Beam forming audio")
    plt.plot(p2[0], p2[1], "b*", markersize=9, label="Time delay audio")
    plt.plot(pa[0], pa[1], "m*", markersize=9, label="Average audio")
    plt.legend()

    plt.show()
