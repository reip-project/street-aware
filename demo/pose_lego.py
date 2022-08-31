# Mosaic.py : For creating mosaic video / each session
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import sys
import cvlib as cv
import glob
import argparse
import os
import joblib

import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
import json
# print(cv2.)

from gst_video import GstVideo

from natsort import natsorted
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

#  Read frame, anonymize faces, detect poses and return it.
def detection(video_path, outdir, frame_rate, w, h, skip=0):
    number_persons, all_boxes, all_poses, all_both = [], [], [], []
    print(video_path)
    gstream = "filesrc location=%s ! image/jpeg,width=%d,height=%d,framerate=30/1 ! jpegdec ! " \
                      "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
                      "appsink" % (video_path, w, h)
    try:
        # reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
        reader = cv2.VideoCapture(video_path)
        # writer = cv2.VideoWriter(outdir, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),frame_rate,(w ,h))     # Writer
    except Exception as e:
        print(e)
    print("Trying to read!")

    ret, frame = reader.read()
    # reader.set(cv2.CAP_PROP_POS_FRAMES, 3600)
    # print(ret,frame)
    c = 0
    while ret:
        c += 1
        if c <= skip:
            print("skipping", c)
            ret, frame = reader.read()
            continue
        # print("Reading!")
        #  Anonymization of faces
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # faces, confidences = cv.detect_face(frame)
        # for face in faces:
        #     (startX, startY) = face[0], face[1]
        #     (endX, endY) = face[2], face[3]
        #     roi = frame[startY:startY + endY, startX:startX + endX]
        #     # applying a gaussian blur over this new rectangle area
        #     roi = cv2.GaussianBlur(roi, (23, 23), 30)
        #     # impose this blurred image on original image to get final image
        #     frame[startY:startY + roi.shape[0], startX:startX + roi.shape[1]] = roi

        # Pose detection in image
        # estimate on the image
        # image = frame[:, :, [2, 1, 0]]

        input = []
        # img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(frame / 255.).permute(2, 0, 1).float().to(CTX)
        input.append(img_tensor)

        # object detection box
        pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
        all_boxes.append(pred_boxes)
        # number_persons.append({'count':len(pred_boxes),'frame_id': c})
        number_persons.append(len(pred_boxes))
        print(len(number_persons), "after", c)
        # pose estimation
        frame_poses = []
        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = frame.copy()
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                if len(pose_preds) >= 1:
                    for kpt in pose_preds:
                        frame_poses.append(kpt)
                        # draw_pose(kpt, frame)  # draw the poses
        all_poses.append(frame_poses)
        all_both.append({"boxes": pred_boxes,
                         "poses": frame_poses})
        # writer.write(frame[:, :, ::-1])
        # if c == 100:
        #     break
        ret, frame = reader.read()

    reader.release()
    # writer.release()
    # data = json.dump(number_persons)
    with open(outdir[:-4] + '.json', 'w') as f:
        json.dump({
            "boxes": all_boxes,
            "poses": all_poses,
            "both": all_both
        }, f, indent=4, cls=NumpyEncoder)

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def draw_bbox(box, img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train keypoints network')
#     # general
#     parser.add_argument('--cfg', type=str, default='inference-config.yaml')
#     parser.add_argument('--showFPS', action='store_true')
#     parser.add_argument('--root',action='store_true')
#     parser.add_argument('--sessionId',action='store_true')
#
#     parser.add_argument('opts',
#                         help='Modify config options using the command-line',
#                         default=None,
#                         nargs=argparse.REMAINDER)
#
#     args = parser.parse_args()
#
#     # args expected by supporting codebase
#     args.modelDir = ''
#     args.logDir = ''
#     args.dataDir = ''
#     args.prevModelDir = ''
#     return args


def plot(session, sensor, camera):
    filename = "occupancy_ss%d_s%d_%s" % (session, sensor, camera)
    data = np.array(json.load(open(filename + ".json")))
    print(data, data.shape)

    fps, n = 15, 10
    m = fps * n
    N = data.shape[0] // m
    binned = np.sum(data[:N * m].reshape(-1, m), axis=1) / m

    plt.figure("Average occupancy", (16, 9))
    plt.plot(np.arange(N) * n / 60, binned)
    plt.xlabel("Session %d time, minutes" % session)
    plt.ylabel("Average occupancy, people")
    plt.title("Chase bank crossing (Sensor %d, %s)" % (sensor, camera))
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=200)
    plt.show()

    exit(0)


def redraw(poses, in_path, out_path, fps, w, h, bitrate=50000, skip=0, stop_after=None):
    vid_in = cv2.VideoCapture(in_path)
    print(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_out = GstVideo(out_path, fps, w, h, bitrate)

    ret, frame = vid_in.read()
    # reader.set(cv2.CAP_PROP_POS_FRAMES, 3600)
    # print(ret,frame)
    c = 0
    while ret:
        c += 1
        if c <= skip:
            print("skipping", c)
            ret, frame = vid_in.read()
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for pose in poses[c-1]:
            draw_pose(np.array(pose), frame)

        vid_out.write(frame)
        if c % 10 == 0:
            print(c)

        if stop_after is not None and c - skip >= stop_after:
            break
        ret, frame = vid_in.read()

    vid_out.close()

def redraw_batch(poses, in_path, out_path, fps, w, h, bitrate=50000, skip=0, stop_after=None, batch_size=16):
    vid_in = cv2.VideoCapture(in_path)
    print(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_out = GstVideo(out_path, fps, w, h, bitrate=bitrate, format="RGB")

    for i in range(skip):
        ret, frame = vid_in.read()

    def job(frame, poses):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for pose in poses:
            draw_pose(np.array(pose), frame)
        return frame

    for i in range(stop_after // batch_size):
        print("batch", i)
        frames = [vid_in.read()[1] for j in range(batch_size)]
        jobs = [joblib.delayed(job)(frames[j], poses[skip + i*16 + j]) for j in range(batch_size)]
        results = joblib.Parallel(n_jobs=batch_size)(jobs)

        for j in range(batch_size):
            vid_out.write(results[j])

    vid_out.close()


if __name__ == "__main__":
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)

    # plot(session=3, sensor=2, camera="left")

    this_dir = osp.dirname(__file__)

    lib_path = osp.join(this_dir, '..', 'lib')
    add_path(lib_path)

    # mm_path = osp.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
    # add_path(mm_path)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # args = parse_args()
    update_config(cfg, None)
    # cfg = 'inference-config.yaml'
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()
    # root = "/mnt/tmp/nec/Aug_2/"
    # sessionID = str(3)
    # # sensor = input("Sensor")
    # sensor = str(3)
    # dir = root+"session_"+sessionID+"/"
    # showFPS = 15

    # cam_path = dir + 'sensor_' + sensor + "/video/1_1659464601_%d.avi"
    # # for cam_path in glob.glob(dir+"*avi"):
    #     # print(cam_path)
    # outdir = dir + os.path.split(cam_path)[-1][1]


    w, h, showFPS, bitrate = None, None, None, None
    dir = "/mnt/cache/legotracker/Aug_24/"

    for unit in [1, 2, 3]:
        for run in [1, 2, 3, 4, 5]:
            for cam in ["fscam", "flir"]:
                if cam == "fscam":
                    w, h, showFPS, bitrate = 3840, 2160, 15, 100000
                if cam == "flir":
                    w, h, showFPS, bitrate = 1920, 1200, 30, 30000

                cam_path = dir + "unit_%d/run_%d/%s.avi" % (unit, run, cam)
                outdir = dir + "unit_%d/%s_%d.avi" % (unit, cam, run)
                print("\n", cam_path, outdir, w, h, showFPS, bitrate)

                # detection(cam_path, outdir, showFPS, w, h, skip=0)

                with open(outdir[:-4] + ".json") as f:
                    poses = json.load(f)["poses"]

                vid = cv2.VideoCapture(cam_path)
                print(vid.get(cv2.CAP_PROP_FRAME_COUNT), len(poses), poses[1000])

                # redraw(poses, cam_path, outdir, showFPS, w, h, stop_after=3000)
                redraw_batch(poses, cam_path, outdir, showFPS, w, h, bitrate=bitrate,
                             stop_after=len(poses), batch_size=16)

    # # dir = "/mnt/cache/lego_2/"
    # dir = "/home/vidaviz/lego/lego1/mnt/ssd/"
    # w, h = 1920, 1200
    # showFPS = 30
    # # w, h = 3840, 2160
    # # showFPS = 15
    #
    # for box in [2]:
    #     cam_path = dir + "run_%d/flir.avi" % box
    #     outdir = dir + "flir_%d.avi" % box
    #     print(cam_path, outdir)
    #
    #     # detection(cam_path, outdir, showFPS, w, h, skip=0)
    #
    #     with open(dir + "flir_%d.json" % box) as f:
    #         poses = json.load(f)
    #         print(len(poses), poses[1000])
    #
    #     vid = cv2.VideoCapture(cam_path)
    #     print(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    #     # redraw(poses, cam_path, outdir, showFPS, w, h, stop_after=3000)
    #     redraw_batch(poses, cam_path, outdir, showFPS, w, h, stop_after=len(poses), batch_size=16)



