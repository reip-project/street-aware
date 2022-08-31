# Object-Detection.py : For detecting objects video / each session
# Depends on https://github.com/HRNet/HRNet-Object-Detection (in object-detection subfolder)
import cvlib as cv
import glob
import argparse
import os
import cv2
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
import json
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

import matplotlib.pyplot as plt
#  Read frame, anonymize faces, detect poses and return it.
def detection(video_path, newpath, frame_rate):

    gstream = "multifilesrc location=%s ! image/jpeg,width=2592,height=1944,framerate=15/1 ! jpegdec ! " \
                      "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
                      "appsink" % video_path
    try:
        reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
        # writer = cv2.VideoWriter(newpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),frame_rate,(2592 ,1944))     # Writer
    except Exception as e:
        print(e)
    print("Trying to read!")
    ret, frame = reader.read()
    number_of_objects = []
    # print(ret,frame)
    checkpoints = '/mnt/tmp/nec/nec-dataset/object-detection/mmdetection/checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    config_file = '/mnt/tmp/nec/nec-dataset/object-detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
    # Set the device to be used for evaluation
    device = 'cuda:0'

    # Load the config
    config = mmcv.Config.fromfile(config_file)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoints, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config
    c =0
    while ret:
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
        result = inference_detector(model, frame)
        # print(result)
        objects = show_result_pyplot(model, frame, result)
        number_of_objects.append(objects)
        # writer.write(frame)
        ret, frame = reader.read()
        c+=1
        if c%50 == 0:
            print("{} frames written".format(c))

    reader.release()
        # data = json.dump(number_persons)
    with open(newpath+'.json', 'w') as f:
        json.dump(number_of_objects, f)

def plot(session, sensor, camera):
    filename = "all_occupancy_ss%d_s%d_%s" % (session, sensor, camera)
    newpath = r'/mnt/tmp/nec/nec-dataset/results/'
    data = np.array(json.load(open(newpath+filename + ".json")))
    print(data, data.shape)
    lists = [list(val) for val in zip(*[d.values() for d in data])]
    fps, n = 15, 10
    m = fps * n
    N = data.shape[0] // m
    binned = []
    for i in range(len(lists)):
        binned.append(np.sum(np.array(lists[i])[:N * m].reshape(-1, m), axis=1) / m)
    plt.figure("Average occupancy", (16, 9))
    for i in range(len(lists)):
        plt.plot(np.arange(N) * n / 60, binned[i])
    plt.xlabel("Session %d time, minutes" % session)
    plt.ylabel("Average occupancy, people")
    plt.title("Chase bank crossing (Sensor %d, %s)" % (sensor, camera))
    plt.tight_layout()
    plt.legend(['person', 'bicycle', 'car', 'motorcycle','bus', 'truck'])
    plt.savefig(newpath+filename + ".png", dpi=200)
    plt.show()

    exit(0)

if __name__ == "__main__":
    plot(session=3, sensor=3, camera="left")
    root = "/mnt/tmp/nec/Aug_2/"
    sessionID = str(3)
    # sensor = input("Sensor")
    sensor = str(3)
    dir = root + "session_" + sessionID + "/"
    showFPS = 15

    cam_path = dir + 'sensor_' + sensor + "/video/1_1659464601_%d.avi"
    # for cam_path in glob.glob(dir+"*avi"):
    # print(cam_path)
    newpath = r'/mnt/tmp/nec/nec-dataset/results/'
    newpath += 'all_occupancy_ss'+sessionID+'_s'+sensor+'_left'
    detection(cam_path, newpath, showFPS)


