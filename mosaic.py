# Mosaic.py : For creating mosaic video / each session
import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from natsort import natsorted


class Reader:
    def __init__(self, json_path, video_path):
        self.json_path = json_path
        self.video_path = video_path
        with open(self.json_path, 'rb') as inFile:
            self.timeline = json.load(inFile)
        self.left_ind = 0
        self.right_ind = 1
        self.sensor_id = os.path.split(self.json_path)[-1][1]
        self.camera_id = os.path.split(self.json_path)[-1][6]

        try:
            gstream = "filesrc location=%s ! image/jpeg,width=2592,height=1944,framerate=15/1 ! jpegdec ! " \
                      "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
                      "appsink" % video_path
            self.reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
            _, self.left_frame = self.reader.read()
            _, self.right_frame = self.reader.read()

        except Exception as e:
            print(e)

    def get_json(self):

        return self.timeline[self.left_ind], self.timeline[self.right_ind]

    def get_frame(self):

        return self.left_frame, self.right_frame

    def read_frame(self):
        # Reading video for camera frame by frame
        if not self.reader.isOpened():
            print("Cam {} down".format(self.camera_id))
        else:
            self.left_frame = self.right_frame
            ret, self.right_frame = self.reader.read()
            self.left_ind = self.right_ind
            self.right_ind += 1

    def __str__(self):
        # self.timeline = self.read_json()
        return "Sensor {} camera {}".format(str(self.sensor_id), str(self.camera_id))
#
# def test(filename):
#     # print(cv2.getBuildInformation())
#     gstStr = "filesrc location=%s ! image/jpeg,width=2592,height=1944,framerate=15/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! appsink" % filename
#
#     #     print(gstStr)
#     vid = cv2.VideoCapture(gstStr, cv2.CAP_GSTREAMER)
#     # vid = cv2.VideoCapture(filename)
#     _, img = vid.read()
#     print(img, img.shape)
#     plt.imshow(img[:,:,::-1])
#     # cv2.imshow("Test", img)
#     plt.show()


if __name__ == "__main__":

    root = "/home/summer/software/aug1/"                               # Root path to one recording session
    session_id = input("Enter Session ID:")                        # Session ID
    threshold = int(input("Enter threshold for picking frames:"))  # Threshold for picking frames
    camera_readers = []                                            # List to store objects for each camera
    master_gt = []                                                 # Master global timeline
    frame_rate = int(input("Enter frame-rate:"))                   # Frame Rate for rendering
    output_video = cv2.VideoWriter('mosaic.avi',
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # Writer object
                                   frame_rate, (7776, 5832))

    # Creating objects for each camera global timeline per session
    for i in range(1, 5):
        # Timeline Paths for each sensor's 2 cameras
        json_paths = natsorted(glob.glob(root + "session_" + str(session_id) + "/sensor_" + str(i) + '/*_gt.json'))
        # Video Paths for each sensor's 2 cameras
        video_paths = natsorted(glob.glob(root + "session_" + str(session_id) + "/sensor_"
                                + str(i) + '/video/merge_*.avi'))

        # Storing objects for each sensor's 2 cameras
        camera_readers.extend([Reader(json_paths[0], video_paths[0]),
                               Reader(json_paths[1], video_paths[1])])

    # Reading master global timeline
    for filename in glob.glob(os.path.join(root + "session_" + str(session_id) + "/*.json")):
        with open(os.path.join(os.getcwd(), filename), 'rb') as file:
            master_gt = json.load(file)
    c = 0
    # Rendering for global timeline
    for time in master_gt:
        output_array = []          # Array to store all frames for one timestamp
        for cam in camera_readers:
            left_tstamp, right_tstamp = cam.get_json()
            left_fr, right_fr = cam.get_frame()

            if np.abs(time - left_tstamp) < np.abs(time - right_tstamp):
                # print("Reading good frame at time stamp is:{} for cam {}".format(timestamp, cam))
                if np.abs(time - left_tstamp < threshold):
                    output_array.append(left_fr)
                else:
                    output_array.append(np.zeros([1944, 2592, 3], dtype=np.uint8))
            else:
                if np.abs(time - right_tstamp < threshold):
                    output_array.append(right_fr)
                    cam.read_frame()
                else:
                    output_array.append(np.zeros([1944, 2592, 3], dtype=np.uint8))
        # print(output_array[0].shape, output_array[1].shape)
        # exit(0)
        white_frame = np.zeros([1944, 2592, 3], dtype=np.uint8)
        white_frame.fill(255)
        row1 = np.concatenate((white_frame, output_array[0], output_array[1]), axis=1)
        # print(row1.shape)
        row2 = np.concatenate((output_array[2], output_array[3], output_array[4]), axis=1)
        # print(row2.shape)
        row3 = np.concatenate((output_array[5], output_array[6], output_array[7]), axis=1)
        # print(row3.shape)
        stack = np.concatenate((row1, row2, row3), axis=0)

        output_video.write(stack)
        c += 1
        if c % 50 == 0:
            print("{} frames have been wrote!".format(c))
        if c == 50:
            exit(0)
