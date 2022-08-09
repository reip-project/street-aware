# Mosaic.py : For creating mosaic video / each session
import os
import json
import cv2
import numpy
import glob


class Reader:
    def __init__(self, json_path, video_path):
        self.json_path = json_path
        self.video_path = video_path
        with open(self.json_path, 'rb') as inFile:
            self.timeline = json.load(inFile)
        self.sensor_id = os.path.split(self.json_path)[-1][1]
        self.camera_id = os.path.split(self.json_path)[-1][5]

    def read_json(self) -> int:
        # Setting camera id, sensor id from filenames and reading the json file

        pass

    def render_frame(self):
        # Reading video for camera frame by frame

        pass

    def __str__(self):
        self.timeline = self.read_json()
        return "Global Timeline for sensor {} camera {}: {}".format(str(self.sensor_id),
                                                                    str(self.camera_id), str(self.timeline))


if __name__ == "__main__":
    root = "/home/summer/nec/Aug_2/"            # Root path to one recording session
    session_id = input("Enter Session ID:")     # Session ID
    session_timelines = []                      # List to store objects for each camera
    master_gt = []                              # Master global timeline

    # Creating objects for each camera global timeline per session
    for i in range(1, 5):
        # Timeline Paths for each sensor's 2 cameras
        json_paths = glob.glob(root + "session_" + str(session_id) + "/sensor_" + str(i) + '/*.json')
        # Video Paths for each sensor's 2 cameras
        video_paths = glob.glob(root + "session_" + str(session_id) + "/sensor_"
                                + str(i) + 'video/*merge.avi')
        # Storing objects for each sensor's 2 cameras
        session_timelines.extend([Reader(json_paths[0], video_paths[0]),
                                  Reader(json_paths[1], video_paths[1])])

    # Reading master global timeline
    for filename in glob.glob(os.path.join(root + "session_" + str(session_id) + "/*.json")):
        with open(os.path.join(os.getcwd(), filename), 'rb') as file:
            master_gt = json.load(file)



