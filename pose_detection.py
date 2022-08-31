# Mosaic.py : For creating mosaic video / each session
import os
import json
import cv2
import cvlib as cv
import matplotlib.pyplot as plt
import numpy as np
import glob
import mediapipe as mp
from natsort import natsorted
from multiprocessing import Pool

#  Read frame, anonymize faces, detect poses and return it.
def detection(video_path, frame_rate):
    gstream = "filesrc location=%s ! image/jpeg,width=2592,height=1944,framerate=15/1 ! jpegdec ! " \
              "videoconvert ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! queue ! " \
              "appsink" % video_path
    try:
        reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
    except Exception as e:
        print(e)
    # Initializing pose detector
    # initialize Pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    ret, frame = reader.read()
    while ret:
        #  Anonymization of faces
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, confidences = cv.detect_face(frame)
        for face in faces:
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            roi = frame[startY:startY + endY, startX:startX + endX]
            # applying a gaussian blur over this new rectangle area
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            # impose this blurred image on original image to get final image
            frame[startY:startY + roi.shape[0], startX:startX + roi.shape[1]] = roi

        # Pose detection in image
        results = pose.process(frame)
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # Writer object
                                    frame_rate, (width * 3, height * 3))


if __name__ == "__main__":
    root = "/home/summer/nec/Aug_1/"                               # Root path to one recording session
    session_id = input("Enter Session ID:")                        # Session ID
    frame_rate = int(input("Enter frame-rate:"))                   # Frame Rate for rendering
    video_format = input("Enter video format:")
    dir = root+"session_"+session_id

    for cam_path in glob.glob(dir+"*"+video_format):
        outdir = dir + os.path.split(cam_path)[-1][1]
        with Pool(16) as p:
            p.map(detection, (cam_path, outdir, frame_rate))