# Import modules
import os
import cv2
from natsort import natsorted
import glob

if __name__ == "__main__":
    # test("/home/summer/nec/Aug_1/session_1/sensor_1/video/0.avi")
    # test("/home/summer/nec/Aug_1/session_1/sensor_1/video/0_1659380678_1.avi")

    # exit(0)

    root = "/home/summer/nec/Aug_2/"                               # Root path to one recording session
    session_id = input("Enter session id:")
    all_paths = []
    c = 0
    # Creating objects for each camera global timeline per session
    for i in range(1, 5):
        path = root + "session_" + str(session_id) + "/sensor_" + str(i) + '/video/'
        for file in os.listdir(path):
            if file.startswith("0_"):
                os.rename(os.path.join(path, file), os.path.join(path, ''.join([file[:2], file[file.rfind('_'):], ".avi"])))
            elif file.startswith("1_"):
                os.rename(os.path.join(path, file), ''.join([file[:2], file[file.rfind('_'):], ".avi"]))
    for i in range(1, 5):
        for cam in [0, 1]:
            path = root + "session_" + str(session_id) + "/sensor_" + str(i) + '/video/' + cam
            gstream = "multifilesrc location=%s ! image/jpeg,width=2592,height=1944,framerate=15/1 ! queue ! avimux ! filesink location=merge_%s.avi" % path %cam
            reader = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)
            reader.read()
