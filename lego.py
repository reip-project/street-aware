import os
import glob

from gst_video import *

W, H, FPS = 1920, 1080, 15


if __name__ == '__main__':
    data_path = '/home/vidaviz/lego/following_example/'

    # writer = GstVideo(data_path + "../following_example.mp4", W, H, FPS, format="BGR", bitrate=3000, variable=True, codec="h264")
    writer = GstVideo(data_path + "../following_example.mkv", W, H, FPS, format="BGR", bitrate=2000, variable=True, codec="h265")

    files = sorted(glob.glob(data_path + "*.jpg"))

    for i, file in enumerate(files):
        print(i, "of", len(files), file)
        # continue

        img = cv2.imread(file)
        if img.shape[0] == 1200:
            img = img[:1080, :, :]
            writer.write(img)
            writer.write(img)
        else:
            img = img[::2, ::2, :]
            for j in range(8):
                writer.write(img)

    writer.close()
