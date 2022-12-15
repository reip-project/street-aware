import cv2
import time
import json
import numpy as np

# https://github.com/jackersson/gstreamer-python
from gstreamer import Gst, GstApp, GstContext, GstPipeline
import gstreamer.utils as utils


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


class GstVideo:
    def __init__(self, filename, width, height, fps, format="RGB", bitrate=10000, variable=True, codec="h265", gpu=0):
        self.context = GstContext()
        self.context.startup()

        assert codec in ["h264", "h265"], "Unsupported codec %d" % codec

        self.caps = "video/x-raw,format=%s,width=%d,height=%d,framerate=%d/1" % (format, width, height, fps)
        self.command = "appsrc emit-signals=True is-live=False caps=%s ! queue ! videoconvert ! " \
                       "nv%senc preset=hq bitrate=%d rc-mode=%s gop-size=45 cuda-device-id=%d ! " \
                       "%sparse ! matroskamux ! filesink location=%s" \
                       % (self.caps, codec, bitrate, "vbr" if variable else "cbr", gpu, codec, filename)
        self.pts = 0  # frame timestamp
        self.duration = 10**9 / fps  # frame duration
        self.pipeline, self.appsrc = GstPipeline(self.command), None
        self.pipeline._on_pipeline_init = self.on_pipeline_init
        self.pipeline.startup()

    def on_pipeline_init(self):
        """Setup AppSrc element"""
        self.appsrc = self.pipeline.get_by_cls(GstApp.AppSrc)[0]  # get AppSrc

        # instructs appsrc that we will be dealing with timed buffer
        self.appsrc.set_property("format", Gst.Format.TIME)

        # instructs appsrc to block pushing buffers until ones in queue are preprocessed
        # allows to avoid huge queue internal queue size in appsrc
        # self.appsrc.set_property("block", True)

        # set input format (caps)
        self.appsrc.set_caps(Gst.Caps.from_string(self.caps))

    def write(self, frame):
        # assert type(frame) == np.ndarray and frame.shape == (self.height, self.width, 3) and frame.dtype == np.uint8

        gst_buffer = utils.ndarray_to_gst_buffer(frame)
        # set pts and duration to be able to record video, calculate fps
        gst_buffer.pts = self.pts
        gst_buffer.duration = self.duration
        self.pts += self.duration

        # emit <push-buffer> event with Gst.Buffer
        self.appsrc.emit("push-buffer", gst_buffer)

    def close(self):
        self.appsrc.emit("end-of-stream")

        while not self.pipeline.is_done:
            time.sleep(.1)

        self.pipeline.shutdown()
        self.context.shutdown()


if __name__ == '__main__':
    path = './'

    w, h, fps, n = 1280, 1024, 120, 1
    writer = GstVideo(path + "test.mp4", w, h, fps, codec="h265")

    t0 = time.time()
    for i in range(n*fps):
        if i % 10 == 0:
            print("writing", i)

        frame = np.zeros((h, w, 3), np.uint8)
        cv2.putText(frame, str(i), (w // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3, cv2.LINE_AA)

        writer.write(frame)
        time.sleep(0.01)

    writer.close()
    print("Written in %.1f sec" % (time.time()-t0))

    reader = cv2.VideoCapture(path + "test.mp4")
    success, image = reader.read()
    count = 0

    while success:
        if count % 10 == 0:
            print("read", count)

        assert image.shape == (h, w, 3)
        # cv2.imwrite(path + "unpack/frame_%d.jpg" % count, image)

        count += 1
        success, image = reader.read()

    print(count, n*fps)
    assert count == n*fps
