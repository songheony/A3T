import time
import numpy as np
from PIL import Image


class Algorithm(object):
    def __init__(self, name):
        self.name = name

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image, boxes):
        raise NotImplementedError()

    def track(self, img_files, box, predicts):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image, predicts[f, :])
            times[f] = time.time() - start_time

        return boxes, times
