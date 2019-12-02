import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["C_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from .expert import Expert

sys.path.append("external/MemDTC/")
from tracking.tracker import Tracker, Model


class MemDTC(Expert):
    def __init__(self):
        super(MemDTC, self).__init__("MemDTC")
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        ckpt = tf.train.get_checkpoint_state(
            "/home/heonsong/Desktop/AAA/AAA-journal/external/MemDTC/output/models"
        )
        self.model = Model(self.sess, ckpt.model_checkpoint_path)

    def initialize(self, image_file, box):
        self.tracker = Tracker(self.model)
        self.tracker.initialize(image_file, box)

    def track(self, image_file):
        bbox, _ = self.tracker.track(image_file)
        return bbox
