import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["C_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tensorflow as tf
import path_config

tf.get_logger().setLevel("INFO")
from base_tracker import BaseTracker

sys.path.append("external/MemDTC/")
from tracking.tracker import Tracker, Model


class MemDTC(BaseTracker):
    def __init__(self):
        super(MemDTC, self).__init__("MemDTC")
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        ckpt = tf.train.get_checkpoint_state(path_config.MEMDTC_MODEL)
        sess = tf.Session(config=config_proto)
        model = Model(sess, ckpt.model_checkpoint_path)
        self.tracker = Tracker(model)

    def initialize(self, image_file, box):
        self.tracker.initialize(image_file, box)

    def track(self, image_file):
        bbox, _ = self.tracker.track(image_file)
        return bbox
