import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["C_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import tensorflow as tf
import path_config

tf.get_logger().setLevel("INFO")
from base_tracker import BaseTracker

sys.path.append("external/MemTrack/")
from tracking.tracker import Tracker, Model


class MemTrack(BaseTracker):
    def __init__(self):
        super(MemTrack, self).__init__("MemTrack")
        self.config_proto = tf.ConfigProto()
        self.config_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config_proto)
        ckpt = tf.train.get_checkpoint_state(
            path_config.MEMTRACK_MODEL
        )
        self.model = Model(self.sess, ckpt.model_checkpoint_path)

    def initialize(self, image_file, box):
        self.tracker = Tracker(self.model)
        self.tracker.initialize(image_file, box)

    def track(self, image_file):
        bbox, _ = self.tracker.track(image_file)
        return bbox
