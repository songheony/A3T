import sys
import tensorflow as tf
from .expert import Expert

sys.path.append("external/MemTrack/")
from tracking.tracker import Tracker, Model


class MemTrack(Expert):
    def __init__(self):
        super(MemTrack, self).__init__("MemTrack")
        self.config_proto = tf.ConfigProto()
        self.config_proto.gpu_options.allow_growth = True
        tf.Graph().as_default()
        self.sess = tf.Session(config=self.config_proto)

    def initialize(self, image_file, box):
        self.model = Model(self.sess)
        self.tracker = Tracker(self.model)
        self.tracker.initialize(image_file, box)

    def track(self, image_file):
        bbox, _ = self.tracker.track(image_file)
        return bbox
