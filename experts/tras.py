import sys
from PIL import Image
from base_tracker import BaseTracker
import path_config

sys.path.append("external/vot-kd-rl/")
from track.Trackers import TRAS, TRAST, TRASFUST
from track.config_track_accv import Configuration


class ETRAS(BaseTracker):
    def __init__(self):
        super(ETRAS, self).__init__("TRAS")
        self.cfg = Configuration()
        self.cfg.CKPT_PATH = path_config.TRAS_MODEL

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker = TRAS(self.cfg)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.tracker.update(image)


class ETRAST(BaseTracker):
    def __init__(self):
        super(ETRAST, self).__init__("TRAST")
        self.cfg = Configuration()
        self.cfg.CKPT_PATH = path_config.TRAS_MODEL

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker = TRAST(self.cfg)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.tracker.update(image)


class ETRASFUST(BaseTracker):
    def __init__(self):
        super(ETRASFUST, self).__init__("TRASFUST")
        self.cfg = Configuration()
        self.cfg.CKPT_PATH = path_config.TRAS_MODEL

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker = TRASFUST(self.cfg)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.tracker.update(image)
