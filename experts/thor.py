import sys
import json
import cv2
from base_tracker import BaseTracker

sys.path.append("external/THOR")
from trackers.tracker import SiamFC_Tracker, SiamRPN_Tracker, SiamMask_Tracker
from benchmark.bench_utils.bbox_helper import cxy_wh_2_rect, rect_2_cxy_wh


class THOR(BaseTracker):
    def __init__(self):
        super(THOR, self).__init__("THOR")
        tracker = "SiamRPN"
        dataset = "OTB2015"
        vanilla = False
        lb_type = "dynamic"  # [dynamic, ensemble]
        json_path = (
            f"/home/heonsong/Desktop/AAA/AAA-journal/external/THOR/configs/{tracker}/"
        )
        json_path += f"{dataset}_"
        if vanilla:
            json_path += "vanilla.json"
        else:
            json_path += f"THOR_{lb_type}.json"
        cfg = json.load(open(json_path))
        cfg["THOR"]["viz"] = False
        cfg["THOR"]["verbose"] = False

        if tracker == "SiamFC":
            self.tracker = SiamFC_Tracker(cfg)
        elif tracker == "SiamRPN":
            self.tracker = SiamRPN_Tracker(cfg)
        elif tracker == "SiamMask":
            self.tracker = SiamMask_Tracker(cfg)
        else:
            raise ValueError(f"Tracker {tracker} does not exist.")

    def initialize(self, image_file, box):
        image = cv2.imread(image_file)
        init_pos, init_sz = rect_2_cxy_wh(box)
        self.state = self.tracker.setup(image, init_pos, init_sz)

    def track(self, image_file):
        image = cv2.imread(image_file)
        self.state = self.tracker.track(image, self.state)
        bbox = cxy_wh_2_rect(self.state["target_pos"], self.state["target_sz"])
        return bbox
