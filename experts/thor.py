import sys
import json
import cv2
import numpy as np
from .expert import Expert

sys.path.append("external/THOR")
from trackers.tracker import SiamFC_Tracker, SiamRPN_Tracker, SiamMask_Tracker


class THOR(Expert):
    def __init__(self):
        super(THOR, self).__init__("THOR")
        tracker = "SiamRPN"
        dataset = "OTB2015"
        vanilla = False
        lb_type = "ensemble"  # [dynamic, ensemble]
        json_path = f"/home/heonsong/Desktop/AAA/AAA-journal/external/THOR/configs/{tracker}/"
        json_path += f"{dataset}_"
        if vanilla:
            json_path += "vanilla.json"
        else:
            json_path += f"THOR_{lb_type}.json"
        cfg = json.load(open(json_path))
        cfg["THOR"]["viz"] = False
        cfg["THOR"]["verbose"] = False

        print("[INFO] Initializing the tracker.")
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
        init_pos = box[:2]
        init_sz = box[2:]
        self.state = self.tracker.setup(image, init_pos, init_sz)

    def track(self, image_file):
        image = cv2.imread(image_file)
        self.state = self.tracker.track(image, self.state)
        bbox = np.array(self.state["target_pos"][0], self.state["target_pos"][1], self.state["target_sz"][0], self.state["target_sz"][1])
        return bbox
