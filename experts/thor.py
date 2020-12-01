import sys
import json
import cv2
import torch
from base_tracker import BaseTracker
import path_config

sys.path.append("external/THOR/")
from trackers.THOR_modules.wrapper import THOR_SiamFC, THOR_SiamRPN, THOR_SiamMask

# SiamFC import
from trackers.SiamFC.net import SiamFC
from trackers.SiamFC.siamfc import SiamFC_init, SiamFC_track

# SiamRPN Imports
from trackers.SiamRPN.net import SiamRPN
from trackers.SiamRPN.siamrpn import SiamRPN_init, SiamRPN_track

# SiamMask Imports
from trackers.SiamMask.net import SiamMaskCustom
from trackers.SiamMask.siammask import SiamMask_init, SiamMask_track
from trackers.SiamMask.utils.load_helper import load_pretrain

sys.path.append("external/THOR/benchmark")
from bench_utils.bbox_helper import cxy_wh_2_rect, rect_2_cxy_wh


class THOR(BaseTracker):
    def __init__(self):
        super(THOR, self).__init__("THOR")
        tracker = "SiamRPN"
        dataset = "OTB2015"
        vanilla = False
        lb_type = "dynamic"  # [dynamic, ensemble]
        json_path = f"{path_config.THOR_CONFIG}/{tracker}/"
        json_path += f"{dataset}_"
        if vanilla:
            json_path += "vanilla.json"
        else:
            json_path += f"THOR_{lb_type}.json"
        cfg = json.load(open(json_path))
        cfg["THOR"]["viz"] = False
        cfg["THOR"]["verbose"] = False

        if tracker == "SiamFC":
            self.tracker = SiamFC_Tracker(cfg, path_config.THOR_SIAMFC_MODEL)
        elif tracker == "SiamRPN":
            self.tracker = SiamRPN_Tracker(cfg, path_config.THOR_SIAMRPN_MODEL)
        elif tracker == "SiamMask":
            self.tracker = SiamMask_Tracker(cfg, path_config.THOR_SIAMMASK_MODEL)
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


class Tracker:
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.mask = False
        self.temp_mem = None

    def init_func(self, im, pos, sz):
        raise NotImplementedError

    def track_func(self, state, im):
        raise NotImplementedError

    def setup(self, im, target_pos, target_sz):
        state = self.init_func(im, target_pos, target_sz)
        self.temp_mem.setup(im, target_pos, target_sz)
        return state

    def track(self, im, state):
        state = self.track_func(state, im)
        self.temp_mem.update(im, state["crop"], state["target_pos"], state["target_sz"])
        return state


class SiamFC_Tracker(Tracker):
    def __init__(self, cfg, model_path):
        super(SiamFC_Tracker, self).__init__()
        self.cfg = cfg

        # setting up the tracker
        # model_path = dirname(abspath(__file__)) + '/SiamFC/model.pth'
        model = SiamFC()
        model.load_state_dict(torch.load(model_path))
        self.model = model.eval().to(self.device)

        # set up template memory
        self.temp_mem = THOR_SiamFC(cfg=cfg["THOR"], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamFC_init(im, pos, sz, self.cfg["tracker"])

    def track_func(self, state, im):
        return SiamFC_track(state, im, self.temp_mem)


class SiamRPN_Tracker(Tracker):
    def __init__(self, cfg, model_path):
        super(SiamRPN_Tracker, self).__init__()
        self.cfg = cfg

        # setting up the model
        # model_path = dirname(abspath(__file__)) + '/SiamRPN/model.pth'
        model = SiamRPN()
        model.load_state_dict(
            torch.load(
                model_path, map_location=("cpu" if str(self.device) == "cpu" else None)
            )
        )
        self.model = model.eval().to(self.device)

        # set up template memory
        self.temp_mem = THOR_SiamRPN(cfg=cfg["THOR"], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamRPN_init(im, pos, sz, self.cfg["tracker"])

    def track_func(self, state, im):
        return SiamRPN_track(state, im, self.temp_mem)


class SiamMask_Tracker(Tracker):
    def __init__(self, cfg, model_path):
        super(SiamMask_Tracker, self).__init__()
        self.cfg = cfg
        self.mask = True

        # setting up the model
        # model_path = dirname(abspath(__file__)) + '/SiamMask/model.pth'
        model = SiamMaskCustom(anchors=cfg["anchors"])
        model = load_pretrain(model, model_path)
        self.model = model.eval().to(self.device)

        # set up template memory
        self.temp_mem = THOR_SiamMask(cfg=cfg["THOR"], net=self.model)

    def init_func(self, im, pos, sz):
        return SiamMask_init(im, pos, sz, self.model, self.cfg["tracker"])

    def track_func(self, state, im):
        return SiamMask_track(state, im, self.temp_mem)
