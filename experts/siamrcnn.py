import sys
from types import SimpleNamespace
from PIL import Image
from base_tracker import BaseTracker

sys.path.append("external/SiamR-CNN/")
from tracking.argmax_tracker import ArgmaxTracker
from tracking.three_stage_tracker import ThreeStageTracker


def build_tracker(args):
    if args.tracker == "ArgmaxTracker":
        return ArgmaxTracker()
    elif args.tracker == "ThreeStageTracker":
        pass
    else:
        assert False, ("Unknown tracker", args.tracker)

    tracklet_param_str = (
        str(args.tracklet_distance_threshold)
        + "_"
        + str(args.tracklet_merging_threshold)
        + "_"
        + str(args.tracklet_merging_second_best_relative_threshold)
    )
    if args.n_proposals is not None:
        tracklet_param_str += "_proposals" + str(args.n_proposals)
    if args.resolution is not None:
        tracklet_param_str += "_resolution-" + str(args.resolution)
    if args.model != "best":
        tracklet_param_str = args.model + "_" + tracklet_param_str
    if args.visualize_tracker:
        tracklet_param_str2 = "viz_" + tracklet_param_str
    else:
        tracklet_param_str2 = tracklet_param_str
    param_str = (
        tracklet_param_str2
        + "_"
        + str(args.ff_gt_score_weight)
        + "_"
        + str(args.ff_gt_tracklet_score_weight)
        + "_"
        + str(args.location_score_weight)
    )

    name = "ThreeStageTracker_" + param_str
    tracker = ThreeStageTracker(
        tracklet_distance_threshold=args.tracklet_distance_threshold,
        tracklet_merging_threshold=args.tracklet_merging_threshold,
        tracklet_merging_second_best_relative_threshold=args.tracklet_merging_second_best_relative_threshold,
        ff_gt_score_weight=args.ff_gt_score_weight,
        ff_gt_tracklet_score_weight=args.ff_gt_tracklet_score_weight,
        location_score_weight=args.location_score_weight,
        name=name,
        do_viz=args.visualize_tracker,
        model=args.model,
        n_proposals=args.n_proposals,
        resolution=args.resolution,
    )
    return tracker


class SiamRCNN(BaseTracker):
    def __init__(self):
        super(SiamRCNN, self).__init__("SiamR-CNN")
        conf = {
            "tracklet_distance_threshold": 0.06,
            "tracklet_merging_threshold": 0.3,
            "tracklet_merging_second_best_relative_threshold": 0.3,
            "ff_gt_score_weight": 0.1,
            "ff_gt_tracklet_score_weight": 0.9,
            "location_score_weight": 7.0,
            "model": "best",
            "tracker": "ThreeStageTracker",
            "n_proposals": None,
            "resolution": None,
            "visualize_tracker": False,
        }
        self.args = SimpleNamespace(**conf)

    def initialize(self, image_file, box):
        image = Image.open(image_file).convert("RGB")
        self.tracker = build_tracker(self.args)
        self.tracker.init(image, box)

    def track(self, image_file):
        image = Image.open(image_file).convert("RGB")
        return self.update(image)
