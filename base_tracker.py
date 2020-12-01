import copy
import time
import os
import numpy as np
import cv2
import path_config
from print_manager import do_not_print


class BaseTracker(object):
    """Base class for all trackers."""

    def __init__(self, name):
        self.name = name
        self.results_dir = "{}/{}".format(path_config.RESULTS_PATH, self.name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def initialize(self, image, state):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image, boxes):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence, experts=None):
        """Run tracker on a sequence."""

        if experts is not None:
            boxes = np.zeros((len(experts), len(sequence.ground_truth_rect), 4))
            tracker_times = np.zeros((len(experts), len(sequence.ground_truth_rect)))
            for n, tracker_name in enumerate(experts):
                results_dir = "{}/{}".format(path_config.RESULTS_PATH, tracker_name)
                base_results_path = "{}/{}".format(results_dir, sequence.name)
                results_path = "{}.txt".format(base_results_path)
                tracker_traj = np.loadtxt(results_path, delimiter="\t", dtype=float)
                times_path = "{}_time.txt".format(base_results_path)
                tracker_time = np.loadtxt(times_path, delimiter="\t", dtype=float)
                boxes[n] = tracker_traj
                tracker_times[n] = tracker_time

        times = []
        start_time = time.time()
        self.initialize(sequence.frames[0], np.array(sequence.init_bbox()))
        init_time = time.time() - start_time
        times.append(init_time)

        # Track
        tracked_bb = [sequence.init_bbox()]
        offline_bb = []
        weights = []
        for n, frame in enumerate(sequence.frames[1:]):
            if experts is not None:
                start_time = time.time()
                state, offline, weight = self.track(frame, boxes[:, n + 1, :])
                calc_time = time.time() - start_time
                last_time = np.max(tracker_times[:, n + 1])
                duration = calc_time + last_time
            else:
                start_time = time.time()
                state = self.track(frame)
                duration = time.time() - start_time
                offline = None
                weight = None
            times.append(duration)
            tracked_bb.append(copy.deepcopy(state))
            offline_bb.append(copy.deepcopy(offline))
            weights.append(copy.deepcopy(weight))

        return tracked_bb, offline_bb, weights, times

    def _read_image(self, image_file: str):
        return cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

    @do_not_print
    def run(self, seq, trackers):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
        """

        output_bb, offline_bb, weights, execution_times = self.track_sequence(
            seq, trackers
        )

        return output_bb, offline_bb, weights, execution_times
