import time
import os
import sys
import numpy as np
import cv2


sys.path.append("external/pysot-toolkit/pysot")
sys.path.append("external/pytracking")
from pytracking.evaluation.environment import env_settings
from utils import vot_overlap


class BaseTracker(object):
    """Base class for all trackers."""

    def __init__(self, name):
        self.name = name
        env = env_settings()
        self.results_dir = "{}/{}".format(env.results_path, self.name)
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
                results_dir = "{}/{}".format(env_settings().results_path, tracker_name)
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
        init_time = getattr(self, "time", time.time() - start_time)
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
                last_time = np.max(tracker_times[:, n])
                duration = calc_time + last_time
            else:
                start_time = time.time()
                state = self.track(frame)
                duration = time.time() - start_time
                offline = None
                weight = None
            times.append(duration)
            tracked_bb.append(state)
            offline_bb.append(offline)
            weights.append(weight)

        return tracked_bb, offline_bb, weights, times

    def track_supervised(self, sequence):
        """Run tracker on a sequence."""

        # Track
        frame_counter = 0
        tracked_bb = []
        for n, frame in enumerate(sequence.frames):
            img = self._read_image(frame)

            if n == frame_counter:
                self.initialize(frame, np.array(sequence.init_bbox()))
                tracked_bb.append(1)
            elif n > frame_counter:
                state, _, _ = self.track(frame)
                overlap = vot_overlap(
                    state, sequence.ground_truth_rect[n], (img.shape[1], img.shape[0])
                )
                if overlap > 0:
                    tracked_bb.append(state)
                else:
                    tracked_bb.append(2)
                    frame_counter = n + 5
            else:
                tracked_bb.append(0)

        return tracked_bb

    def _read_image(self, image_file: str):
        return cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

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
