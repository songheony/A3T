import time
import os
import cv2
import sys
import numpy as np

sys.path.append("external/pytracking")
from pytracking.evaluation.environment import env_settings


class Expert(object):
    """Base class for all trackers."""

    def __init__(self, name):
        self.name = name
        env = env_settings()
        self.results_dir = "{}/{}".format(env.results_path, self.name)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.initialize(image, np.array(sequence.init_state))
        init_time = getattr(self, "time", time.time() - start_time)
        times.append(init_time)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

        return tracked_bb, times

    def _read_image(self, image_file: str):
        return cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

    def run(self, seq):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
        """

        output_bb, execution_times = self.track_sequence(seq)

        return output_bb, execution_times
