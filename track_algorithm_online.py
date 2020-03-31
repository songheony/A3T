import os
import sys
import pickle
import random
import torch
import numpy as np
import zmq
from track_expert_online import Message, MessageType
from options import select_algorithms, select_datasets

sys.path.append("external/pysot-toolkit/pysot")
from utils import vot_overlap

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)


def run_sequence(seq, tracker, experts, ventilators, sinks):
    """Runs a tracker on a sequence."""

    def initialize(image_file, box):
        data = {"image_file": image_file, "box": box, "target": "all"}
        message = Message(MessageType["init"], data)

        for ventilator in ventilators:
            ventilator.send_pyobj(message)

        tracker.initialize(image_file, box)

    def track(image_file):
        data = {"image_file": image_file, "target": "all"}
        message = Message(MessageType["track"], data)
        for ventilator in ventilators:
            ventilator.send_pyobj(message)

        box = {}
        for sink in sinks:
            result = sink.recv_pyobj()
            result = result.data
            box[result["name"]] = result["box"]

        box = np.array([box[expert] for expert in experts])

        state, offline, weight = tracker.track(frame, box)
        return state, offline, weight, box

    base_results_path = "{}/{}_supervised".format(tracker.results_dir, seq.name)
    results_path = "{}.txt".format(base_results_path)
    weights_path = "{}_weight.txt".format(base_results_path)
    offline_path = "{}_offline.pkl".format(base_results_path)
    boxes_path = "{}_boxes.txt".format(base_results_path)

    if os.path.isfile(results_path):
        return

    print("Tracker: {},  Sequence: {}".format(tracker.name, seq.name))

    boxes = [[] for _ in range(len(experts))]
    tracked_bb = []
    offline_bb = []
    weights = []

    # Track
    frame_counter = 0
    for n, frame in enumerate(seq.frames):
        img = tracker._read_image(frame)
        if n == frame_counter:
            initialize(seq.frames[n], np.array(seq.init_bbox()))
            for i in range(len(experts)):
                boxes[i].append(1)
            offline_bb.append(1)
            weights.append(1)

        elif n > frame_counter:
            state, offline, weight, box = track(frame)

            overlap = vot_overlap(
                state, seq.ground_truth_rect[n], (img.shape[1], img.shape[0])
            )
            if overlap > 0:
                for i in range(len(experts)):
                    boxes[i].append(box[i])
                tracked_bb.append(state)
                offline_bb.append(offline)
                weights.append(weight)
            else:
                for i in range(len(experts)):
                    boxes[i].append(2)
                tracked_bb.append(2)
                offline_bb.append(2)
                weights.append(2)

        else:
            for i in range(len(experts)):
                boxes[i].append(0)
            tracked_bb.append(0)
            offline_bb.append(0)
            weights.append(0)

    with open(weights_path, "wb") as fp:
        pickle.dump(weights, fp)
    with open(offline_path, "wb") as fp:
        pickle.dump(offline_bb, fp)
    with open(boxes_path, "wb") as fp:
        pickle.dump(boxes, fp)
    with open(results_path, "wb") as fp:
        pickle.dump(tracked_bb, fp)


def main(algorithm_name, experts, dataset_name, **kwargs):
    algorithm = select_algorithms(algorithm_name, experts, **kwargs)
    dataset = select_datasets(dataset_name)

    context = zmq.Context()

    ventilators = [context.socket(zmq.PUSH) for _ in range(len(experts))]
    for ventilator, expert in zip(ventilators, experts):
        ventilator.connect("tcp://%s:8888" % expert)

    sinks = [context.socket(zmq.PULL) for _ in range(len(experts))]
    for sink, expert in zip(sinks, experts):
        sink.connect("tcp://%s:6006" % expert)

    for seq in dataset:
        run_sequence(seq, algorithm, experts, ventilator, sinks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="AAA", type=str)
    parser.add_argument("-e", "--experts", default=list(), nargs="+")
    parser.add_argument("-d", "--dataset", default="VOT", type=str)
    parser.add_argument("-n", "--mode", default="Expert", type=str)
    parser.add_argument("-t", "--iou_threshold", default=0.0, type=float)
    parser.add_argument("-r", "--feature_threshold", default=0.0, type=float)
    parser.add_argument("-s", "--reset_target", action="store_true")
    parser.add_argument("-m", "--only_max", action="store_true")
    parser.add_argument("-i", "--use_iou", action="store_true")
    parser.add_argument("-f", "--use_feature", action="store_false")
    parser.add_argument("-x", "--cost_iou", action="store_false")
    parser.add_argument("-y", "--cost_feature", action="store_false")
    parser.add_argument("-z", "--cost_score", action="store_false")
    args = parser.parse_args()

    main(
        args.algorithm,
        args.experts,
        args.dataset,
        mode=args.mode,
        iou_threshold=args.iou_threshold,
        feature_threshold=args.feature_threshold,
        reset_target=args.reset_target,
        only_max=args.only_max,
        use_iou=args.use_iou,
        use_feature=args.use_feature,
        cost_iou=args.cost_iou,
        cost_feature=args.cost_feature,
        cost_score=args.cost_score,
    )
