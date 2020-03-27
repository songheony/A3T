import argparse
import time
import numpy as np
import zmq
from options import select_expert


MessageType = {"init": 0, "track": 1, "result": 2}


class Message:
    def __init__(self, messageType, data):
        self.messageType = messageType
        self.data = data


def main(tracker_name):
    tracker = select_expert(tracker_name)
    context = zmq.Context()

    ventilator = context.socket(zmq.PULL)
    ventilator.bind("tcp://*:8888")

    sink = context.socket(zmq.PUSH)
    sink.bind("tcp://*:6006")

    while True:
        msg = ventilator.recv_pyobj()
        target = msg.data["target"]
        if target == "all" or target == tracker_name:
            if msg.messageType == MessageType["init"]:
                image_file = msg.data["image_file"]
                box = np.array(msg.data["box"])
                tracker.initialize(image_file, box)
            elif msg.messageType == MessageType["track"]:
                image_file = msg.data["image_file"]
                start_time = time.time()
                box = tracker.track(image_file)
                duration = time.time() - start_time
                data = {"name": tracker_name, "box": box, "time": duration}
                message = Message(MessageType["result"], data)
                sink.send_pyobj(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert", default="ECO-HC", type=str, help="expert")
    args = parser.parse_args()

    main(args.expert)
