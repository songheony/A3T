import numpy as np
from ortools.graph import pywrapgraph
import torch
from torchvision import models
from torchvision import transforms


def map_node2id(detections):
    node2id = {}
    node2id["source"] = 0
    node2id["sink"] = 1

    nextid = 2
    for frame_id, rects in enumerate(detections):
        for i, rect in enumerate(rects):
            node2id[(frame_id, i)] = nextid
            nextid += 1
    return node2id


def map_id2node(detections):
    id2node = {}
    id2node[0] = "source"
    id2node[1] = "sink"

    nextid = 2
    for frame_id, rects in enumerate(detections):
        for i, rect in enumerate(rects):
            id2node[nextid] = (frame_id, i)
            nextid += 1
    return id2node


class MinCostFlowTracker:
    """
        Object tracking based on data association via minimum cost flow algorithm
        L. Zhang et al.,
        "Global data association for multi-object tracking using network flows",
        CVPR 2008
    """

    def __init__(self, cost_link):
        self._cost_link = cost_link

    def _build_network(self, detections, f2i_factor=10000):
        self.mcf = pywrapgraph.SimpleMinCostFlow()
        self._id2node = map_id2node(detections)
        self._node2id = map_node2id(detections)

        for frame_id, infos in enumerate(detections):
            if frame_id == 0:
                for i in range(len(infos)):
                    self.mcf.AddArcWithCapacityAndUnitCost(
                        self._node2id["source"], self._node2id[(frame_id, i)], 1, 0
                    )
            elif frame_id == len(detections) - 1:
                for i in range(len(infos)):
                    self.mcf.AddArcWithCapacityAndUnitCost(
                        self._node2id[(frame_id, i)], self._node2id["sink"], 1, 0
                    )

            if frame_id == 0:
                continue

            prev_frame_id = frame_id - 1

            for i, i_info in enumerate(detections[prev_frame_id]):
                for j, j_info in enumerate(infos):
                    self.mcf.AddArcWithCapacityAndUnitCost(
                        self._node2id[(prev_frame_id, i)],
                        self._node2id[(frame_id, j)],
                        1,
                        int(self._cost_link(i_info, j_info) * f2i_factor),
                    )

    def _make_flow_dict(self):
        self.flow_dict = {}
        self.ids = []
        for i in range(self.mcf.NumArcs()):
            if self.mcf.Flow(i) > 0:
                tail = self.mcf.Tail(i)
                head = self.mcf.Head(i)
                if self._id2node[tail] in self.flow_dict:
                    self.flow_dict[self._id2node[tail]][self._id2node[head]] = 1
                else:
                    self.flow_dict[self._id2node[tail]] = {self._id2node[head]: 1}
                self.ids.append(self._id2node[head])

    def run(self, detections):
        self._build_network(detections)

        self.mcf.SetNodeSupply(self._node2id["source"], 1)
        self.mcf.SetNodeSupply(self._node2id["sink"], -1)

        if self.mcf.Solve() == self.mcf.OPTIMAL:
            cost = self.mcf.OptimalCost()
        else:
            print("There was an issue with the min cost flow input.")
            return None

        self._make_flow_dict()
        return cost


def calc_overlap(rect1, rect2):
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def cosine_similarity(ft1, ft2):
    return np.dot(ft1, ft2) / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


class Extractor:
    def __init__(self, device):
        self.model = models.resnet18(pretrained=True).to(device)
        self.model.eval()
        self.extractor = self.model._modules.get("avgpool")
        self.transform = transforms.Compose(
            [
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def extract(self, image, bboxes):
        features = []
        croped_images = []
        for bbox in bboxes:
            max_x = min(image.size[0], bbox[0] + bbox[2])
            max_y = min(image.size[1], bbox[1] + bbox[3])
            min_x = max(0, bbox[0])
            min_y = max(0, bbox[1])
            croped_images.append(image.crop((min_x, min_y, max_x, max_y)))
        features = self._get_vector(croped_images)
        features = features.data.cpu().numpy()
        return features

    def _get_vector(self, x):
        x = self.transform(x)
        x = x.cuda()
        embedding = torch.zeros((x.shape[0], 512))

        def copy_data(m, i, o):
            embedding.copy_(o.data.view(x.shape[0], -1).data)

        h = self.extractor.register_forward_hook(copy_data)
        self.resnet(x)
        h.remove()

        return embedding
