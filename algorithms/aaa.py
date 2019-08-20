import sys
import igraph
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
import scipy.special as sc


class WAADelayed:
    def __init__(self):
        pass

    def init(self, n):
        self.w = np.ones(n) / n
        self.est_D = 1
        self.real_D = 0

    """
    gradient_losses should be n * len(dt)
    """

    def update(self, gradient_losses):
        # check the number of element
        assert gradient_losses.shape[0] == len(self.w)

        for i in range(1, gradient_losses.shape[1] + 1):
            self.real_D += i
            if self.est_D < self.real_D:
                self.est_D *= 2

        lr = np.sqrt(self.est_D * np.log(len(self.w)))

        changes = lr * gradient_losses.sum(axis=1)
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))


class ShortestPathTracker:
    """
        Object tracking based on data association via minimum cost flow algorithm
        L. Zhang et al.,
        "Global data association for multi-object tracking using network flows",
        CVPR 2008
    """

    def __init__(self, cost_link):
        self._cost_link = cost_link

    def initialize(self, detection):
        self.frame_id = -1
        self.edges = []
        if len(self.edges) == 0:
            self.edges.append(("source", str((self.frame_id, 0)), 0))
        self.prev_detections = [detection]

    def track(self, detections, f2i_factor=10000, is_last=False):
        self.frame_id += 1

        prev_frame_id = self.frame_id - 1
        for i, i_info in enumerate(self.prev_detections):
            for j, j_info in enumerate(detections):
                self.edges.append(
                    (
                        str((prev_frame_id, i)),
                        str((self.frame_id, j)),
                        int(self._cost_link(i_info, j_info) * f2i_factor),
                    )
                )
        if is_last:
            for i in range(len(detections)):
                self.last_edges = self.edges[:]
                self.last_edges.append((str((self.frame_id, i)), "sink", 0))
        self.prev_detections = detections

    def run(self):
        g = igraph.Graph.TupleList(self.last_edges, weights=True, directed=True)
        path = g.get_shortest_paths(
            "source", to="sink", weights="weight", mode=igraph.OUT, output="vpath"
        )
        ids = []
        for i in path[0][1:-1]:
            ids.append(tuple(int(s) for s in g.vs[i]["name"].strip("()").split(",")))
        return ids


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
            croped_image = self.transform(image.crop((min_x, min_y, max_x, max_y)))
            croped_images.append(croped_image)
        croped_images = torch.stack(croped_images)
        features = self._get_vector(croped_images)
        features = features.data.cpu().numpy()
        return features

    def _get_vector(self, x):
        x = x.cuda()
        embedding = torch.zeros((x.shape[0], 512))

        def copy_data(m, i, o):
            embedding.copy_(o.data.view(x.shape[0], -1).data)

        h = self.extractor.register_forward_hook(copy_data)
        self.model(x)
        h.remove()

        return embedding
