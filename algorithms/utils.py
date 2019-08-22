import numpy as np
from scipy.spatial import distance


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


def iou_score(boxes):
    scores = []
    for i, box1 in enumerate(boxes):
        score = []
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            score.append(calc_overlap(box1, box2))
        score = np.mean(score)
        scores.append(score)
    return scores


def cosine_similarity(ft1, ft2):
    return np.dot(ft1, ft2) / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


def calc_distance(rect1, rect2):
    center1 = rect1[:2] + rect1[2:] / 2
    center2 = rect2[:2] + rect2[2:] / 2

    score = 1 / (distance.euclidean(center1, center2) + 1e-7)
    return score if score < 1.0 else 1.0
