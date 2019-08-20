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


def cosine_similarity(ft1, ft2):
    return np.dot(ft1, ft2) / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


def calc_distance(rect1, rect2):
    center1 = rect1[:2] + rect1[2:] / 2
    center2 = rect2[:2] + rect2[2:] / 2

    score = 1 / distance.euclidean(center1, center2)
    return score if score < 1 else 1


def calc_cost_link(
    target,
    info1,
    info2,
    check_dist=False,
    check_feature=True,
    check_target=False,
    eps=1e-7,
):
    rect1 = info1["rect"]
    feature1 = info1["feature"]
    rect2 = info2["rect"]
    feature2 = info2["feature"]
    prob_iou = calc_overlap(rect1, rect2)

    if check_dist:
        prob_distance = calc_distance(rect1, rect2)
    else:
        prob_distance = 1

    if check_feature:
        prob_feature = cosine_similarity(feature1, feature2)
    else:
        prob_feature = 1

    if check_target:
        prob_target = cosine_similarity(target["feature"], feature2)
    else:
        prob_target = 1

    prob_sim = prob_iou * prob_feature * prob_target * prob_distance
    return -np.log(prob_sim + eps)
