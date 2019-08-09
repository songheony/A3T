import numpy as np
import cv2
import torch
from .expert import Expert
from external.TADT_python.defaults import _C as cfg
from external.TADT_python.tadt_tracker import Tadt_Tracker, cal_srch_window_location
from external.TADT_python.backbone_v2 import build_vgg16
from external.TADT_python.feature_utils_v2 import (
    get_subwindow_feature,
    generate_patch_feature,
    round_python2,
    features_selection,
    resize_tensor,
)
from external.TADT_python.tracking_utils import (
    calculate_scale,
    generate_2d_window,
    cal_window_size,
)
from external.TADT_python.taf import taf_model


class TADT(Expert):
    def __init__(self):
        super(TADT, self).__init__("TADT")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self, image, box):
        self.model = build_vgg16(cfg)
        self.tracker = Tadt_Tracker(
            cfg, model=self.model, device=self.device, display=False
        )

        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.target_location = box
        origin_target_size = np.sqrt(self.target_location[2] * self.target_location[3])
        origin_image_size = img.shape[0:2][::-1]  # [width,height]
        if origin_target_size > self.config.MODEL.MAX_SIZE:
            self.rescale = self.config.MODEL.MAX_SIZE / origin_target_size
        elif origin_target_size < self.config.MODEL.MIN_SIZE:
            self.rescale = self.config.MODEL.MIN_SIZE / origin_target_size

        # ----------------scale image cv2 numpy.adarray---------------
        image = cv2.resize(
            img,
            tuple((np.ceil(np.array(origin_image_size) * self.rescale)).astype(int)),
            interpolation=cv2.INTER_LINEAR,
        )

        # ------scaled target location, get position and size [x1,y1,width,height]------
        self.target_location = round_python2(
            np.array(self.target_location) * self.rescale
        ) - np.array(
            [1, 1, 0, 0]
        )  # 0-index
        target_size = self.target_location[2:4]  # [width, height]
        image_size = image.shape[0:2]  # [height, width]
        search_size, ratio = cal_window_size(
            self.config.MODEL.MAX_SIZE,
            image_size,
            self.config.MODEL.SCALE_NUM,
            self.config.MODEL.TOTAL_STRIDE,
        )
        self.input_size = np.array([search_size, search_size])

        # ------------First frame processing--------------------
        self.srch_window_location = cal_srch_window_location(
            self.target_location, search_size
        )
        features = get_subwindow_feature(
            self.model,
            image,
            self.srch_window_location,
            self.input_size,
            visualize=False,
        )
        # ----------- crop the target exemplar from the feature map------------------
        patch_features, patch_locations = generate_patch_feature(
            target_size[::-1], self.srch_window_location, features
        )
        self.feature_pad = 2
        self.b_feature_pad = int(self.feature_pad / 2)
        self.filter_sizes = [
            torch.tensor(feature.shape).numpy() for feature in patch_features
        ]
        # -------------compute the indecis of target-aware features----------------
        self.feature_weights, self.balance_weights = taf_model(
            features, self.filter_sizes, self.device
        )
        # -------------select the target-awares features---------------------------
        self.exemplar_features = features_selection(
            patch_features, self.feature_weights, self.balance_weights, mode="reduction"
        )
        # self.exemplar_features = fuse_feature(patch_features)

    def update(self, image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(
            img,
            tuple((np.ceil(np.array(img.shape[0:2][::-1]) * self.rescale)).astype(int)),
            interpolation=cv2.INTER_LINEAR,
        )
        # -------------get multi-scale feature--------------------------------------
        features = get_subwindow_feature(
            self.model,
            image,
            self.srch_window_location,
            self.input_size,
            visualize=False,
        )
        feature_size = (torch.tensor(features[0].shape)).numpy().astype(int)[-2:]
        # selected_features = fuse_feature(features)
        selected_features = features_selection(
            features, self.feature_weights, self.balance_weights, mode="reduction"
        )
        selected_features_1 = resize_tensor(
            selected_features, tuple(feature_size + self.feature_pad)
        )
        selected_features_3 = resize_tensor(
            selected_features, tuple(feature_size - self.feature_pad)
        )
        selected_features_1 = selected_features_1[
            :,
            :,
            self.b_feature_pad : feature_size[0] + self.b_feature_pad,
            self.b_feature_pad : feature_size[1] + self.b_feature_pad,
        ]

        selected_features_3 = torch.nn.functional.pad(
            selected_features_3,
            (
                self.b_feature_pad,
                self.b_feature_pad,
                self.b_feature_pad,
                self.b_feature_pad,
            ),
        )
        scaled_features = torch.cat(
            (selected_features_1, selected_features, selected_features_3), dim=0
        )

        # -------------get response map-----------------------------------------------
        response_map = self.siamese_model(scaled_features, self.exemplar_features).to(
            "cpu"
        )
        scaled_response_map = torch.squeeze(
            resize_tensor(
                response_map,
                tuple(self.srch_window_location[-2:].astype(int)),
                mode="bicubic",
                align_corners=True,
            )
        )
        hann_window = generate_2d_window(
            "hann",
            tuple(self.srch_window_location[-2:].astype(int)),
            scaled_response_map.shape[0],
        )
        scaled_response_maps = scaled_response_map + hann_window

        # -------------find max-response----------------------------------------------
        scale_ind = calculate_scale(
            scaled_response_maps, self.config.MODEL.SCALE_WEIGHTS
        )
        response_map = scaled_response_maps[scale_ind, :, :].numpy()
        max_h, max_w = np.where(response_map == np.max(response_map))
        if len(max_h) > 1:
            max_h = np.array([max_h[0]])
        if len(max_w) > 1:
            max_w = np.array([max_w[0]])

        # -------------update tracking state and save tracking result----------------------------------------
        target_loc_center = np.append(
            self.target_location[0:2] + (self.target_location[2:4]) / 2,
            self.target_location[2:4],
        )
        target_loc_center[0:2] = (
            target_loc_center[0:2]
            + (np.append(max_w, max_h) - (self.srch_window_location[2:4] / 2 - 1))
            * self.config.MODEL.SCALES[scale_ind]
        )
        target_loc_center[2:4] = (
            target_loc_center[2:4] * self.config.MODEL.SCALES[scale_ind]
        )
        # print('target_loc_center in current frame:',target_loc_center)
        self.target_location = np.append(
            target_loc_center[0:2] - (target_loc_center[2:4]) / 2,
            target_loc_center[2:4],
        )
        # print('target_location in current frame:', target_location)

        self.srch_window_location[2:4] = round_python2(
            self.srch_window_location[2:4] * self.config.MODEL.SCALES[scale_ind]
        )
        self.srch_window_location[0:2] = (
            target_loc_center[0:2] - (self.srch_window_location[2:4]) / 2
        )

        tracking_bbox = (
            self.target_location + np.array([1, 1, 0, 0])
        ) / self.rescale - np.array(
            [1, 1, 0, 0]
        )  # tracking_bbox: 0-index
        return tracking_bbox
