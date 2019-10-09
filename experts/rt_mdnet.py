import sys
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from .expert import Expert

sys.path.append("external/pytracking/")
from modules.sample_generator import gen_samples, SampleGenerator
from modules.model import BinaryLoss, MDNet
from modules.img_cropper import imgCropper
from modules.roi_align import RoIAlignAdaMax
from modules.utils import samples2maskroi
from modules.bbreg import BBRegressor
from tracker import set_optimizer, train
from options import opts


class RTMDNet(Expert):
    def __init__(self):
        super(RTMDNet, self).__init__("RT-MDNet")
        # option setting
        opts["model_path"] = "rt-mdnet.pth"
        opts["result_path"] = "result.npy"
        opts["visual_log"] = False
        opts["set_type"] = "OTB100"
        opts["visualize"] = False
        opts["adaptive_align"] = True
        opts["padding"] = 1.2
        opts["jitter"] = True

    def initialize(self, image_file, box):
        self.target_bbox = box
        self.model = MDNet(opts["model_path"])
        if opts["adaptive_align"]:
            self.align_h = self.model.roi_align_model.aligned_height
            self.align_w = self.model.roi_align_model.aligned_width
            self.spatial_s = self.model.roi_align_model.spatial_scale
            self.model.roi_align_model = RoIAlignAdaMax(
                self.align_h, self.align_w, self.spatial_s
            )
        if opts["use_gpu"]:
            self.model = self.model.cuda()

        self.model.set_learnable_params(opts["ft_layers"])

        # Init image crop model
        self.img_crop_model = imgCropper(1.0)
        if opts["use_gpu"]:
            self.img_crop_model.gpuEnable()

        # Init criterion and optimizer
        self.criterion = BinaryLoss()
        self.init_optimizer = set_optimizer(self.model, opts["lr_init"])
        self.update_optimizer = set_optimizer(self.model, opts["lr_update"])

        self.cur_image = Image.open(image_file).convert("RGB")
        self.cur_image = np.asarray(self.cur_image)

        # Draw pos/neg samples
        self.ishape = self.cur_image.shape
        self.pos_examples = gen_samples(
            SampleGenerator("gaussian", (self.ishape[1], self.ishape[0]), 0.1, 1.2),
            self.target_bbox,
            opts["n_pos_init"],
            opts["overlap_pos_init"],
        )
        self.neg_examples = gen_samples(
            SampleGenerator("uniform", (self.ishape[1], self.ishape[0]), 1, 2, 1.1),
            self.target_bbox,
            opts["n_neg_init"],
            opts["overlap_neg_init"],
        )
        self.neg_examples = np.random.permutation(self.neg_examples)

        self.cur_bbreg_examples = gen_samples(
            SampleGenerator("uniform", (self.ishape[1], self.ishape[0]), 0.3, 1.5, 1.1),
            self.target_bbox,
            opts["n_bbreg"],
            opts["overlap_bbreg"],
            opts["scale_bbreg"],
        )

        # compute padded sample
        self.padded_x1 = (
            self.neg_examples[:, 0]
            - self.neg_examples[:, 2] * (opts["padding"] - 1.0) / 2.0
        ).min()
        self.padded_y1 = (
            self.neg_examples[:, 1]
            - self.neg_examples[:, 3] * (opts["padding"] - 1.0) / 2.0
        ).min()
        self.padded_x2 = (
            self.neg_examples[:, 0]
            + self.neg_examples[:, 2] * (opts["padding"] + 1.0) / 2.0
        ).max()
        self.padded_y2 = (
            self.neg_examples[:, 1]
            + self.neg_examples[:, 3] * (opts["padding"] + 1.0) / 2.0
        ).max()
        self.padded_scene_box = np.reshape(
            np.asarray(
                (
                    self.padded_x1,
                    self.padded_y1,
                    self.padded_x2 - self.padded_x1,
                    self.padded_y2 - self.padded_y1,
                )
            ),
            (1, 4),
        )

        self.scene_boxes = np.reshape(np.copy(self.padded_scene_box), (1, 4))
        if opts["jitter"]:
            # horizontal shift
            self.jittered_scene_box_horizon = np.copy(self.padded_scene_box)
            self.jittered_scene_box_horizon[0, 0] -= 4.0
            self.jitter_scale_horizon = 1.0

            # vertical shift
            self.jittered_scene_box_vertical = np.copy(self.padded_scene_box)
            self.jittered_scene_box_vertical[0, 1] -= 4.0
            self.jitter_scale_vertical = 1.0

            self.jittered_scene_box_reduce1 = np.copy(self.padded_scene_box)
            self.jitter_scale_reduce1 = 1.1 ** (-1)

            # vertical shift
            self.jittered_scene_box_enlarge1 = np.copy(self.padded_scene_box)
            self.jitter_scale_enlarge1 = 1.1 ** (1)

            # scale reduction
            self.jittered_scene_box_reduce2 = np.copy(self.padded_scene_box)
            self.jitter_scale_reduce2 = 1.1 ** (-2)
            # scale enlarge
            self.jittered_scene_box_enlarge2 = np.copy(self.padded_scene_box)
            self.jitter_scale_enlarge2 = 1.1 ** (2)

            self.scene_boxes = np.concatenate(
                [
                    self.scene_boxes,
                    self.jittered_scene_box_horizon,
                    self.jittered_scene_box_vertical,
                    self.jittered_scene_box_reduce1,
                    self.jittered_scene_box_enlarge1,
                    self.jittered_scene_box_reduce2,
                    self.jittered_scene_box_enlarge2,
                ],
                axis=0,
            )
            self.jitter_scale = [
                1.0,
                self.jitter_scale_horizon,
                self.jitter_scale_vertical,
                self.jitter_scale_reduce1,
                self.jitter_scale_enlarge1,
                self.jitter_scale_reduce2,
                self.jitter_scale_enlarge2,
            ]
        else:
            self.jitter_scale = [1.0]

        self.model.eval()
        for bidx in range(0, self.scene_boxes.shape[0]):
            self.crop_img_size = (
                self.scene_boxes[bidx, 2:4]
                * ((opts["img_size"], opts["img_size"]) / self.target_bbox[2:4])
            ).astype("int64") * self.jitter_scale[bidx]
            self.cropped_image, cur_image_var = self.img_crop_model.crop_image(
                self.cur_image,
                np.reshape(self.scene_boxes[bidx], (1, 4)),
                self.crop_img_size,
            )
            self.cropped_image = self.cropped_image - 128.0

            self.feat_map = self.model(self.cropped_image, out_layer="conv3")

            self.rel_target_bbox = np.copy(self.target_bbox)
            self.rel_target_bbox[0:2] -= self.scene_boxes[bidx, 0:2]

            self.batch_num = np.zeros((self.pos_examples.shape[0], 1))
            self.cur_pos_rois = np.copy(self.pos_examples)
            self.cur_pos_rois[:, 0:2] -= np.repeat(
                np.reshape(self.scene_boxes[bidx, 0:2], (1, 2)),
                self.cur_pos_rois.shape[0],
                axis=0,
            )
            self.scaled_obj_size = float(opts["img_size"]) * self.jitter_scale[bidx]
            self.cur_pos_rois = samples2maskroi(
                self.cur_pos_rois,
                self.model.receptive_field,
                (self.scaled_obj_size, self.scaled_obj_size),
                self.target_bbox[2:4],
                opts["padding"],
            )
            self.cur_pos_rois = np.concatenate(
                (self.batch_num, self.cur_pos_rois), axis=1
            )
            self.cur_pos_rois = Variable(
                torch.from_numpy(self.cur_pos_rois.astype("float32"))
            ).cuda()
            self.cur_pos_feats = self.model.roi_align_model(
                self.feat_map, self.cur_pos_rois
            )
            self.cur_pos_feats = self.cur_pos_feats.view(
                self.cur_pos_feats.size(0), -1
            ).data.clone()

            self.batch_num = np.zeros((self.neg_examples.shape[0], 1))
            self.cur_neg_rois = np.copy(self.neg_examples)
            self.cur_neg_rois[:, 0:2] -= np.repeat(
                np.reshape(self.scene_boxes[bidx, 0:2], (1, 2)),
                self.cur_neg_rois.shape[0],
                axis=0,
            )
            self.cur_neg_rois = samples2maskroi(
                self.cur_neg_rois,
                self.model.receptive_field,
                (self.scaled_obj_size, self.scaled_obj_size),
                self.target_bbox[2:4],
                opts["padding"],
            )
            self.cur_neg_rois = np.concatenate(
                (self.batch_num, self.cur_neg_rois), axis=1
            )
            self.cur_neg_rois = Variable(
                torch.from_numpy(self.cur_neg_rois.astype("float32"))
            ).cuda()
            self.cur_neg_feats = self.model.roi_align_model(
                self.feat_map, self.cur_neg_rois
            )
            self.cur_neg_feats = self.cur_neg_feats.view(
                self.cur_neg_feats.size(0), -1
            ).data.clone()

            # bbreg rois
            self.batch_num = np.zeros((self.cur_bbreg_examples.shape[0], 1))
            self.cur_bbreg_rois = np.copy(self.cur_bbreg_examples)
            self.cur_bbreg_rois[:, 0:2] -= np.repeat(
                np.reshape(self.scene_boxes[bidx, 0:2], (1, 2)),
                self.cur_bbreg_rois.shape[0],
                axis=0,
            )
            self.scaled_obj_size = float(opts["img_size"]) * self.jitter_scale[bidx]
            self.cur_bbreg_rois = samples2maskroi(
                self.cur_bbreg_rois,
                self.model.receptive_field,
                (self.scaled_obj_size, self.scaled_obj_size),
                self.target_bbox[2:4],
                opts["padding"],
            )
            self.cur_bbreg_rois = np.concatenate(
                (self.batch_num, self.cur_bbreg_rois), axis=1
            )
            self.cur_bbreg_rois = Variable(
                torch.from_numpy(self.cur_bbreg_rois.astype("float32"))
            ).cuda()
            self.cur_bbreg_feats = self.model.roi_align_model(
                self.feat_map, self.cur_bbreg_rois
            )
            self.cur_bbreg_feats = self.cur_bbreg_feats.view(
                self.cur_bbreg_feats.size(0), -1
            ).data.clone()

            self.feat_dim = self.cur_pos_feats.size(-1)

            if bidx == 0:
                self.pos_feats = self.cur_pos_feats
                self.neg_feats = self.cur_neg_feats
                # bbreg feature
                self.bbreg_feats = self.cur_bbreg_feats
                self.bbreg_examples = self.cur_bbreg_examples
            else:
                self.pos_feats = torch.cat((self.pos_feats, self.cur_pos_feats), dim=0)
                self.neg_feats = torch.cat((self.neg_feats, self.cur_neg_feats), dim=0)
                # bbreg feature
                self.bbreg_feats = torch.cat(
                    (self.bbreg_feats, self.cur_bbreg_feats), dim=0
                )
                self.bbreg_examples = np.concatenate(
                    (self.bbreg_examples, self.cur_bbreg_examples), axis=0
                )

        if self.pos_feats.size(0) > opts["n_pos_init"]:
            self.pos_idx = np.asarray(range(self.pos_feats.size(0)))
            np.random.shuffle(self.pos_idx)
            self.pos_feats = self.pos_feats[self.pos_idx[0 : opts["n_pos_init"]], :]
        if self.neg_feats.size(0) > opts["n_neg_init"]:
            self.neg_idx = np.asarray(range(self.neg_feats.size(0)))
            np.random.shuffle(self.neg_idx)
            self.neg_feats = self.neg_feats[self.neg_idx[0 : opts["n_neg_init"]], :]

        # bbreg
        if self.bbreg_feats.size(0) > opts["n_bbreg"]:
            self.bbreg_idx = np.asarray(range(self.bbreg_feats.size(0)))
            np.random.shuffle(self.bbreg_idx)
            self.bbreg_feats = self.bbreg_feats[self.bbreg_idx[0 : opts["n_bbreg"]], :]
            self.bbreg_examples = self.bbreg_examples[
                self.bbreg_idx[0 : opts["n_bbreg"]], :
            ]
            # print bbreg_examples.shape

        # open images and crop patch from obj
        self.extra_obj_size = np.array((opts["img_size"], opts["img_size"]))
        self.extra_crop_img_size = self.extra_obj_size * (opts["padding"] + 0.6)
        self.replicateNum = 100
        for iidx in range(self.replicateNum):
            self.extra_target_bbox = np.copy(self.target_bbox)

            self.extra_scene_box = np.copy(self.extra_target_bbox)
            self.extra_scene_box_center = (
                self.extra_scene_box[0:2] + self.extra_scene_box[2:4] / 2.0
            )
            self.extra_scene_box_size = self.extra_scene_box[2:4] * (
                opts["padding"] + 0.6
            )
            self.extra_scene_box[0:2] = (
                self.extra_scene_box_center - self.extra_scene_box_size / 2.0
            )
            self.extra_scene_box[2:4] = self.extra_scene_box_size

            self.extra_shift_offset = np.clip(2.0 * np.random.randn(2), -4, 4)
            self.cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)

            self.extra_scene_box[0] += self.extra_shift_offset[0]
            self.extra_scene_box[1] += self.extra_shift_offset[1]
            self.extra_scene_box[2:4] *= self.cur_extra_scale[0]

            self.scaled_obj_size = float(opts["img_size"]) / self.cur_extra_scale[0]

            self.cur_extra_cropped_image, _ = self.img_crop_model.crop_image(
                self.cur_image,
                np.reshape(self.extra_scene_box, (1, 4)),
                self.extra_crop_img_size,
            )
            self.cur_extra_cropped_image = self.cur_extra_cropped_image.detach()
            # extra_target_bbox = np.array(list(map(int, extra_target_bbox)))
            self.cur_extra_pos_examples = gen_samples(
                SampleGenerator("gaussian", (self.ishape[1], self.ishape[0]), 0.1, 1.2),
                self.extra_target_bbox,
                opts["n_pos_init"] // self.replicateNum,
                opts["overlap_pos_init"],
            )
            self.cur_extra_neg_examples = gen_samples(
                SampleGenerator(
                    "uniform", (self.ishape[1], self.ishape[0]), 0.3, 2, 1.1
                ),
                self.extra_target_bbox,
                opts["n_neg_init"] / self.replicateNum // 4,
                opts["overlap_neg_init"],
            )

            # bbreg sample
            self.cur_extra_bbreg_examples = gen_samples(
                SampleGenerator(
                    "uniform", (self.ishape[1], self.ishape[0]), 0.3, 1.5, 1.1
                ),
                self.extra_target_bbox,
                opts["n_bbreg"] / self.replicateNum // 4,
                opts["overlap_bbreg"],
                opts["scale_bbreg"],
            )

            self.batch_num = iidx * np.ones((self.cur_extra_pos_examples.shape[0], 1))
            self.cur_extra_pos_rois = np.copy(self.cur_extra_pos_examples)
            self.cur_extra_pos_rois[:, 0:2] -= np.repeat(
                np.reshape(self.extra_scene_box[0:2], (1, 2)),
                self.cur_extra_pos_rois.shape[0],
                axis=0,
            )
            self.cur_extra_pos_rois = samples2maskroi(
                self.cur_extra_pos_rois,
                self.model.receptive_field,
                (self.scaled_obj_size, self.scaled_obj_size),
                self.extra_target_bbox[2:4],
                opts["padding"],
            )
            self.cur_extra_pos_rois = np.concatenate(
                (self.batch_num, self.cur_extra_pos_rois), axis=1
            )

            self.batch_num = iidx * np.ones((self.cur_extra_neg_examples.shape[0], 1))
            self.cur_extra_neg_rois = np.copy(self.cur_extra_neg_examples)
            self.cur_extra_neg_rois[:, 0:2] -= np.repeat(
                np.reshape(self.extra_scene_box[0:2], (1, 2)),
                self.cur_extra_neg_rois.shape[0],
                axis=0,
            )
            self.cur_extra_neg_rois = samples2maskroi(
                self.cur_extra_neg_rois,
                self.model.receptive_field,
                (self.scaled_obj_size, self.scaled_obj_size),
                self.extra_target_bbox[2:4],
                opts["padding"],
            )
            self.cur_extra_neg_rois = np.concatenate(
                (self.batch_num, self.cur_extra_neg_rois), axis=1
            )

            # bbreg rois
            self.batch_num = iidx * np.ones((self.cur_extra_bbreg_examples.shape[0], 1))
            self.cur_extra_bbreg_rois = np.copy(self.cur_extra_bbreg_examples)
            self.cur_extra_bbreg_rois[:, 0:2] -= np.repeat(
                np.reshape(self.extra_scene_box[0:2], (1, 2)),
                self.cur_extra_bbreg_rois.shape[0],
                axis=0,
            )
            self.cur_extra_bbreg_rois = samples2maskroi(
                self.cur_extra_bbreg_rois,
                self.model.receptive_field,
                (self.scaled_obj_size, self.scaled_obj_size),
                self.extra_target_bbox[2:4],
                opts["padding"],
            )
            self.cur_extra_bbreg_rois = np.concatenate(
                (self.batch_num, self.cur_extra_bbreg_rois), axis=1
            )

            if iidx == 0:
                self.extra_cropped_image = self.cur_extra_cropped_image

                self.extra_pos_rois = np.copy(self.cur_extra_pos_rois)
                self.extra_neg_rois = np.copy(self.cur_extra_neg_rois)
                # bbreg rois
                self.extra_bbreg_rois = np.copy(self.cur_extra_bbreg_rois)
                self.extra_bbreg_examples = np.copy(self.cur_extra_bbreg_examples)
            else:
                self.extra_cropped_image = torch.cat(
                    (self.extra_cropped_image, self.cur_extra_cropped_image), dim=0
                )

                self.extra_pos_rois = np.concatenate(
                    (self.extra_pos_rois, np.copy(self.cur_extra_pos_rois)), axis=0
                )
                self.extra_neg_rois = np.concatenate(
                    (self.extra_neg_rois, np.copy(self.cur_extra_neg_rois)), axis=0
                )
                # bbreg rois
                self.extra_bbreg_rois = np.concatenate(
                    (self.extra_bbreg_rois, np.copy(self.cur_extra_bbreg_rois)), axis=0
                )
                self.extra_bbreg_examples = np.concatenate(
                    (self.extra_bbreg_examples, np.copy(self.cur_extra_bbreg_examples)),
                    axis=0,
                )

        self.extra_pos_rois = Variable(
            torch.from_numpy(self.extra_pos_rois.astype("float32"))
        ).cuda()
        self.extra_neg_rois = Variable(
            torch.from_numpy(self.extra_neg_rois.astype("float32"))
        ).cuda()
        # bbreg rois
        self.extra_bbreg_rois = Variable(
            torch.from_numpy(self.extra_bbreg_rois.astype("float32"))
        ).cuda()

        self.extra_cropped_image -= 128.0

        self.extra_feat_maps = self.model(self.extra_cropped_image, out_layer="conv3")
        # Draw pos/neg samples
        self.ishape = self.cur_image.shape

        self.extra_pos_feats = self.model.roi_align_model(
            self.extra_feat_maps, self.extra_pos_rois
        )
        self.extra_pos_feats = self.extra_pos_feats.view(
            self.extra_pos_feats.size(0), -1
        ).data.clone()

        self.extra_neg_feats = self.model.roi_align_model(
            self.extra_feat_maps, self.extra_neg_rois
        )
        self.extra_neg_feats = self.extra_neg_feats.view(
            self.extra_neg_feats.size(0), -1
        ).data.clone()
        # bbreg feat
        self.extra_bbreg_feats = self.model.roi_align_model(
            self.extra_feat_maps, self.extra_bbreg_rois
        )
        self.extra_bbreg_feats = self.extra_bbreg_feats.view(
            self.extra_bbreg_feats.size(0), -1
        ).data.clone()

        # concatenate extra features to original_features
        self.pos_feats = torch.cat((self.pos_feats, self.extra_pos_feats), dim=0)
        self.neg_feats = torch.cat((self.neg_feats, self.extra_neg_feats), dim=0)
        # concatenate extra bbreg feats to original_bbreg_feats
        self.bbreg_feats = torch.cat((self.bbreg_feats, self.extra_bbreg_feats), dim=0)
        self.bbreg_examples = np.concatenate(
            (self.bbreg_examples, self.extra_bbreg_examples), axis=0
        )

        torch.cuda.empty_cache()
        self.model.zero_grad()

        # Initial training
        train(
            self.model,
            self.criterion,
            self.init_optimizer,
            self.pos_feats,
            self.neg_feats,
            opts["maxiter_init"],
        )

        # bbreg train
        if self.bbreg_feats.size(0) > opts["n_bbreg"]:
            self.bbreg_idx = np.asarray(range(self.bbreg_feats.size(0)))
            np.random.shuffle(self.bbreg_idx)
            self.bbreg_feats = self.bbreg_feats[self.bbreg_idx[0 : opts["n_bbreg"]], :]
            self.bbreg_examples = self.bbreg_examples[
                self.bbreg_idx[0 : opts["n_bbreg"]], :
            ]
        self.bbreg = BBRegressor((self.ishape[1], self.ishape[0]))
        self.bbreg.train(self.bbreg_feats, self.bbreg_examples, self.target_bbox)

        if self.pos_feats.size(0) > opts["n_pos_update"]:
            self.pos_idx = np.asarray(range(self.pos_feats.size(0)))
            np.random.shuffle(self.pos_idx)
            self.pos_feats_all = [
                self.pos_feats.index_select(
                    0, torch.from_numpy(self.pos_idx[0 : opts["n_pos_update"]]).cuda()
                )
            ]
        if self.neg_feats.size(0) > opts["n_neg_update"]:
            self.neg_idx = np.asarray(range(self.neg_feats.size(0)))
            np.random.shuffle(self.neg_idx)
            self.neg_feats_all = [
                self.neg_feats.index_select(
                    0, torch.from_numpy(self.neg_idx[0 : opts["n_neg_update"]]).cuda()
                )
            ]

        # Main loop
        self.trans_f = opts["trans_f"]
        self.frame = 1

    def track(self, image_file):
        # Load image
        self.cur_image = Image.open(image_file).convert("RGB")
        self.cur_image = np.asarray(self.cur_image)

        # Estimate target bbox
        self.ishape = self.cur_image.shape
        self.samples = gen_samples(
            SampleGenerator(
                "gaussian",
                (self.ishape[1], self.ishape[0]),
                self.trans_f,
                opts["scale_f"],
                valid=True,
            ),
            self.target_bbox,
            opts["n_samples"],
        )

        self.padded_x1 = (
            self.samples[:, 0] - self.samples[:, 2] * (opts["padding"] - 1.0) / 2.0
        ).min()
        self.padded_y1 = (
            self.samples[:, 1] - self.samples[:, 3] * (opts["padding"] - 1.0) / 2.0
        ).min()
        self.padded_x2 = (
            self.samples[:, 0] + self.samples[:, 2] * (opts["padding"] + 1.0) / 2.0
        ).max()
        self.padded_y2 = (
            self.samples[:, 1] + self.samples[:, 3] * (opts["padding"] + 1.0) / 2.0
        ).max()
        self.padded_scene_box = np.asarray(
            (
                self.padded_x1,
                self.padded_y1,
                self.padded_x2 - self.padded_x1,
                self.padded_y2 - self.padded_y1,
            )
        )

        if self.padded_scene_box[0] > self.cur_image.shape[1]:
            self.padded_scene_box[0] = self.cur_image.shape[1] - 1
        if self.padded_scene_box[1] > self.cur_image.shape[0]:
            self.padded_scene_box[1] = self.cur_image.shape[0] - 1
        if self.padded_scene_box[0] + self.padded_scene_box[2] < 0:
            self.padded_scene_box[2] = -self.padded_scene_box[0] + 1
        if self.padded_scene_box[1] + self.padded_scene_box[3] < 0:
            self.padded_scene_box[3] = -self.padded_scene_box[1] + 1

        self.crop_img_size = (
            self.padded_scene_box[2:4]
            * ((opts["img_size"], opts["img_size"]) / self.target_bbox[2:4])
        ).astype("int64")
        self.cropped_image, self.cur_image_var = self.img_crop_model.crop_image(
            self.cur_image,
            np.reshape(self.padded_scene_box, (1, 4)),
            self.crop_img_size,
        )
        cropped_image = self.cropped_image - 128.0

        self.model.eval()
        self.feat_map = self.model(cropped_image, out_layer="conv3")

        # relative target bbox with padded_scene_box
        self.rel_target_bbox = np.copy(self.target_bbox)
        self.rel_target_bbox[0:2] -= self.padded_scene_box[0:2]

        # Extract sample features and get target location
        self.batch_num = np.zeros((self.samples.shape[0], 1))
        self.sample_rois = np.copy(self.samples)
        self.sample_rois[:, 0:2] -= np.repeat(
            np.reshape(self.padded_scene_box[0:2], (1, 2)),
            self.sample_rois.shape[0],
            axis=0,
        )
        self.sample_rois = samples2maskroi(
            self.sample_rois,
            self.model.receptive_field,
            (opts["img_size"], opts["img_size"]),
            self.target_bbox[2:4],
            opts["padding"],
        )
        self.sample_rois = np.concatenate((self.batch_num, self.sample_rois), axis=1)
        self.sample_rois = Variable(
            torch.from_numpy(self.sample_rois.astype("float32"))
        ).cuda()
        self.sample_feats = self.model.roi_align_model(self.feat_map, self.sample_rois)
        self.sample_feats = self.sample_feats.view(
            self.sample_feats.size(0), -1
        ).clone()
        self.sample_scores = self.model(self.sample_feats, in_layer="fc4")
        self.top_scores, self.top_idx = self.sample_scores[:, 1].topk(5)
        self.top_idx = self.top_idx.data.cpu().numpy()
        self.target_score = self.top_scores.data.mean()
        self.target_bbox = self.samples[self.top_idx].mean(axis=0)

        self.success = self.target_score > opts["success_thr"]

        # Expand search area at failure
        if self.success:
            self.trans_f = opts["trans_f"]
        else:
            self.trans_f = opts["trans_f_expand"]

        # Bbox regression
        if self.success:
            self.bbreg_feats = self.sample_feats[self.top_idx, :]
            self.bbreg_samples = self.samples[self.top_idx]
            self.bbreg_samples = self.bbreg.predict(
                self.bbreg_feats.data, self.bbreg_samples
            )
            self.bbreg_bbox = self.bbreg_samples.mean(axis=0)
        else:
            self.bbreg_bbox = self.target_bbox

        # Data collect
        if self.success:

            # Draw pos/neg samples
            self.pos_examples = gen_samples(
                SampleGenerator("gaussian", (self.ishape[1], self.ishape[0]), 0.1, 1.2),
                self.target_bbox,
                opts["n_pos_update"],
                opts["overlap_pos_update"],
            )
            self.neg_examples = gen_samples(
                SampleGenerator("uniform", (self.ishape[1], self.ishape[0]), 1.5, 1.2),
                self.target_bbox,
                opts["n_neg_update"],
                opts["overlap_neg_update"],
            )

            self.padded_x1 = (
                self.neg_examples[:, 0]
                - self.neg_examples[:, 2] * (opts["padding"] - 1.0) / 2.0
            ).min()
            self.padded_y1 = (
                self.neg_examples[:, 1]
                - self.neg_examples[:, 3] * (opts["padding"] - 1.0) / 2.0
            ).min()
            self.padded_x2 = (
                self.neg_examples[:, 0]
                + self.neg_examples[:, 2] * (opts["padding"] + 1.0) / 2.0
            ).max()
            self.padded_y2 = (
                self.neg_examples[:, 1]
                + self.neg_examples[:, 3] * (opts["padding"] + 1.0) / 2.0
            ).max()
            self.padded_scene_box = np.reshape(
                np.asarray(
                    (
                        self.padded_x1,
                        self.padded_y1,
                        self.padded_x2 - self.padded_x1,
                        self.padded_y2 - self.padded_y1,
                    )
                ),
                (1, 4),
            )

            self.scene_boxes = np.reshape(np.copy(self.padded_scene_box), (1, 4))
            self.jitter_scale = [1.0]

            for bidx in range(0, self.scene_boxes.shape[0]):
                self.crop_img_size = (
                    self.scene_boxes[bidx, 2:4]
                    * ((opts["img_size"], opts["img_size"]) / self.target_bbox[2:4])
                ).astype("int64") * self.jitter_scale[bidx]
                self.cropped_image, self.cur_image_var = self.img_crop_model.crop_image(
                    self.cur_image,
                    np.reshape(self.scene_boxes[bidx], (1, 4)),
                    self.crop_img_size,
                )
                self.cropped_image = cropped_image - 128.0

                self.feat_map = self.model(cropped_image, out_layer="conv3")

                self.rel_target_bbox = np.copy(self.target_bbox)
                self.rel_target_bbox[0:2] -= self.scene_boxes[bidx, 0:2]

                self.batch_num = np.zeros((self.pos_examples.shape[0], 1))
                self.cur_pos_rois = np.copy(self.pos_examples)
                self.cur_pos_rois[:, 0:2] -= np.repeat(
                    np.reshape(self.scene_boxes[bidx, 0:2], (1, 2)),
                    self.cur_pos_rois.shape[0],
                    axis=0,
                )
                self.scaled_obj_size = float(opts["img_size"]) * self.jitter_scale[bidx]
                self.cur_pos_rois = samples2maskroi(
                    self.cur_pos_rois,
                    self.model.receptive_field,
                    (self.scaled_obj_size, self.scaled_obj_size),
                    self.target_bbox[2:4],
                    opts["padding"],
                )
                self.cur_pos_rois = np.concatenate(
                    (self.batch_num, self.cur_pos_rois), axis=1
                )
                self.cur_pos_rois = Variable(
                    torch.from_numpy(self.cur_pos_rois.astype("float32"))
                ).cuda()
                self.cur_pos_feats = self.model.roi_align_model(
                    self.feat_map, self.cur_pos_rois
                )
                self.cur_pos_feats = self.cur_pos_feats.view(
                    self.cur_pos_feats.size(0), -1
                ).data.clone()

                self.batch_num = np.zeros((self.neg_examples.shape[0], 1))
                self.cur_neg_rois = np.copy(self.neg_examples)
                self.cur_neg_rois[:, 0:2] -= np.repeat(
                    np.reshape(self.scene_boxes[bidx, 0:2], (1, 2)),
                    self.cur_neg_rois.shape[0],
                    axis=0,
                )
                self.cur_neg_rois = samples2maskroi(
                    self.cur_neg_rois,
                    self.model.receptive_field,
                    (self.scaled_obj_size, self.scaled_obj_size),
                    self.target_bbox[2:4],
                    opts["padding"],
                )
                self.cur_neg_rois = np.concatenate(
                    (self.batch_num, self.cur_neg_rois), axis=1
                )
                self.cur_neg_rois = Variable(
                    torch.from_numpy(self.cur_neg_rois.astype("float32"))
                ).cuda()
                self.cur_neg_feats = self.model.roi_align_model(
                    self.feat_map, self.cur_neg_rois
                )
                self.cur_neg_feats = self.cur_neg_feats.view(
                    self.cur_neg_feats.size(0), -1
                ).data.clone()

                self.feat_dim = self.cur_pos_feats.size(-1)

                if bidx == 0:
                    self.pos_feats = self.cur_pos_feats  # index select
                    self.neg_feats = self.cur_neg_feats
                else:
                    self.pos_feats = torch.cat(
                        (self.pos_feats, self.cur_pos_feats), dim=0
                    )
                    self.neg_feats = torch.cat(
                        (self.neg_feats, self.cur_neg_feats), dim=0
                    )

            if self.pos_feats.size(0) > opts["n_pos_update"]:
                self.pos_idx = np.asarray(range(self.pos_feats.size(0)))
                np.random.shuffle(self.pos_idx)
                self.pos_feats = self.pos_feats.index_select(
                    0, torch.from_numpy(self.pos_idx[0 : opts["n_pos_update"]]).cuda()
                )
            if self.neg_feats.size(0) > opts["n_neg_update"]:
                self.neg_idx = np.asarray(range(self.neg_feats.size(0)))
                np.random.shuffle(self.neg_idx)
                self.neg_feats = self.neg_feats.index_select(
                    0, torch.from_numpy(self.neg_idx[0 : opts["n_neg_update"]]).cuda()
                )

            self.pos_feats_all.append(self.pos_feats)
            self.neg_feats_all.append(self.neg_feats)

            if len(self.pos_feats_all) > opts["n_frames_long"]:
                del self.pos_feats_all[0]
            if len(self.neg_feats_all) > opts["n_frames_short"]:
                del self.neg_feats_all[0]

        # Short term update
        if not self.success:
            self.nframes = min(opts["n_frames_short"], len(self.pos_feats_all))
            self.pos_data = torch.stack(self.pos_feats_all[-self.nframes :], 0).view(
                -1, self.feat_dim
            )
            self.neg_data = torch.stack(self.neg_feats_all, 0).view(-1, self.feat_dim)
            train(
                self.model,
                self.criterion,
                self.update_optimizer,
                self.pos_data,
                self.neg_data,
                opts["maxiter_update"],
            )

        # Long term update
        elif self.frame % opts["long_interval"] == 0:
            self.pos_data = torch.stack(self.pos_feats_all, 0).view(-1, self.feat_dim)
            self.neg_data = torch.stack(self.neg_feats_all, 0).view(-1, self.feat_dim)
            train(
                self.model,
                self.criterion,
                self.update_optimizer,
                self.pos_data,
                self.neg_data,
                opts["maxiter_update"],
            )
        self.frame += 1

        return self.bbreg_bbox
