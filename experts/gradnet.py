import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["C_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import cv2
import numpy as np
import path_config
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from base_tracker import BaseTracker

sys.path.append("external/GradNet-Tensorflow")
from parameters import configParams
from track import (
    getOpts,
    createLabels,
    makeScalePyramid,
    getSubWinTracking,
    trackerEval,
)
from siamese import SiameseNet
from region_to_bbox import region_to_bbox


class GradNet(BaseTracker):
    def __init__(self):
        super(GradNet, self).__init__("GradNet")
        self.opts = configParams()
        self.opts = getOpts(self.opts)

        """define input tensors and network"""
        self.exemplarOp_init = tf.placeholder(
            tf.float32, [1, self.opts["exemplarSize"], self.opts["exemplarSize"], 3]
        )
        self.instanceOp_init = tf.placeholder(
            tf.float32, [1, self.opts["instanceSize"], self.opts["instanceSize"], 3]
        )
        self.instanceOp = tf.placeholder(
            tf.float32, [3, self.opts["instanceSize"], self.opts["instanceSize"], 3]
        )
        self.template_Op = tf.placeholder(tf.float32, [1, 6, 6, 256])
        self.search_tr_Op = tf.placeholder(tf.float32, [3, 22, 22, 32])
        self.isTrainingOp = tf.convert_to_tensor(
            False, dtype="bool", name="is_training"
        )
        self.lr = tf.constant(0.0001, dtype="float32")
        self.sn = SiameseNet()

        """build the model"""
        # initial embedding
        with tf.variable_scope("siamese") as scope:
            self.zFeat2Op_init, self.zFeat5Op_init = self.sn.extract_gra_fea_template(
                self.exemplarOp_init, self.opts, self.isTrainingOp
            )
            self.scoreOp_init = self.sn.response_map_cal(
                self.instanceOp_init, self.zFeat5Op_init, self.opts, self.isTrainingOp
            )
        # gradient calculation
        self.labels = np.ones([8], dtype=np.float32)
        self.respSz = int(self.scoreOp_init.get_shape()[1])
        self.respSz = [self.respSz, self.respSz]
        self.respStride = 8
        self.fixedLabel, self.instanceWeight = createLabels(
            self.respSz,
            self.opts["lossRPos"] / self.respStride,
            self.opts["lossRNeg"] / self.respStride,
            1,
        )
        self.instanceWeightOp = tf.constant(self.instanceWeight, dtype=tf.float32)
        self.yOp = tf.constant(self.fixedLabel, dtype=tf.float32)
        with tf.name_scope("logistic_loss"):
            self.lossOp_init = self.sn.loss(
                self.scoreOp_init, self.yOp, self.instanceWeightOp
            )
        self.grad_init = tf.gradients(self.lossOp_init, self.zFeat2Op_init)
        # template update and get score map
        with tf.variable_scope("siamese") as scope:
            self.zFeat5Op_gra, self.zFeat2Op_gra = self.sn.template_update_based_grad(
                self.zFeat2Op_init, self.grad_init[0], self.opts, self.isTrainingOp
            )
            scope.reuse_variables()
            self.zFeat5Op_sia = self.sn.extract_sia_fea_template(
                self.exemplarOp_init, self.opts, self.isTrainingOp
            )
            self.scoreOp_sia = self.sn.response_map_cal(
                self.instanceOp, self.zFeat5Op_sia, self.opts, self.isTrainingOp
            )
            self.scoreOp_gra = self.sn.response_map_cal(
                tf.expand_dims(self.instanceOp[1], 0),
                self.zFeat5Op_gra,
                self.opts,
                self.isTrainingOp,
            )

        """restore pretrained network"""
        self.saver = tf.train.Saver()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.saver.restore(
            self.sess,
            path_config.GRADNET_MODEL,
        )

    def initialize(self, image_file, box):
        im = cv2.imread(image_file, cv2.IMREAD_COLOR)
        cx, cy, w, h = region_to_bbox(box)
        self.targetSize = np.array([h, w])
        self.targetPosition = np.array([cy, cx])
        self.avgChans = np.mean(
            im, axis=(0, 1)
        )  # [np.mean(np.mean(img[:, :, 0])), np.mean(np.mean(img[:, :, 1])), np.mean(np.mean(img[:, :, 2]))]
        self.wcz = self.targetSize[1] + self.opts["contextAmount"] * np.sum(
            self.targetSize
        )
        self.hcz = self.targetSize[0] + self.opts["contextAmount"] * np.sum(
            self.targetSize
        )
        self.sz = np.sqrt(self.wcz * self.hcz)
        self.scalez = self.opts["exemplarSize"] / self.sz

        self.zCrop, _ = getSubWinTracking(
            im,
            self.targetPosition,
            (self.opts["exemplarSize"], self.opts["exemplarSize"]),
            (np.around(self.sz), np.around(self.sz)),
            self.avgChans,
        )

        if self.opts["subMean"]:
            pass

        self.dSearch = (self.opts["instanceSize"] - self.opts["exemplarSize"]) / 2
        self.pad = self.dSearch / self.scalez
        self.sx = self.sz + 2 * self.pad

        self.minSx = 0.2 * self.sx
        self.maxSx = 5.0 * self.sx

        self.winSz = self.opts["scoreSize"] * self.opts["responseUp"]
        if self.opts["windowing"] == "cosine":
            self.hann = np.hanning(self.winSz).reshape(self.winSz, 1)
            self.window = self.hann.dot(self.hann.T)
        elif self.opts["windowing"] == "uniform":
            self.window = np.ones((self.winSz, self.winSz), dtype=np.float32)

        self.window = self.window / np.sum(self.window)
        self.scales = np.array(
            [
                self.opts["scaleStep"] ** i
                for i in range(
                    int(np.ceil(self.opts["numScale"] / 2.0) - self.opts["numScale"]),
                    int(np.floor(self.opts["numScale"] / 2.0) + 1),
                )
            ]
        )

        """initialization at the first frame"""
        self.xCrops = makeScalePyramid(
            im,
            self.targetPosition,
            self.sx * self.scales,
            self.opts["instanceSize"],
            self.avgChans,
            None,
            self.opts,
        )
        self.xCrops0 = np.expand_dims(self.xCrops[1], 0)
        self.zCrop = np.expand_dims(self.zCrop, axis=0)
        self.zCrop0 = np.copy(self.zCrop)

        self.zFeat5_gra_init, self.zFeat2_gra_init, self.zFeat5_sia_init = self.sess.run(
            [self.zFeat5Op_gra, self.zFeat2Op_gra, self.zFeat5Op_sia],
            feed_dict={
                self.exemplarOp_init: self.zCrop0,
                self.instanceOp_init: self.xCrops0,
                self.instanceOp: self.xCrops,
            },
        )
        self.template_gra = np.copy(self.zFeat5_gra_init)
        self.template_sia = np.copy(self.zFeat5_sia_init)
        self.hid_gra = np.copy(self.zFeat2_gra_init)

        self.train_all = []
        self.frame_all = []
        self.F_max_all = 0
        self.A_all = []
        self.F_max_thred = 0
        self.F_max = 0
        self.train_all.append(self.xCrops0)
        self.A_all.append(0)
        self.frame_all.append(0)
        self.updata_features = []
        self.updata_features_score = []
        self.updata_features_frame = []
        self.no_cos = 1
        self.refind = 0

        self.frame = 0

        """tracking results"""
        self.rectPosition = self.targetPosition - self.targetSize / 2.0
        self.Position_now = np.concatenate(
            [
                np.round(self.rectPosition).astype(int)[::-1],
                np.round(self.targetSize).astype(int)[::-1],
            ],
            0,
        )

        if (
            self.Position_now[0] + self.Position_now[2] > im.shape[1]
            and self.F_max < self.F_max_thred * 0.5
        ):
            self.refind = 1

        """if you want use groundtruth"""

        # region = np.copy(gt[i])

        # cx, cy, w, h = getAxisAlignedBB(region)
        # pos = np.array([cy, cx])
        # targetSz = np.array([h, w])
        # iou_ = _compute_distance(region, Position_now)
        #

        """save the reliable training sample"""
        if self.F_max >= min(
            self.F_max_thred * 0.5, np.mean(self.updata_features_score)
        ):
            self.scaledInstance = self.sx * self.scales
            self.xCrops = makeScalePyramid(
                im,
                self.targetPosition,
                self.scaledInstance,
                self.opts["instanceSize"],
                self.avgChans,
                None,
                self.opts,
            )
            self.updata_features.append(self.xCrops)
            self.updata_features_score.append(self.F_max)
            self.updata_features_frame.append(self.frame)
            if self.updata_features_score.__len__() > 5:
                del self.updata_features_score[0]
                del self.updata_features[0]
                del self.updata_features_frame[0]
        else:
            if self.frame < 10 and self.F_max < self.F_max_thred * 0.4:
                self.scaledInstance = self.sx * self.scales
                self.xCrops = makeScalePyramid(
                    im,
                    self.targetPosition,
                    self.scaledInstance,
                    self.opts["instanceSize"],
                    self.avgChans,
                    None,
                    self.opts,
                )
                self.template_gra, self.zFeat2_gra = self.sess.run(
                    [self.zFeat5Op_gra, self.zFeat2Op_gra],
                    feed_dict={
                        self.zFeat2Op_init: self.hid_gra,
                        self.instanceOp_init: np.expand_dims(self.xCrops[1], 0),
                    },
                )
                self.hid_gra = np.copy(0.3 * self.hid_gra + 0.7 * self.zFeat2_gra)

        """update the template every 5 frames"""

        if self.frame % 5 == 0:
            self.template_gra, self.zFeat2_gra = self.sess.run(
                [self.zFeat5Op_gra, self.zFeat2Op_gra],
                feed_dict={
                    self.zFeat2Op_init: self.hid_gra,
                    self.instanceOp_init: np.expand_dims(
                        self.updata_features[np.argmax(self.updata_features_score)][1],
                        0,
                    ),
                },
            )
            self.hid_gra = np.copy(0.4 * self.hid_gra + 0.6 * self.zFeat2_gra)

    def track(self, image_file):
        im = cv2.imread(image_file, cv2.IMREAD_COLOR)
        self.frame += 1

        if self.frame - self.updata_features_frame[-1] == 9 and self.no_cos:
            self.opts["wInfluence"] = 0
            self.no_cos = 0
        else:
            self.opts["wInfluence"] = 0.25

        if im.shape[-1] == 1:
            tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
            tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
            im = tmp

        self.scaledInstance = self.sx * self.scales
        self.scaledTarget = np.array([self.targetSize * scale for scale in self.scales])

        self.xCrops = makeScalePyramid(
            im,
            self.targetPosition,
            self.scaledInstance,
            self.opts["instanceSize"],
            self.avgChans,
            None,
            self.opts,
        )

        self.score_gra, self.score_sia = self.sess.run(
            [self.scoreOp_gra, self.scoreOp_sia],
            feed_dict={
                self.zFeat5Op_gra: self.template_gra,
                self.zFeat5Op_sia: self.template_sia,
                self.instanceOp: self.xCrops,
            },
        )
        # sio.savemat('score.mat', {'score': score})
        # score_gra = np.copy(np.expand_dims(score_sia[1],0))

        self.newTargetPosition, self.newScale = trackerEval(
            self.score_sia,
            self.score_gra,
            round(self.sx),
            self.targetPosition,
            self.window,
            self.opts,
        )

        self.targetPosition = self.newTargetPosition
        self.sx = max(
            self.minSx,
            min(
                self.maxSx,
                (1 - self.opts["scaleLr"]) * self.sx
                + self.opts["scaleLr"] * self.scaledInstance[self.newScale],
            ),
        )
        self.F_max = np.max(self.score_sia)
        self.targetSize = (1 - self.opts["scaleLr"]) * self.targetSize + self.opts[
            "scaleLr"
        ] * self.scaledTarget[self.newScale]
        # print('frame:%d--loss:%f--frame_now:%d' %(i, np.max(score),frame_now))

        if self.refind:

            self.xCrops = makeScalePyramid(
                im,
                np.array([im.shape[0] / 2, im.shape[1] / 2]),
                self.scaledInstance,
                self.opts["instanceSize"],
                self.avgChans,
                None,
                self.opts,
            )

            self.score_gra, self.score_sia = self.sess.run(
                [self.scoreOp_gra, self.scoreOp_sia],
                feed_dict={
                    self.zFeat5Op_gra: self.template_gra,
                    self.zFeat5Op_sia: self.template_sia,
                    self.instanceOp: self.xCrops,
                },
            )
            self.F_max2 = np.max(self.score_sia)
            self.F_max3 = np.max(self.score_gra)
            if self.F_max2 > self.F_max and self.F_max3 > self.F_max:
                self.newTargetPosition, self.newScale = trackerEval(
                    self.score_sia,
                    self.score_gra,
                    round(self.sx),
                    np.array([im.shape[0] / 2, im.shape[1] / 2]),
                    self.window,
                    self.opts,
                )

                self.targetPosition = self.newTargetPosition
                self.sx = max(
                    self.minSx,
                    min(
                        self.maxSx,
                        (1 - self.opts["scaleLr"]) * self.sx
                        + self.opts["scaleLr"] * self.scaledInstance[self.newScale],
                    ),
                )
                self.F_max = np.max(self.score_sia)
                self.targetSize = (
                    (1 - self.opts["scaleLr"]) * self.targetSize
                    + self.opts["scaleLr"] * self.scaledTarget[self.newScale]
                )

            self.refind = 0

        """use the average of the first five frames to set the threshold"""
        if self.frame < 6:
            self.F_max_all = self.F_max_all + self.F_max
        if self.frame == 5:
            self.F_max_thred = self.F_max_all / 5.0

        """tracking results"""
        self.rectPosition = self.targetPosition - self.targetSize / 2.0
        self.Position_now = np.concatenate(
            [
                np.round(self.rectPosition).astype(int)[::-1],
                np.round(self.targetSize).astype(int)[::-1],
            ],
            0,
        )
        bbox = self.Position_now[:]

        if (
            self.Position_now[0] + self.Position_now[2] > im.shape[1]
            and self.F_max < self.F_max_thred * 0.5
        ):
            self.refind = 1

        """if you want use groundtruth"""

        # region = np.copy(gt[i])

        # cx, cy, w, h = getAxisAlignedBB(region)
        # pos = np.array([cy, cx])
        # targetSz = np.array([h, w])
        # iou_ = _compute_distance(region, Position_now)
        #

        """save the reliable training sample"""
        if self.F_max >= min(
            self.F_max_thred * 0.5, np.mean(self.updata_features_score)
        ):
            self.scaledInstance = self.sx * self.scales
            self.xCrops = makeScalePyramid(
                im,
                self.targetPosition,
                self.scaledInstance,
                self.opts["instanceSize"],
                self.avgChans,
                None,
                self.opts,
            )
            self.updata_features.append(self.xCrops)
            self.updata_features_score.append(self.F_max)
            self.updata_features_frame.append(self.frame)
            if self.updata_features_score.__len__() > 5:
                del self.updata_features_score[0]
                del self.updata_features[0]
                del self.updata_features_frame[0]
        else:
            if self.frame < 10 and self.F_max < self.F_max_thred * 0.4:
                self.scaledInstance = self.sx * self.scales
                self.xCrops = makeScalePyramid(
                    im,
                    self.targetPosition,
                    self.scaledInstance,
                    self.opts["instanceSize"],
                    self.avgChans,
                    None,
                    self.opts,
                )
                self.template_gra, self.zFeat2_gra = self.sess.run(
                    [self.zFeat5Op_gra, self.zFeat2Op_gra],
                    feed_dict={
                        self.zFeat2Op_init: self.hid_gra,
                        self.instanceOp_init: np.expand_dims(self.xCrops[1], 0),
                    },
                )
                self.hid_gra = np.copy(0.3 * self.hid_gra + 0.7 * self.zFeat2_gra)

        """update the template every 5 frames"""

        if self.frame % 5 == 0:
            self.template_gra, self.zFeat2_gra = self.sess.run(
                [self.zFeat5Op_gra, self.zFeat2Op_gra],
                feed_dict={
                    self.zFeat2Op_init: self.hid_gra,
                    self.instanceOp_init: np.expand_dims(
                        self.updata_features[np.argmax(self.updata_features_score)][1],
                        0,
                    ),
                },
            )
            self.hid_gra = np.copy(0.4 * self.hid_gra + 0.6 * self.zFeat2_gra)

        return bbox
