import sys
from PIL import Image
import numpy as np
import yaml
import torch
from base_tracker import BaseTracker

sys.path.append("external/py_Vital")
sys.path.append("external/py_Vital/gnet")
sys.path.append("external/py_Vital/tracking")
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from bbreg import BBRegressor
from g_init import NetG
from g_pretrain import g_pretrain
from run_tracker import forward_samples, train


class Vital(BaseTracker):
    def __init__(self):
        super(Vital, self).__init__("Vital")
        # TODO: edit this path and edit the file
        self.opts = yaml.safe_load(
            open(
                "/home/heonsong/Desktop/AAA/AAA-journal/external/py_Vital/tracking/options.yaml",
                "r",
            )
        )

    def initialize(self, image_file, box):
        self.frame = 0
        image = Image.open(image_file).convert("RGB")

        # Init bbox
        self.target_bbox = np.array(box)
        self.before_target = self.target_bbox

        # Init model
        self.model = MDNet(self.opts["model_path"])
        self.model_g = NetG()
        self.model = self.model.cuda()
        self.model_g = self.model_g.cuda()

        # Init criterion and optimizer
        self.criterion = BCELoss()
        self.criterion_g = torch.nn.MSELoss(reduction="mean")
        self.model.set_learnable_params(self.opts["ft_layers"])
        self.model_g.set_learnable_params(self.opts["ft_layers"])
        self.init_optimizer = set_optimizer(
            self.model, self.opts["lr_init"], self.opts["lr_mult"]
        )
        self.update_optimizer = set_optimizer(
            self.model, self.opts["lr_update"], self.opts["lr_mult"]
        )

        # Draw pos/neg samples
        self.pos_examples = SampleGenerator(
            "gaussian", image.size, self.opts["trans_pos"], self.opts["scale_pos"]
        )(self.target_bbox, self.opts["n_pos_init"], self.opts["overlap_pos_init"])

        self.neg_examples = np.concatenate(
            [
                SampleGenerator(
                    "uniform",
                    image.size,
                    self.opts["trans_neg_init"],
                    self.opts["scale_neg_init"],
                )(
                    self.target_bbox,
                    int(self.opts["n_neg_init"] * 0.5),
                    self.opts["overlap_neg_init"],
                ),
                SampleGenerator("whole", image.size)(
                    self.target_bbox,
                    int(self.opts["n_neg_init"] * 0.5),
                    self.opts["overlap_neg_init"],
                ),
            ]
        )
        self.neg_examples = np.random.permutation(self.neg_examples)

        # Extract pos/neg features
        self.pos_feats = forward_samples(self.model, image, self.pos_examples)
        self.neg_feats = forward_samples(self.model, image, self.neg_examples)

        # Initial training
        train(
            self.model,
            None,
            self.criterion,
            self.init_optimizer,
            self.pos_feats,
            self.neg_feats,
            self.opts["maxiter_init"],
        )
        del self.init_optimizer, self.neg_feats
        torch.cuda.empty_cache()
        g_pretrain(self.model, self.model_g, self.criterion_g, self.pos_feats)
        torch.cuda.empty_cache()

        # Train bbox regressor
        self.bbreg_examples = SampleGenerator(
            "uniform",
            image.size,
            self.opts["trans_bbreg"],
            self.opts["scale_bbreg"],
            self.opts["aspect_bbreg"],
        )(self.target_bbox, self.opts["n_bbreg"], self.opts["overlap_bbreg"])
        self.bbreg_feats = forward_samples(self.model, image, self.bbreg_examples)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(self.bbreg_feats, self.bbreg_examples, self.target_bbox)
        del self.bbreg_feats
        torch.cuda.empty_cache()

        # Init sample generators for update
        self.sample_generator = SampleGenerator(
            "gaussian", image.size, self.opts["trans"], self.opts["scale"]
        )
        self.pos_generator = SampleGenerator(
            "gaussian", image.size, self.opts["trans_pos"], self.opts["scale_pos"]
        )
        self.neg_generator = SampleGenerator(
            "uniform", image.size, self.opts["trans_neg"], self.opts["scale_neg"]
        )

        # Init pos/neg features for update
        self.neg_examples = self.neg_generator(
            self.target_bbox, self.opts["n_neg_update"], self.opts["overlap_neg_init"]
        )
        self.neg_feats = forward_samples(self.model, image, self.neg_examples)
        self.pos_feats_all = [self.pos_feats]
        self.neg_feats_all = [self.neg_feats]

    def track(self, image_file):
        self.frame += 1
        image = Image.open(image_file).convert("RGB")

        # Estimate target bbox
        self.samples = self.sample_generator(self.target_bbox, self.opts["n_samples"])
        self.sample_scores = forward_samples(
            self.model, image, self.samples, out_layer="fc6"
        )

        self.top_scores, self.top_idx = self.sample_scores[:, 1].topk(5)
        self.top_idx = self.top_idx.cpu()
        self.target_score = self.top_scores.mean()
        self.target_bbox = self.samples[self.top_idx]
        if self.top_idx.shape[0] > 1:
            self.target_bbox = self.target_bbox.mean(axis=0)
        self.success = self.target_score > 0

        # Expand search area at failure
        if self.success:
            self.sample_generator.set_trans(self.opts["trans"])
        else:
            self.sample_generator.expand_trans(self.opts["trans_limit"])

        # Bbox regression
        if self.success:
            self.bbreg_samples = self.samples[self.top_idx]
            if self.top_idx.shape[0] == 1:
                self.bbreg_samples = self.bbreg_samples[None, :]
            self.bbreg_feats = forward_samples(self.model, image, self.bbreg_samples)
            self.bbreg_samples = self.bbreg.predict(
                self.bbreg_feats, self.bbreg_samples
            )
            self.bbreg_bbox = self.bbreg_samples.mean(axis=0)
        else:
            self.bbreg_bbox = self.target_bbox

        # Data collect
        if self.success:
            self.pos_examples = self.pos_generator(
                self.target_bbox,
                self.opts["n_pos_update"],
                self.opts["overlap_pos_update"],
            )
            self.pos_feats = forward_samples(self.model, image, self.pos_examples)
            self.pos_feats_all.append(self.pos_feats)
            if len(self.pos_feats_all) > self.opts["n_frames_long"]:
                del self.pos_feats_all[0]

            self.neg_examples = self.neg_generator(
                self.target_bbox,
                self.opts["n_neg_update"],
                self.opts["overlap_neg_update"],
            )
            self.neg_feats = forward_samples(self.model, image, self.neg_examples)
            self.neg_feats_all.append(self.neg_feats)
            if len(self.neg_feats_all) > self.opts["n_frames_short"]:
                del self.neg_feats_all[0]

        # Short term update
        if not self.success:
            self.nframes = min(self.opts["n_frames_short"], len(self.pos_feats_all))
            self.pos_data = torch.cat(self.pos_feats_all[-self.nframes :], 0)
            self.neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                None,
                self.criterion,
                self.update_optimizer,
                self.pos_data,
                self.neg_data,
                self.opts["maxiter_update"],
            )

        # Long term update
        elif self.frame % self.opts["long_interval"] == 0:
            self.pos_data = torch.cat(self.pos_feats_all, 0)
            self.neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                self.model_g,
                self.criterion,
                self.update_optimizer,
                self.pos_data,
                self.neg_data,
                self.opts["maxiter_update"],
            )

        torch.cuda.empty_cache()

        return self.bbreg_bbox
