import sys
from PIL import Image
import numpy as np
import yaml
import torch
from .expert import Expert

sys.path.append("external/py_Vital")
sys.path.append("external/py_Vital/gnet")
sys.path.append("external/py_Vital/tracking")
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from bbreg import BBRegressor
from g_init import NetG
from g_pretrain import g_pretrain
from run_tracker import forward_samples, train


class Vital(Expert):
    def __init__(self):
        super(Vital, self).__init__("Vital")
        # TODO: edit this path and edit the file
        self.opts = yaml.safe_load(
            open(
                "/home/heonsong/Desktop/AAA/AAA-journal/external/py_Vital/tracking/options.yaml",
                "r",
            )
        )

    def initialize(self, image, box):
        self.frame = 0
        image = Image.fromarray(image)

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
        init_optimizer = set_optimizer(
            self.model, self.opts["lr_init"], self.opts["lr_mult"]
        )
        self.update_optimizer = set_optimizer(
            self.model, self.opts["lr_update"], self.opts["lr_mult"]
        )

        # Draw pos/neg samples
        pos_examples = SampleGenerator(
            "gaussian", image.size, self.opts["trans_pos"], self.opts["scale_pos"]
        )(self.target_bbox, self.opts["n_pos_init"], self.opts["overlap_pos_init"])

        neg_examples = np.concatenate(
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
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.model, image, pos_examples)
        neg_feats = forward_samples(self.model, image, neg_examples)

        # Initial training
        train(
            self.model,
            None,
            self.criterion,
            init_optimizer,
            pos_feats,
            neg_feats,
            self.opts["maxiter_init"],
        )
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()
        g_pretrain(self.model, self.model_g, self.criterion_g, pos_feats)
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator(
            "uniform",
            image.size,
            self.opts["trans_bbreg"],
            self.opts["scale_bbreg"],
            self.opts["aspect_bbreg"],
        )(self.target_bbox, self.opts["n_bbreg"], self.opts["overlap_bbreg"])
        bbreg_feats = forward_samples(self.model, image, bbreg_examples)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, self.target_bbox)
        del bbreg_feats
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
        neg_examples = self.neg_generator(
            self.target_bbox, self.opts["n_neg_update"], self.opts["overlap_neg_init"]
        )
        neg_feats = forward_samples(self.model, image, neg_examples)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

    def track(self, image):
        self.frame += 1
        image = Image.fromarray(image)

        # Estimate target bbox
        samples = self.sample_generator(self.target_bbox, self.opts["n_samples"])
        sample_scores = forward_samples(self.model, image, samples, out_layer="fc6")

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        self.target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            self.target_bbox = self.target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(self.opts["trans"])
        else:
            self.sample_generator.expand_trans(self.opts["trans_limit"])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(self.model, image, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = self.target_bbox

        # Data collect
        if success:
            pos_examples = self.pos_generator(
                self.target_bbox,
                self.opts["n_pos_update"],
                self.opts["overlap_pos_update"],
            )
            pos_feats = forward_samples(self.model, image, pos_examples)
            self.pos_feats_all.append(pos_feats)
            if len(self.pos_feats_all) > self.opts["n_frames_long"]:
                del self.pos_feats_all[0]

            neg_examples = self.neg_generator(
                self.target_bbox,
                self.opts["n_neg_update"],
                self.opts["overlap_neg_update"],
            )
            neg_feats = forward_samples(self.model, image, neg_examples)
            self.neg_feats_all.append(neg_feats)
            if len(self.neg_feats_all) > self.opts["n_frames_short"]:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(self.opts["n_frames_short"], len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                None,
                self.criterion,
                self.update_optimizer,
                pos_data,
                neg_data,
                self.opts["maxiter_update"],
            )

        # Long term update
        elif self.frame % self.opts["long_interval"] == 0:
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(
                self.model,
                self.model_g,
                self.criterion,
                self.update_optimizer,
                pos_data,
                neg_data,
                self.opts["maxiter_update"],
            )

        torch.cuda.empty_cache()

        return bbreg_bbox
