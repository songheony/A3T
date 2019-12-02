import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["C_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from .expert import Expert
import tensorflow as tf

tf.get_logger().setLevel("INFO")

sys.path.append("external/siam-mcf")
from src.parse_arguments import parse_arguments
from src.siam_mcf.siam_mcf_tracker import SiamMcfTracker
import src.siamese as siam
from src.region_to_bbox import region_to_bbox


class SiamMCF(Expert):
    def __init__(self):
        super(SiamMCF, self).__init__("SiamMCF")
        root_dir = "/home/heonsong/Desktop/AAA/AAA-journal/external/siam-mcf/"
        self.hp, self.evaluation, self.env, self.design = parse_arguments(root_dir)
        self.final_score_sz = self.hp.response_up * (self.design.score_sz - 1) + 1
        # build TF graph once for all
        self.filename, self.image, self.templates_x, self.templates_z, self.scores_list = siam.build_tracking_graph(
            root_dir, self.final_score_sz, self.design, self.env, self.hp
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)
        vars_to_load = []
        for v in tf.global_variables():
            if "postnorm" not in v.name:
                vars_to_load.append(v)

        siam_ckpt_name = "/home/heonsong/Desktop/AAA/AAA-journal/external/siam-mcf/pretrained/siam_mcf.ckpt-50000"
        siam_saver = tf.train.Saver(vars_to_load)
        siam_saver.restore(self.sess, siam_ckpt_name)

    def initialize(self, image_file, box):
        pos_x, pos_y, target_w, target_h = region_to_bbox(box)
        self.tracker = SiamMcfTracker(
            self.design.context,
            self.design.exemplar_sz,
            self.design.search_sz,
            self.hp.scale_step,
            self.hp.scale_num,
            self.hp.scale_penalty,
            self.hp.scale_lr,
            self.hp.window_influence,
            self.design.tot_stride,
            self.hp.response_up,
            self.final_score_sz,
            pos_x,
            pos_y,
            target_w,
            target_h,
            image_file,
            self.sess,
            self.templates_z,
            self.filename,
        )

    def track(self, image_file):
        bbox = self.tracker.track(
            image_file,
            self.sess,
            self.templates_z,
            self.templates_x,
            self.scores_list,
            self.filename,
        )
        return bbox
