import argparse

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    
    cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
    cfg.merge_from_list(None)
    cfg.freeze()

    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heaptamps=False,
        mask_per_dim=2,
        min_image_size=224,
    )


