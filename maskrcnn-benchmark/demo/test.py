
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo

import time


def main():
    
    cfg.merge_from_file("../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
    #cfg.merge_from_list(None)
    cfg.freeze()

    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    )

    img = cv2.imread("test.jpg")
    composite = coco_demo.run_on_opencv_image(img)
    print(composite is None)
    cv2.imwrite("./result.jpg", composite)

if __name__=='__main__':
    main()