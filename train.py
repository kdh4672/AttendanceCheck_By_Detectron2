from detectron2.modeling.meta_arch.build import build_model
import torch.nn as nn
from detectron2.data.datasets import register_coco_instances
register_coco_instances("Acheck", {}, "./Acheck_hair.json", "./img_hair")
from detectron2.data import MetadataCatalog
MetadataCatalog.get("Acheck").thing_classes = ["Kong", "Lee", "Huh"]
Acheck_metadata = MetadataCatalog.get("Acheck")

from detectron2.data import DatasetCatalog
dataset_dicts = DatasetCatalog.get("Acheck")

import random
import cv2
from detectron2.utils.visualizer import Visualizer

# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=Acheck_metadata, scale=0.5) ##채널순서가 image랑 imread했을 때 달라서 뒤집은것
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("a",vis.get_image()[:, :, ::-1])
#     cv2.waitKey()


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.MODEL.DEVICE='cuda:1'
yaml  = "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
weight = "./output/model_final_cascade_with_hair/mask_rcnn_R_101_C4_3x/model_final.pth"
cfg.merge_from_file(
    "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"
)
cfg.DATASETS.TRAIN = ("Acheck",)
cfg.DATASETS.TEST = ("Acheck", )
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = (
50000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (Kong, lee, Huh)
os.makedirs('./output/model_final_cascade_with_hair/mask_rcnn_R_101_C4_3x', exist_ok=True)
cfg.OUTPUT_DIR = "./output/model_final_cascade_with_hair/mask_rcnn_R_101_C4_3x"
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume= "./output/model_final_cascade_with_hair/mask_rcnn_R_101_C4_3x/model_final.pth")
trainer.train()