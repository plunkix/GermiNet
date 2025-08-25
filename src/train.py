
import os
import random
import json
import datetime
import pickle
import numpy as np
import torch
import cv2
from tqdm import tqdm
import logging

import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger


setup_logger()
logger = logging.getLogger("detectron2")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"
            ),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.3, horizontal=False, vertical=True),
            T.RandomRotation(angle=[-15, 15]),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
        ]
        mapper = detectron2.data.DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

def setup_cfg(output_dir, train_dataset_name, val_dataset_name, num_classes):
    cfg = get_cfg()
    
    # Use Mask R-CNN R50-FPN from public Detectron2 model zoo
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset & output
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.OUTPUT_DIR = output_dir
    
    # Number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Training params
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = (7000, 13000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    # Validation during training
    cfg.TEST.EVAL_PERIOD = 1000
    
    # Input sizes
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 640, 768, 896, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    
    # ROI heads thresholds
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    return cfg


def save_config(cfg, run_dir, class_names):
    os.makedirs(run_dir, exist_ok=True)
    
    # Save YAML
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    
    # Save pickle
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)
    
    # Save class names
    with open(os.path.join(run_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)


# Count & visualize predictions

def count_objects(predictor, image_path, metadata):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    outputs = predictor(img)
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    
    counts = {class_name: 0 for class_name in metadata.thing_classes}
    for pred_class in pred_classes:
        counts[metadata.thing_classes[pred_class]] += 1
    
    # Visualization
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = vis.get_image()[:, :, ::-1]
    
    # Overlay counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    for class_name, count in counts.items():
        text = f"{class_name}: {count}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.rectangle(result_img, (10, y_pos-25), (10 + text_size[0], y_pos+5), (0,0,0), -1)
        cv2.putText(result_img, text, (10, y_pos), font, 1, (0,255,0), 2, cv2.LINE_AA)
        y_pos += 40
    
    return counts, result_img

def count_objects_in_folder(predictor, image_folder, metadata, output_folder=None):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    total_counts = {cls: 0 for cls in metadata.thing_classes}
    image_counts = {}
    
    for image_file in tqdm(image_files, desc="Counting objects"):
        image_path = os.path.join(image_folder, image_file)
        counts, vis_img = count_objects(predictor, image_path, metadata)
        for cls, c in counts.items():
            total_counts[cls] += c
        image_counts[image_file] = counts
        
        if output_folder:
            cv2.imwrite(os.path.join(output_folder, f"counted_{image_file}"), vis_img)
    
    print("Total counts:", total_counts)
    return total_counts, image_counts


# Train Detectron2 model

def train_detectron2(coco_json_path, image_folder, output_dir="./detectron2_output"):
    """
    - coco_json_path: Path to COCO-format JSON annotations
    - image_folder: Path to image folder
    - output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    

    # Split dataset into train/val

    from pycocotools.coco import COCO
    coco = COCO(coco_json_path)
    all_img_ids = list(coco.imgs.keys())
    np.random.shuffle(all_img_ids)
    
    val_size = int(len(all_img_ids) * 0.2)
    val_img_ids = all_img_ids[:val_size]
    train_img_ids = all_img_ids[val_size:]
    
    # Create split JSONs
    def save_split(img_ids, path):
        imgs = [img for img in coco.dataset['images'] if img['id'] in img_ids]
        anns = [ann for ann in coco.dataset['annotations'] if ann['image_id'] in img_ids]
        split = {'images': imgs, 'annotations': anns, 'categories': coco.dataset['categories']}
        if 'info' in coco.dataset: split['info'] = coco.dataset['info']
        if 'licenses' in coco.dataset: split['licenses'] = coco.dataset['licenses']
        with open(path, 'w') as f:
            json.dump(split, f)
    
    train_json = os.path.join(run_dir, "train.json")
    val_json = os.path.join(run_dir, "val.json")
    save_split(train_img_ids, train_json)
    save_split(val_img_ids, val_json)
    
    # Register datasets
    train_name = f"train_{timestamp}"
    val_name = f"val_{timestamp}"
    register_coco_instances(train_name, {}, train_json, image_folder)
    register_coco_instances(val_name, {}, val_json, image_folder)
    
    # Class names
    categories = coco.dataset['categories']
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    MetadataCatalog.get(train_name).thing_classes = class_names
    MetadataCatalog.get(val_name).thing_classes = class_names
    num_classes = len(class_names)
    
    logger.info(f"Classes: {class_names}")
    
    # Setup config
    cfg = setup_cfg(run_dir, train_name, val_name, num_classes)
    save_config(cfg, run_dir, class_names)
    
    # Train
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Load predictor
    cfg.MODEL.WEIGHTS = os.path.join(run_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    # Evaluate
    evaluator = COCOEvaluator(val_name, cfg, False, output_dir=os.path.join(run_dir, "eval"))
    val_loader = build_detection_test_loader(cfg, val_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    logger.info(f"Evaluation results: {results}")
    
    return predictor, MetadataCatalog.get(val_name), run_dir

if __name__ == "__main__":
    # TODO: Replace with your dataset paths
    COCO_JSON_PATH = "path/to/annotations.json"
    IMAGE_FOLDER = "path/to/images"
    OUTPUT_DIR = "./detectron2_output"
    
    predictor, metadata, run_dir = train_detectron2(COCO_JSON_PATH, IMAGE_FOLDER, OUTPUT_DIR)
    
    # Count objects in the dataset
    counted_folder = os.path.join(run_dir, "counted_images")
    total_counts, _ = count_objects_in_folder(predictor, IMAGE_FOLDER, metadata, output_folder=counted_folder)
    
    print("Training and counting complete!")
    print(f"Outputs saved in: {run_dir}")
    print("Total class counts:", total_counts)
