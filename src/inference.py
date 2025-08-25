import os
import cv2
import json
import pickle
import argparse
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

#Load model configuration

def load_model(model_dir, threshold=0.5):
    """
    Loads a Detectron2 model from the specified directory.
    
    Args:
        model_dir: Directory containing config.pkl, class_names.json, model_final.pth
        threshold: Score threshold for predictions
    Returns:
        predictor, metadata
    """
    # Load config
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    
    # Load class names
    with open(os.path.join(model_dir, "class_names.json"), "r") as f:
        class_names = json.load(f)
    
    # Set model weights
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    
    # Create metadata
    dataset_name = "inference_dataset"
    MetadataCatalog.get(dataset_name).thing_classes = class_names
    metadata = MetadataCatalog.get(dataset_name)
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    return predictor, metadata

# Single image inference

def infer_image(predictor, metadata, image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    outputs = predictor(img)
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    
    # Count objects
    counts = {cls: 0 for cls in metadata.thing_classes}
    for pred_class in pred_classes:
        counts[metadata.thing_classes[pred_class]] += 1
    
    # Visualize
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = vis.get_image()[:, :, ::-1]
    
    # Overlay counts
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    for cls, count in counts.items():
        text = f"{cls}: {count}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.rectangle(result_img, (10, y_pos-25), (10+text_size[0], y_pos+5), (0,0,0), -1)
        cv2.putText(result_img, text, (10, y_pos), font, 1, (0,255,0), 2, cv2.LINE_AA)
        y_pos += 40
    
    # Save visualization if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"Visualization saved to: {output_path}")
    
    return counts, result_img


### Folder inference

def infer_folder(predictor, metadata, folder_path, output_folder=None):
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    total_counts = {cls: 0 for cls in metadata.thing_classes}
    image_counts = {}
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    for img_file in tqdm(images, desc="Running inference"):
        img_path = os.path.join(folder_path, img_file)
        counts, vis_img = infer_image(predictor, metadata, img_path)
        image_counts[img_file] = counts
        for cls, c in counts.items():
            total_counts[cls] += c
        if output_folder:
            cv2.imwrite(os.path.join(output_folder, f"pred_{img_file}"), vis_img)
    
    print("Total counts:", total_counts)
    return total_counts, image_counts


# Main CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained Detectron2 model")
    parser.add_argument("--model_dir", required=True, help="Directory with trained model (config.pkl, model_final.pth, class_names.json)")
    parser.add_argument("--image", help="Path to single image for inference")
    parser.add_argument("--folder", help="Path to folder containing images for batch inference")
    parser.add_argument("--output", help="Path to save output visualization(s)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for predictions")
    
    args = parser.parse_args()
    
    predictor, metadata = load_model(args.model_dir, threshold=args.threshold)
    
    if args.image:
        out_path = args.output if args.output else None
        counts, _ = infer_image(predictor, metadata, args.image, output_path=out_path)
        print(f"Inference counts for {args.image}: {counts}")
    
    elif args.folder:
        out_folder = args.output if args.output else os.path.join(args.folder, "inference_outputs")
        total_counts, _ = infer_folder(predictor, metadata, args.folder, output_folder=out_folder)
        print(f"Inference complete. Outputs saved to {out_folder}")
    
    else:
        print("Please provide either --image or --folder for inference.")
