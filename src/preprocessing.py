import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil

def preprocess_coco_dataset(coco_json_path, image_folder, output_folder):
    """
    Preprocess a COCO dataset to ensure it's in the correct format for Detectron2.
    This function:
    1. Validates and cleans the COCO JSON annotations
    2. Ensures images are valid and properly formatted
    3. Creates a new, cleaned COCO JSON file
    4. Copies validated images to the output folder
    
    Args:
        coco_json_path (str): Path to the COCO JSON annotation file
        image_folder (str): Path to the folder containing images
        output_folder (str): Path to save preprocessed data
    
    Returns:
        str: Path to the preprocessed COCO JSON file
    """
    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    output_images_folder = os.path.join(output_folder, "images")
    os.makedirs(output_images_folder, exist_ok=True)
    
    # Load the COCO JSON file
    print(f"Loading COCO annotations from {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Check required fields
    required_fields = ['images', 'annotations', 'categories']
    for field in required_fields:
        if field not in coco_data:
            raise ValueError(f"COCO JSON is missing required field: {field}")
    
    # Add info field if missing
    if 'info' not in coco_data:
        coco_data['info'] = {
            "description": "Preprocessed COCO dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "Preprocessing script",
            "date_created": ""
        }
    
    # Add licenses field if missing
    if 'licenses' not in coco_data:
        coco_data['licenses'] = [{
            "url": "",
            "id": 1,
            "name": "Unknown"
        }]
    
    # Validate and fix categories
    print("Validating categories...")
    for i, category in enumerate(coco_data['categories']):
        # Ensure required fields exist
        if 'id' not in category:
            category['id'] = i + 1
        if 'name' not in category:
            category['name'] = f"category_{i+1}"
        if 'supercategory' not in category:
            category['supercategory'] = "object"
    
    # Create a map of category IDs to index
    category_ids = [cat['id'] for cat in coco_data['categories']]
    
    # Print category information
    print(f"Categories in dataset:")
    for cat in coco_data['categories']:
        print(f"  - ID: {cat['id']}, Name: {cat['name']}")
    
    # Initialize lists for cleaned data
    valid_images = []
    valid_annotations = []
    skipped_images = []
    skipped_annotations = []
    
    # Create a map of image ID to index for quick lookup
    image_id_map = {img['id']: i for i, img in enumerate(coco_data['images'])}
    
    # Validate images
    print("Validating images...")
    for img in tqdm(coco_data['images']):
        # Check for required fields
        if 'id' not in img or 'file_name' not in img:
            skipped_images.append(img)
            continue
        
        # Build image path
        image_path = os.path.join(image_folder, os.path.basename(img['file_name']))
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            skipped_images.append(img)
            continue
        
        # Try to read the image to verify it's valid
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image couldn't be read: {image_path}")
                skipped_images.append(img)
                continue
            
            # Add width and height if missing
            if 'width' not in img or 'height' not in img:
                height, width = image.shape[:2]
                img['width'] = width
                img['height'] = height
            
            # Copy image to output folder
            output_path = os.path.join(output_images_folder, os.path.basename(img['file_name']))
            shutil.copy2(image_path, output_path)
            
            # Add image to valid images
            valid_images.append(img)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            skipped_images.append(img)
    
    # Create a set of valid image IDs for quick lookup
    valid_image_ids = {img['id'] for img in valid_images}
    
    # Validate annotations
    print("Validating annotations...")
    for ann in tqdm(coco_data['annotations']):
        # Check for required fields
        if 'image_id' not in ann or 'category_id' not in ann or 'id' not in ann:
            skipped_annotations.append(ann)
            continue
        
        # Check if the image_id exists in valid images
        if ann['image_id'] not in valid_image_ids:
            skipped_annotations.append(ann)
            continue
        
        # Check if category_id is valid
        if ann['category_id'] not in category_ids:
            skipped_annotations.append(ann)
            continue
        
        # Check for segmentation data
        if 'segmentation' not in ann or not ann['segmentation']:
            skipped_annotations.append(ann)
            continue
        
        # Make sure segmentation is properly formatted
        seg = ann['segmentation']
        if isinstance(seg, dict):  # RLE format
            if 'counts' not in seg or 'size' not in seg:
                skipped_annotations.append(ann)
                continue
        elif isinstance(seg, list):  # Polygon format
            if not seg or not isinstance(seg[0], list):
                skipped_annotations.append(ann)
                continue
            
            # Ensure each polygon has the right number of points
            valid_seg = True
            for polygon in seg:
                if len(polygon) < 6 or len(polygon) % 2 != 0:  # At least 3 points (x,y pairs)
                    valid_seg = False
                    break
            
            if not valid_seg:
                skipped_annotations.append(ann)
                continue
        else:
            skipped_annotations.append(ann)
            continue
        
        # Check for bbox
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            # Get the image dimensions
            img_idx = image_id_map.get(ann['image_id'])
            if img_idx is not None:
                img = coco_data['images'][img_idx]
                width, height = img.get('width', 0), img.get('height', 0)
                
                # Compute bbox from segmentation
                if isinstance(seg, list):  # Polygon format
                    all_x, all_y = [], []
                    for polygon in seg:
                        for i in range(0, len(polygon), 2):
                            all_x.append(polygon[i])
                            all_y.append(polygon[i+1])
                    
                    if all_x and all_y:
                        x_min, y_min = min(all_x), min(all_y)
                        x_max, y_max = max(all_x), max(all_y)
                        ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                    else:
                        skipped_annotations.append(ann)
                        continue
                else:
                    skipped_annotations.append(ann)
                    continue
            else:
                skipped_annotations.append(ann)
                continue
        
        # Add area if missing
        if 'area' not in ann or ann['area'] <= 0:
            # Compute area from bbox
            x, y, w, h = ann['bbox']
            ann['area'] = w * h
        
        # Add iscrowd if missing
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
        
        # Add annotation to valid annotations
        valid_annotations.append(ann)
    
    # Create new COCO data with only valid entries
    cleaned_coco_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': valid_images,
        'annotations': valid_annotations
    }
    
    # Save cleaned COCO data
    output_json_path = os.path.join(output_folder, "annotations.json")
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_coco_data, f)
    
    # Print summary
    print("\nPreprocessing complete!")
    print(f"Original images: {len(coco_data['images'])}, Valid images: {len(valid_images)}")
    print(f"Original annotations: {len(coco_data['annotations'])}, Valid annotations: {len(valid_annotations)}")
    print(f"Skipped images: {len(skipped_images)}")
    print(f"Skipped annotations: {len(skipped_annotations)}")
    print(f"\nPreprocessed COCO JSON saved to: {output_json_path}")
    print(f"Preprocessed images saved to: {output_images_folder}")
    
    return output_json_path

def check_image_formats(folder_path):
    """
    Check all images in a folder for format issues and output statistics
    
    Args:
        folder_path (str): Path to the folder containing images
    """
    print(f"\nChecking image formats in {folder_path}...")
    
    extensions = {}
    dimensions = {}
    channels = {}
    invalid_images = []
    
    # Check each file in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Skip if not a file
        if not os.path.isfile(filepath):
            continue
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
        
        # Try to read the image
        try:
            img = cv2.imread(filepath)
            if img is None:
                invalid_images.append(filename)
                continue
            
            # Get dimensions
            height, width, ch = img.shape
            dim_key = f"{width}x{height}"
            dimensions[dim_key] = dimensions.get(dim_key, 0) + 1
            
            # Get channel count
            channels[ch] = channels.get(ch, 0) + 1
            
        except Exception:
            invalid_images.append(filename)
    
    # Print statistics
    print("\nImage statistics:")
    print(f"Total files: {len(os.listdir(folder_path))}")
    
    print("\nExtensions:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count}")
    
    print("\nDimensions:")
    for dim, count in dimensions.items():
        print(f"  {dim}: {count}")
    
    print("\nChannels:")
    for ch, count in channels.items():
        channel_name = {1: "Grayscale", 3: "RGB", 4: "RGBA"}.get(ch, f"{ch} channels")
        print(f"  {channel_name}: {count}")
    
    print(f"\nInvalid images: {len(invalid_images)}")
    if invalid_images:
        print("First 10 invalid images:")
        for i, img in enumerate(invalid_images[:10]):
            print(f"  {i+1}. {img}")

if __name__ == "__main__":

    COCO_PATH = '/path/to/annotations.json'
    IMAGE_FOLDER = "/path/to/images"
    OUTPUT_FOLDER = "./preprocessed_dataset"
    
    # Check image formats first
    check_image_formats(IMAGE_FOLDER)
    
    # Preprocess the dataset
    output_json = preprocess_coco_dataset(COCO_PATH, IMAGE_FOLDER, OUTPUT_FOLDER)
    
    print(f"\nYou can now use the preprocessed dataset for training:")
    print(f"COCO_PATH = '{output_json}'")
    print(f"IMAGE_FOLDER = '{os.path.join(OUTPUT_FOLDER, 'images')}'")
