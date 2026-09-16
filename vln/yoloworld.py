import os

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm
import argparse
import numpy as np
import shutil

def process_batch(json_path, output_root, start_idx=0, end_idx=None, confidence_direct=0.6, confidence_potential=0.4, gpu_id=0, max_objects=5):
    """Process JSON data within a specified range and apply YOLO model for object detection"""
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Using GPU: {gpu_id}")
    
    # Load YOLO model
    yolo_model = YOLO("xxx/yolov8x-worldv2.pt")
    
    # Read result JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine the processing range
    if end_idx is None:
        end_idx = len(data)
    batch_data = data[start_idx:end_idx]
    
    print(f"Processing {start_idx} to {end_idx} of {len(batch_data)} data")
    
    # Counter
    skipped_count = 0
    processed_count = 0
    
    # Scan root
    scan_root = 'RGB_Observations_nochange'
    
    # Get all scans in the specified batch_data
    processed_scans = set()
    for item in batch_data:
        scan = item['scan']
        processed_scans.add(scan)
    
    print(f"Scenes to be processed in this batch: {processed_scans}")
    
    # Process all found scans, ensuring no images are missed
    for scan in processed_scans:
        print(f"Processing scene: {scan}")
        # Get the scene directory
        try:
            scene_dir = set(os.listdir(os.path.join(scan_root, scan)))
        except FileNotFoundError:
            print(f"Directory not found: {os.path.join(scan_root, scan)}")
            continue
        
        # Find the data items related to this scan, get direct objects and potential objects
        scan_data = [item for item in batch_data if item['scan'] == scan]
        all_direct_obj = set()
        all_potential_obj = set()
        
        # Merge the object lists of all related data items
        for item in scan_data:
            direct_obj = item.get('direct_obj', [])
            potential_obj = item.get('potential_obj', [])
            all_direct_obj.update(direct_obj)
            all_potential_obj.update(potential_obj)
        
        print(f"Scene {scan} detection objects: direct objects {len(all_direct_obj)}, potential objects {len(all_potential_obj)}")
        
        # Set YOLO categories, if there are detection objects
        if all_direct_obj or all_potential_obj:
            yolo_model.set_classes(list(all_direct_obj) + list(all_potential_obj))
        
        # Process each scene
        for scene in scene_dir:
            scene_path = os.path.join(scan_root, scan, scene)
            try:
                image_files = os.listdir(scene_path)
            except:
                print(f"Cannot read scene directory: {scene_path}")
                continue
            
            # Process each image
            for img_name in image_files:
                img_path = os.path.join(scene_path, img_name)
                try:
                    
                    # Construct output path
                    out_dir = os.path.join(output_root, scan, scene)
                    os.makedirs(out_dir, exist_ok=True)
                    out_img_path = os.path.join(out_dir, img_name)
                    
                    # If there are no detection objects, copy the original image
                    if not all_direct_obj and not all_potential_obj:
                        shutil.copy2(img_path, out_img_path)
                        processed_count += 1
                        continue
                    
                    # Use YOLO to predict
                    results = yolo_model.predict(img_path)
                    
                    # If there are no detection objects, copy the original image
                    if (len(results) == 0 or not hasattr(results[0], 'boxes') or 
                        len(results[0].boxes) == 0):
                        shutil.copy2(img_path, out_img_path)
                        processed_count += 1
                        continue
                    
                    # Extract detection boxes and labels
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    labels = results[0].boxes.cls.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    names = results[0].names
                    
                    # Create a list, containing all detected objects and their confidence and type
                    detected_objects = []
                    for i, (box, label, conf) in enumerate(zip(boxes, labels, confs)):
                        obj_name = names[label]
                        obj_type = "direct" if obj_name in all_direct_obj and conf > confidence_direct else \
                                   "potential" if obj_name in all_potential_obj and conf > confidence_potential else "none"
                        
                        # If it is a valid object, add to the list
                        if obj_type != "none":
                            detected_objects.append({
                                'index': i,
                                'name': obj_name,
                                'box': box,
                                'conf': conf,
                                'label': label,
                                'type': obj_type
                            })
                    
                    # If there are no valid objects, copy the original image
                    if not detected_objects:
                        shutil.copy2(img_path, out_img_path)
                        processed_count += 1
                        continue
                    
                    # Sort by object type and confidence: direct first, then confidence
                    detected_objects.sort(key=lambda x: (0 if x['type'] == 'direct' else 1, -x['conf']))
                    
                    # Limit to display at most max_objects objects
                    selected_objects = detected_objects[:max_objects]
                    
                    # Process image only when there is a detection result
                    if selected_objects:
                        # Open image
                        image = Image.open(img_path).convert("RGB")
                        draw = ImageDraw.Draw(image)
                        
                        # Try to load font
                        try:
                            font = ImageFont.truetype("ARIAL.TTF", 25)
                        except:
                            font = ImageFont.load_default()
                        
                        # Draw detection boxes
                        found_objects = False
                        for obj in selected_objects:
                            obj_name = obj['name']
                            x1, y1, x2, y2 = obj['box']
                            
                            # Calculate text position - ensure it is in the top left corner of the box
                            # Get text size to correctly place it
                            if hasattr(font, 'getbbox'):  # Newer PIL
                                text_bbox = font.getbbox(obj_name)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                            else:  # Older PIL
                                text_width, text_height = font.getsize(obj_name)
                            
                            # Text position: top left corner of the box
                            text_x = x1
                            text_y = y1 - text_height - 2  # Text moved up, not covering the box
                            
                            # Ensure text does not exceed image boundaries
                            if text_y < 0:
                                text_y = y1  # If there is not enough space above, place it at the top of the box
                            
                            # Direct objects in green
                            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                            draw.text((text_x, text_y), f"{obj_name}", fill=(0, 255, 0), font=font)
                            found_objects = True
                        
                        # If found objects, save the annotated image
                        if found_objects:
                            image.save(out_img_path, quality=100)
                            processed_count += 1
                        else:
                            # If no objects are found, copy the original image
                            shutil.copy2(img_path, out_img_path)
                            processed_count += 1
                    else:
                        # If no objects are selected, copy the original image
                        shutil.copy2(img_path, out_img_path)
                        processed_count += 1
                
                except Exception as e:
                    print(f"Error processing image: {img_path}, error: {e}")
                    # When an error occurs, try to copy the original image
                    try:
                        if not os.path.exists(out_img_path):  # Only copy when the output file does not exist
                            out_dir = os.path.join(output_root, scan, scene)
                            os.makedirs(out_dir, exist_ok=True)
                            out_img_path = os.path.join(out_dir, img_name)
                            shutil.copy2(img_path, out_img_path)
                            processed_count += 1
                    except Exception as copy_error:
                        print(f"Error copying original image: {img_path}, error: {copy_error}")
    
    print(f"Processing completed! Processed {processed_count} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data and apply YOLO for object detection")
    parser.add_argument("--json", type=str, default="instr_obj_results.json", help="Path to the JSON file containing instructions and objects")
    parser.add_argument("--output", type=str, default="RGB_Observations_yolo", help="Root directory for output images")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--conf_direct", type=float, default=0.6, help="Confidence threshold for direct objects")
    parser.add_argument("--conf_potential", type=float, default=0.4, help="Confidence threshold for potential objects")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--max_objects", type=int, default=4, help="Maximum number of objects to display per image")
    
    args = parser.parse_args()
    
    process_batch(args.json, args.output, args.start, args.end, args.conf_direct, args.conf_potential, args.gpu, args.max_objects)