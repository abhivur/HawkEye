import os
import json
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# =============================================================================
# 🔧 FOLDER SEARCH CONFIGURATION - CHANGE THESE FOR YOUR SEARCH
# =============================================================================

# Path to your images folder (CHANGE THIS!)
IMAGES_FOLDER_PATH = "C:/Users/lukev/Projects/HawkEye/panover_frames"  # Folder containing all images
# Examples: 
# IMAGES_FOLDER_PATH = "C:/Users/YourName/Desktop/photos/"     # Windows
# IMAGES_FOLDER_PATH = "/home/user/images/"                   # Linux
# IMAGES_FOLDER_PATH = "/Users/username/Pictures/dataset/"    # Mac

# Extract folder name and generate output JSON path dynamically
folder_name = os.path.basename(IMAGES_FOLDER_PATH.rstrip('/\\'))
if folder_name.endswith('_frames'):
    base_name = folder_name[:-7]  # Remove '_frames' suffix
else:
    base_name = folder_name  # Fallback if pattern doesn't match

# Output JSON file for the mission planning system
OUTPUT_JSON_PATH = f"{base_name}_detection_results.json"



# Supported image formats
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# =============================================================================
# Model Configuration (usually don't need to change these)
# =============================================================================

# Initialize YOLOv8 model with Open Images V7 (600 classes)
YOLO_MODEL_PATH = 'yolov8s.pt' #'yolov8n-oiv7.pt'  # 600 classes model
yolo_model = None

# Initialize Sentence Transformer model - UPGRADED!
SENTENCE_TRANSFORMER_MODEL = 'all-mpnet-base-v2'  # 768 dim, much better than MiniLM
sentence_model = None

def initialize_yolo():
    """Initialize YOLOv8 model using Ultralytics"""
    global yolo_model
    try:
        print(f"   Loading YOLOv8 model: {YOLO_MODEL_PATH}")
        yolo_model = YOLO(YOLO_MODEL_PATH)  # Will download automatically if not present
        print("   ✅ YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading YOLOv8 model: {e}")
        raise

def initialize_sentence_transformer():
    """Initialize Sentence Transformer model"""
    global sentence_model
    try:
        print(f"   Loading Sentence Transformer model: {SENTENCE_TRANSFORMER_MODEL}")
        sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        print(f"   ✅ Sentence Transformer loaded successfully")
        print(f"   ✅ Model produces {sentence_model.get_sentence_embedding_dimension()}-dimensional embeddings")
    except Exception as e:
        print(f"   ❌ Error loading Sentence Transformer model: {e}")
        raise

def process_all_frames_and_save_json():
    """Process all images in folder and save results to JSON for mission planning system"""
    
    print("🔍 FULL FOLDER PROCESSING - Process All Frames")
    print("="*70)
    print(f"📂 Folder: {IMAGES_FOLDER_PATH}")
    print(f"💾 Output JSON: {OUTPUT_JSON_PATH}")
    print("="*70)
    
    # Check if folder exists
    if not os.path.exists(IMAGES_FOLDER_PATH):
        print(f"❌ Error: Folder not found at {IMAGES_FOLDER_PATH}")
        print("💡 Please update IMAGES_FOLDER_PATH with the correct folder path")
        return
    
    # Get all image files from folder
    image_files = get_image_files(IMAGES_FOLDER_PATH)
    
    if not image_files:
        print(f"❌ No image files found in {IMAGES_FOLDER_PATH}")
        print(f"💡 Supported formats: {SUPPORTED_FORMATS}")
        return
    
    print(f"📊 Found {len(image_files)} images to process")
    
    # Initialize models
    print("\n🤖 Initializing models...")
    initialize_yolo()
    initialize_sentence_transformer()
    
    # Process all images
    print(f"\n🔍 Processing all {len(image_files)} frames...")
    print("-" * 70)
    
    frame_data_list = []
    total_objects_processed = 0
    
    for img_index, image_file in enumerate(image_files, 1):
        
        # Generate GPS coordinates - NULL since we don't know actual location
        gps_coords = [None, None, None]  # [lat, lon, alt] - will be null in JSON
        
        # Generate timestamp (you can modify this to read from EXIF or file timestamps)
        timestamp = float(img_index)
        
        # Detect objects in current image
        objects = detect_individual_objects(image_file)
        
        if not objects:
            # Still add frame data even with no objects
            frame_data = {
                "frame_id": img_index,
                "timestamp": timestamp,
                "frame_path": os.path.abspath(image_file),
                "gps_coords": gps_coords,
                "detected_objects": []
            }
            frame_data_list.append(frame_data)
            continue
        
        total_objects_processed += len(objects)
        
        # Show detected objects
        detected_objects_json = []
        for i, obj in enumerate(objects, 1):
            
            # Convert to JSON format expected by mission planning system
            obj_json = {
                "label": obj['label'],
                "confidence": obj['confidence'],
                "bbox": obj['bounding_box'],
                "area": obj['area']
            }
            detected_objects_json.append(obj_json)
        
        # Create frame data entry
        frame_data = {
            "frame_id": img_index,
            "timestamp": timestamp,
            "frame_path": os.path.abspath(image_file),
            "gps_coords": gps_coords,
            "detected_objects": detected_objects_json
        }
        
        frame_data_list.append(frame_data)
    
    # Save results to JSON file
    output_data = frame_data_list  # Direct list format as expected by mission planning system
    
    try:
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print("\n" + "🎉" * 20)
        print("✅ PROCESSING COMPLETED! ✅")
        print("🎉" * 20)
        print(f"📁 Processed {len(image_files)} frames")
        print(f"🔢 Total objects detected: {total_objects_processed}")
        print(f"💾 Results saved to: {os.path.abspath(OUTPUT_JSON_PATH)}")
        print(f"📊 Frames with objects: {len([f for f in frame_data_list if f['detected_objects']])}")
        print(f"📊 Frames without objects: {len([f for f in frame_data_list if not f['detected_objects']])}")
        
        # Show sample of what was saved
        print(f"\n📄 First frame processed:")
        if frame_data_list:
            sample_frame = frame_data_list[0]
            print(f"   Frame ID: {sample_frame['frame_id']}")
            print(f"   GPS: {sample_frame['gps_coords']}")
            print(f"   Objects detected: {len(sample_frame['detected_objects'])}")
            if sample_frame['detected_objects']:
                print(f"   Sample object: {sample_frame['detected_objects'][0]['label']} (conf: {sample_frame['detected_objects'][0]['confidence']:.3f})")
        
        print(f"\n💡 You can now use '{OUTPUT_JSON_PATH}' with the mission planning system!")
        
    except Exception as e:
        print(f"\n❌ Error saving JSON file: {e}")



def get_image_files(folder_path):
    """Get all image files from the specified folder and ALL subfolders (recursive)"""
    image_files = []
    
    try:
        print(f"🔍 Searching for images in {folder_path} and all subfolders...")
        
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_FORMATS):
                    full_path = os.path.join(root, file)
                    if os.path.isfile(full_path):
                        image_files.append(full_path)
        
        # Sort files for consistent processing order
        image_files.sort()
        
        if image_files:
            print(f"✅ Found {len(image_files)} images across all subfolders")
            # Show first few folders found
            folders_found = set(os.path.dirname(img) for img in image_files[:10])
            print(f"📂 Sample folders: {list(folders_found)[:3]}...")
        
        return image_files
        
    except Exception as e:
        print(f"❌ Error reading folder: {e}")
        return []

def detect_individual_objects(image_path):
    """Detect individual objects in image using YOLOv8 and return list of object data"""
    try:
        if yolo_model is None:
            raise Exception("YOLOv8 model not initialized")
            
        # Run inference with YOLOv8 - LOWER confidence for harder-to-detect objects
        results = yolo_model(image_path, conf=0.25, iou=0.45, verbose=False)  # Lowered from 0.5 to 0.25
        
        # Get the first result (since we're processing one image)
        result = results[0]
        
        if len(result.boxes) == 0:
            return []
        
        individual_objects = []
        
        # Process each detection
        for idx, box in enumerate(result.boxes):
            # Get class name and confidence
            class_id = int(box.cls.item())
            label = result.names[class_id]  # Convert class ID to label name
            confidence = box.conf.item()
            
            # Get bounding box coordinates (xyxy format)
            coords = box.xyxy[0].cpu().numpy()  # Convert to numpy
            xmin, ymin, xmax, ymax = coords
            
            # Create detailed caption for this specific object
            object_caption = f"{label} detected with {confidence:.2f} confidence"
            
            # Add size information if bounding box is available
            width = xmax - xmin
            height = ymax - ymin
            size_desc = ""
            if width > 0 and height > 0:
                # Determine relative size
                total_area = width * height
                if total_area > 50000:  # Large object (adjust threshold as needed)
                    size_desc = "large "
                elif total_area < 10000:  # Small object
                    size_desc = "small "
                else:
                    size_desc = ""
            
            # Enhanced caption with context (simple is better for similarity)
            enhanced_caption = f"{size_desc}{label}"
            
            # Store object information
            object_info = {
                'object_id': f"obj_{idx + 1}",
                'label': label,
                'confidence': confidence,
                'caption': object_caption,
                'enhanced_caption': enhanced_caption,
                'bounding_box': {
                    'xmin': float(xmin),
                    'ymin': float(ymin), 
                    'xmax': float(xmax),
                    'ymax': float(ymax)
                },
                'area': float(width * height) if width > 0 and height > 0 else 0
            }
            
            individual_objects.append(object_info)
        
        return individual_objects
        
    except Exception as e:
        print(f"Error in YOLOv8 object detection: {e}")
        return []



def main():
    """Main function - Process all frames and save to JSON"""
    print("🎯 YOLO + Object Detection - Full Frame Processing Tool")
    print("💡 Edit IMAGES_FOLDER_PATH and OUTPUT_JSON_PATH at the top of the file")
    print("📦 Using YOLOv8 + Sentence Transformers")
    print("📄 Outputs JSON format compatible with mission planning system")
    print()
    
    # Run full processing
    process_all_frames_and_save_json()

if __name__ == "__main__":
    main()
