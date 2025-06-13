import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import torch
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# =============================================================================
# 🔧 FOLDER SEARCH CONFIGURATION - CHANGE THESE FOR YOUR SEARCH
# =============================================================================

# Path to your images folder (CHANGE THIS!)
IMAGES_FOLDER_PATH = "C:/Users/lukev/Projects/HawkEye/IMG_6231_frames"  # Folder containing all images
# Examples: 
# IMAGES_FOLDER_PATH = "C:/Users/YourName/Desktop/photos/"     # Windows
# IMAGES_FOLDER_PATH = "/home/user/images/"                   # Linux
# IMAGES_FOLDER_PATH = "/Users/username/Pictures/dataset/"    # Mac

# Your search query (CHANGE THIS!)
TEST_QUERY = "coffee mug"  # What you're looking for
# Examples:
# TEST_QUERY = "red circular objects"
# TEST_QUERY = "animals"
# TEST_QUERY = "vehicles"
# TEST_QUERY = "people"
# TEST_QUERY = "sports equipment"

# Similarity threshold to stop searching (80% = 0.80)
SIMILARITY_THRESHOLD = 0.80  # Stop when match is this good or better

# Supported image formats
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# =============================================================================
# Model Configuration (usually don't need to change these)
# =============================================================================

# Initialize YOLOv8 model with Open Images V7 (600 classes)
YOLO_MODEL_PATH = 'yolov8n-oiv7.pt'  # 600 classes model
yolo_model = None

# Initialize Sentence Transformer model - UPGRADED!
SENTENCE_TRANSFORMER_MODEL = 'all-mpnet-base-v2'  # 768 dim, much better than MiniLM
sentence_model = None

def initialize_yolo():
    """Initialize YOLOv8 model using Ultralytics"""
    global yolo_model
    try:
        print(f"   Loading YOLOv8-OIV7 model: {YOLO_MODEL_PATH}")
        yolo_model = YOLO(YOLO_MODEL_PATH)  # Will download automatically if not present
        print("   ✅ YOLOv8 COCO model loaded successfully (80 classes - like the website)")
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

def search_folder_for_object():
    """Search through all images in folder until finding object with 90%+ similarity"""
    
    print("🔍 FOLDER SEARCH MODE - Find Object and Stop")
    print("="*70)
    print(f"📂 Folder: {IMAGES_FOLDER_PATH}")
    print(f"🎯 Query: '{TEST_QUERY}'")
    print(f"🎚️  Similarity Threshold: {SIMILARITY_THRESHOLD*100:.0f}%")
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
    
    # Search through images
    print(f"\n🔍 Starting search for '{TEST_QUERY}'...")
    print("-" * 70)
    
    total_objects_processed = 0
    
    for img_index, image_file in enumerate(image_files, 1):
        print(f"\n📸 Processing image {img_index}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Detect objects in current image
        objects = detect_individual_objects(image_file)
        
        if not objects:
            print(f"   ℹ️  No objects detected")
            continue
        
        print(f"   ✅ Detected {len(objects)} objects")
        total_objects_processed += len(objects)
        
        # Show ALL detected objects (for debugging)
        print(f"   📋 All objects found:")
        for i, obj in enumerate(objects, 1):
            print(f"      {i}. {obj['enhanced_caption']} (conf: {obj['confidence']:.3f})")
        
        # Generate embeddings for all objects in this image
        captions = [obj['enhanced_caption'] for obj in objects]
        embeddings = generate_embeddings_quiet(captions)
        
        if len(embeddings) == 0:
            print(f"   ❌ Failed to generate embeddings")
            continue
        
        # Check each object for similarity match
        query_embedding = sentence_model.encode([TEST_QUERY], normalize_embeddings=True)
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # Find best match in this image
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        best_object = objects[best_match_idx]
        
        print(f"   🎯 Best match: {best_object['enhanced_caption']} (similarity: {best_similarity:.3f})")
        
        # Check if we found our target!
        if best_similarity >= SIMILARITY_THRESHOLD:
            print("\n" + "🎉" * 20)
            print("🎯 TARGET FOUND! 🎯")
            print("🎉" * 20)
            print(f"📁 Image: {os.path.basename(image_file)}")
            print(f"📍 Full path: {image_file}")
            print(f"🎯 Object: {best_object['enhanced_caption']}")
            print(f"📊 Similarity: {best_similarity:.3f} ({best_similarity*100:.1f}%)")
            print(f"🎗️  Confidence: {best_object['confidence']:.3f}")
            print(f"📏 Bounding box: {best_object['bounding_box']}")
            print(f"📈 Search stats: Processed {img_index} images, {total_objects_processed} total objects")
            print("\n✅ SEARCH COMPLETED - Target found!")
            return  # Stop searching!
        
        # Show progress for objects that didn't meet threshold
        for i, obj in enumerate(objects):
            similarity = similarities[i]
            if similarity > 0.3:  # Only show decent matches
                status = "🟡" if similarity > 0.6 else "🟠" if similarity > 0.4 else "🔴"
                print(f"      {status} {obj['enhanced_caption']}: {similarity:.3f}")
    
    # If we get here, we didn't find a match above threshold
    print("\n" + "❌" * 20)
    print("🔍 SEARCH COMPLETED - No match found")
    print("❌" * 20)
    print(f"📊 Processed {len(image_files)} images")
    print(f"🔢 Analyzed {total_objects_processed} total objects")
    print(f"🎚️  No object reached {SIMILARITY_THRESHOLD*100:.0f}% similarity threshold")
    print(f"💡 Try lowering SIMILARITY_THRESHOLD or using a different query")

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

def generate_embeddings_quiet(texts):
    """Generate embeddings without progress bar (for folder processing)"""
    try:
        if sentence_model is None:
            raise Exception("Sentence Transformer model not initialized")
        
        if not texts:
            return np.array([])
        
        # Generate embeddings quietly
        embeddings = sentence_model.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
        
    except Exception as e:
        print(f"   ❌ Error generating embeddings: {e}")
        return np.array([])

def detect_individual_objects(image_path):
    """Detect individual objects in image using YOLOv8-OIV7 and return list of object data"""
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
    """Main function - Search folder for object with 90%+ similarity"""
    print("🎯 YOLO + Sentence Transformers - Folder Search Tool")
    print("💡 Edit IMAGES_FOLDER_PATH and TEST_QUERY at the top of the file")
    print("📦 Using YOLOv8 COCO (80 classes) + Sentence Transformers - TESTING")
    print("🧪 Testing with same model type as successful website")
    print("🛑 Stops when finding object with 90%+ similarity")
    print()
    
    # Run folder search
    search_folder_for_object()

if __name__ == "__main__":
    main()