import os
import csv
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

from openai import OpenAI
import torch
import yolov7

# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Fixed the key name

# Global variables
fileData = []
# Field that contains image paths or URLs
imageFieldName = 'image_path'  # Change this to your image field name
# Field name for the generated object classification
classificationFieldName = 'object_classification'

# Initialize YOLOv7 model
# You can use different YOLOv7 variants: 'yolov7.pt', 'yolov7x.pt', 'yolov7-w6.pt', etc.
YOLO_MODEL_PATH = 'yolov7.pt'  # Will download automatically if not present
yolo_model = None

def initialize_yolo():
    """Initialize YOLOv7 model"""
    global yolo_model
    try:
        # Load YOLOv7 model
        yolo_model = torch.hub.load('WongKinYiu/yolov7', 'custom', YOLO_MODEL_PATH, trust_repo=True)
        yolo_model.conf = 0.5  # Confidence threshold
        yolo_model.iou = 0.45  # IoU threshold for NMS
        print("YOLOv7 model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLOv7 model: {e}")
        raise

def load_image(image_path_or_url):
    """Load image from local path or URL (kept for compatibility but YOLOv7 handles this internally)"""
    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path_or_url).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path_or_url}: {e}")
        return None

def classify_objects(image_path):
    """Detect objects in image using YOLOv7 and return classification text"""
    try:
        if yolo_model is None:
            raise Exception("YOLOv7 model not initialized")
            
        # Run inference
        results = yolo_model(image_path)
        
        # Parse results
        detections = results.pandas().xyxy[0]  # Results in pandas format
        
        if len(detections) == 0:
            return "No objects detected"
        
        # Extract object labels and confidence scores
        classifications = []
        object_counts = {}
        
        for _, detection in detections.iterrows():
            label = detection['name']
            confidence = detection['confidence']
            
            # Count occurrences of each object type
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1
            
            # Store individual detections with confidence
            classifications.append(f"{label} (conf: {confidence:.2f})")
        
        # Create summary text
        summary_parts = []
        for obj_type, count in object_counts.items():
            if count > 1:
                summary_parts.append(f"{count} {obj_type}s")
            else:
                summary_parts.append(f"1 {obj_type}")
        
        # Create comprehensive classification text
        summary = "Objects detected: " + ", ".join(summary_parts)
        details = "Individual detections: " + ", ".join(classifications)
        classification_text = f"{summary}. {details}"
        
        return classification_text
        
    except Exception as e:
        print(f"Error in YOLOv7 object detection: {e}")
        return "Object detection failed"

def get_data(file_name):
    """Load CSV data"""
    csvfile = open(file_name, encoding="utf8")
    csvreader = csv.DictReader(csvfile)
    for col in csvreader:
        fileData.append(col)
    csvfile.close()
    print(f"Loaded {len(fileData)} records from {file_name}")

def process_and_embed(input_file, output_file, limit=None):
    """Process images through YOLOv7 object detection and generate embeddings"""
    
    # Initialize YOLOv7 model
    initialize_yolo()
    
    csvfile_out = open(output_file, 'w', encoding='utf8', newline='')
    fieldnames = ['id', 'name', imageFieldName, classificationFieldName, 'embedding']
    output_writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    output_writer.writeheader()
    
    llm = OpenAI(api_key=OPENAI_API_KEY)
    
    processed_count = 0
    
    for i, record in enumerate(fileData):
        if limit and processed_count >= limit:
            break
            
        # Check if image field exists and is not empty
        if imageFieldName not in record or not record[imageFieldName]:
            print(f"Skipping record {i}: No image path provided")
            continue
            
        print(f"Processing record {i+1}/{len(fileData)}: {record.get('name', 'Unknown')}")
        
        # Check if image file exists (for local paths)
        image_path = record[imageFieldName]
        if not image_path.startswith(('http://', 'https://')) and not os.path.exists(image_path):
            print(f"Skipping record {i}: Image file not found at {image_path}")
            continue
        
        # Get object classification using YOLOv7
        classification = classify_objects(image_path)
        print(f"Classification: {classification}")
        
        # Generate embedding from the classification text
        try:
            response = llm.embeddings.create(
                input=classification,
                model='text-embedding-3-large'
            )
            
            # Handle the BOM issue with id field
            id_field = record.get('\ufeffid', record.get('id', ''))
            
            output_writer.writerow({
                'id': id_field,
                'name': record.get('name', ''),
                imageFieldName: record[imageFieldName],
                classificationFieldName: classification,
                'embedding': response.data[0].embedding
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error generating embedding for record {i}: {e}")
            continue
    
    csvfile_out.close()
    print(f"Processing complete. Generated embeddings for {processed_count} records.")
    print(f"YOLOv7 model configuration: conf={yolo_model.conf}, iou={yolo_model.iou}")

def main():
    """Main execution function"""
    input_csv = "InputFile.csv"
    output_csv = "OutputFile.csv"
    
    # Load data
    get_data(input_csv)
    
    # Process images and generate embeddings
    # Add limit parameter to test with fewer records first
    process_and_embed(input_csv, output_csv, limit=None)

if __name__ == "__main__":
    main()