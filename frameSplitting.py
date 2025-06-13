import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir=None, frame_interval=3, image_format='jpg', 
                   resize_width=None, resize_height=None, quality=95):
    """
    Extract frames from a video file and save them as individual images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames (default: creates folder based on video name)
        frame_interval (int): Extract every nth frame (1 = all frames, 2 = every other frame, etc.)
        image_format (str): Output image format ('jpg', 'png', 'bmp', etc.)
        resize_width (int): Resize frame width (maintains aspect ratio if height not specified)
        resize_height (int): Resize frame height (maintains aspect ratio if width not specified)
        quality (int): JPEG quality (0-100, higher is better quality)
    
    Returns:
        int: Number of frames extracted
    """
    
    # Validate video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if not specified
    if output_dir is None:
        video_name = Path(video_path).stem
        output_dir = f"{video_name}_frames"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎥 Video Info:")
    print(f"  📁 File: {Path(video_path).name}")
    print(f"  📊 Total frames: {total_frames:,}")
    print(f"  🎬 FPS: {fps:.2f}")
    print(f"  📐 Resolution: {width}x{height}")
    print(f"  ⏱️  Duration: {total_frames/fps:.2f} seconds")
    print(f"  📂 Output directory: {output_dir}")
    print(f"  ⏭️  Frame interval: {frame_interval} (every {frame_interval} frames)")
    if resize_width or resize_height:
        print(f"  🔄 Will resize to: {resize_width or 'auto'}x{resize_height or 'auto'}")
    
    expected_extractions = total_frames // frame_interval
    print(f"  📤 Expected extractions: ~{expected_extractions:,} frames")
    print(f"  🔄 Processing...")
    
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame based on interval
            if frame_count % frame_interval == 0:
                # Resize frame if specified
                processed_frame = frame
                if resize_width or resize_height:
                    if resize_width and resize_height:
                        processed_frame = cv2.resize(frame, (resize_width, resize_height))
                    elif resize_width:
                        aspect_ratio = frame.shape[0] / frame.shape[1]
                        new_height = int(resize_width * aspect_ratio)
                        processed_frame = cv2.resize(frame, (resize_width, new_height))
                    elif resize_height:
                        aspect_ratio = frame.shape[1] / frame.shape[0]
                        new_width = int(resize_height * aspect_ratio)
                        processed_frame = cv2.resize(frame, (new_width, resize_height))
                
                # Generate filename with zero-padded frame number
                filename = f"frame_{frame_count:06d}.{image_format}"
                output_path = os.path.join(output_dir, filename)
                
                # Set compression parameters
                save_params = []
                if image_format.lower() in ['jpg', 'jpeg']:
                    save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                elif image_format.lower() == 'png':
                    save_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
                
                # Save frame
                success = cv2.imwrite(output_path, processed_frame, save_params)
                
                if success:
                    extracted_count += 1
                    # Simple progress indicator every 100 frames
                    if extracted_count % 100 == 0:
                        print(f"  ✅ Extracted {extracted_count} frames...")
                else:
                    print(f"  ⚠️ Warning: Failed to save frame {frame_count}")
            
            frame_count += 1
    
    finally:
        cap.release()
    
    print(f"\n🎉 Extraction complete!")
    print(f"  📤 {extracted_count:,} frames saved to '{output_dir}'")
    print(f"  💾 Storage: ~{extracted_count * 2 / 1024:.1f} MB (estimated)")
    
    return extracted_count


def main():
    # Hardcoded for testing - no command line arguments needed
    video_path = r"C:\Users\lukev\Downloads\IMG_6231.mov"  # Note the 'r' prefix
    
    try:
        extract_frames(
            video_path=video_path,
            output_dir=None,  # Will create IMG_6227_frames folder
            frame_interval=3,
            image_format='jpg'
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())