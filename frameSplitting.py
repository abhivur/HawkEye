import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir=None, frame_interval=3, image_format='jpg'):
    """
    Extract frames from a video file and save them as individual images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames (default: creates folder based on video name)
        frame_interval (int): Extract every nth frame (1 = all frames, 2 = every other frame, etc.)
        image_format (str): Output image format ('jpg', 'png', 'bmp', etc.)
    
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
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"  Output directory: {output_dir}")
    print(f"  Frame interval: {frame_interval}")
    
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame based on interval
            if frame_count % frame_interval == 0:
                # Generate filename with zero-padded frame number
                filename = f"frame_{frame_count:06d}.{image_format}"
                output_path = os.path.join(output_dir, filename)
                
                # Save frame
                success = cv2.imwrite(output_path, frame)
                
                if success:
                    extracted_count += 1
                    if extracted_count % 100 == 0:  # Progress indicator
                        print(f"  Extracted {extracted_count} frames...")
                else:
                    print(f"  Warning: Failed to save frame {frame_count}")
            
            frame_count += 1
    
    finally:
        cap.release()
    
    print(f"Extraction complete! {extracted_count} frames saved to '{output_dir}'")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("C:\Users\lukev\Downloads\VisDrone2019-VID-train\VisDrone2019-VID-train\sequences\uav0000150_02310_v", help="Path to the input video file")
    parser.add_argument("-o", "--output", help="Output directory for frames")
    parser.add_argument("-i", "--interval", type=int, default=3, 
                       help="Frame interval (extract every nth frame, default: 3)")
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "png", "bmp"],
                       help="Output image format (default: jpg)")
    
    args = parser.parse_args()
    
    try:
        extract_frames(
            video_path=args.video_path,
            output_dir=args.output,
            frame_interval=args.interval,
            image_format=args.format
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())