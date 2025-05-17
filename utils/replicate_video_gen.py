# agents/video_design.py
from utils.types import StoryState

# utils/replicate_video_gen.py
# utils/replicate_video_gen.py
import replicate
import os
import requests
import tempfile
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def generate_kling_video(prompt: str, image_url: str, output_path: str = None) -> str:
    try:
        print(f"Downloading image from: {image_url}")
        
        # Download the image to a temporary file first
        temp_image_path = None
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Download image to temp file
            img_response = requests.get(image_url, headers=headers, stream=True)
            if img_response.status_code == 200:
                # Create temp file
                fd, temp_image_path = tempfile.mkstemp(suffix='.jpg')
                with os.fdopen(fd, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Image downloaded to temporary file: {temp_image_path}")
            else:
                print(f"Failed to download image: {img_response.status_code}")
                return ""
        except Exception as e:
            print(f"Error downloading image: {e}")
            return ""
        
        # Now use the local file for video generation
        try:
            # Open file in binary mode
            with open(temp_image_path, 'rb') as img_file:
                print(f"Running Kling model with local image file")
                output = client.run(
                    "kwaivgi/kling-v1.6-standard",
                    input={
                        "prompt": prompt,
                        "start_image": img_file  # Pass file object instead of URL
                    }
                )
                
                print(f"Kling model output: {output}")
                
                # If output_path is provided, save video to disk
                if output_path and output:
                    video_response = requests.get(output, stream=True)
                    if video_response.status_code == 200:
                        with open(output_path, "wb") as f:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Video saved to: {output_path}")
                    else:
                        print(f"Failed to download video: {video_response.status_code}")
                        output_path = ""
                
                # Remove the temporary file
                try:
                    os.remove(temp_image_path)
                    print(f"Removed temporary file: {temp_image_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")
                
                return output_path if output_path else output
        except Exception as e:
            print(f"Error running Kling model: {e}")
            # Remove the temporary file on error
            if temp_image_path:
                try:
                    os.remove(temp_image_path)
                except:
                    pass
            return ""

    except Exception as e:
        print(f"❌ Error in generate_kling_video: {e}")
        return ""

