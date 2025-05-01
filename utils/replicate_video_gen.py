# agents/video_design.py
from utils.types import StoryState

# utils/replicate_video_gen.py
# utils/replicate_video_gen.py
import replicate
import os
import requests
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def generate_kling_video(prompt: str, image_url: str, output_path: str = None) -> str:
    try:
        output = client.run(
            "kwaivgi/kling-v1.6-standard",
            input={
                "prompt": prompt,
                "start_image": image_url
            }
        )

        # If output_path is provided, save video to disk
        if output_path:
            response = requests.get(output, stream=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path

        return output

    except Exception as e:
        print(f"❌ Error generating video: {e}")
        return ""

