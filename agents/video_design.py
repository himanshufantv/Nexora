# agents/video_design.py

from utils.types import StoryState
from utils.replicate_video_gen import generate_kling_video
import os

def video_design_agent(state: StoryState) -> StoryState:
    print("🎞️ Running Video Design Agent...")

    video_clips = []
    ad_prompts = state.ad_prompts
    ad_images = state.ad_images
    scene_scripts = state.scene_scripts

    os.makedirs("videos", exist_ok=True)

    for scene_key, shots in scene_scripts.items():
        for idx, shot in enumerate(shots):
            unique_key = f"{scene_key}_shot{idx+1}"
            prompt = ad_prompts.get(unique_key, "")
            image_url = ad_images.get(unique_key, "")

            if not prompt or not image_url:
                print(f"⚠️ Skipping {unique_key}: missing prompt or image")
                continue

            output_path = f"videos/{unique_key}.mp4"
            print(f"🎬 Generating video for {unique_key}...")

            try:
                video_file = generate_kling_video(prompt, image_url, output_path=output_path)
                print(f"✅ Saved: {video_file}")
                video_clips.append(video_file)
            except Exception as e:
                print(f"❌ Error generating video for {unique_key}: {e}")
                continue

    return state.copy(update={
        "video_clips": video_clips
    })
