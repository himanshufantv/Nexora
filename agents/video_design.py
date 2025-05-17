# agents/video_design.py

from utils.types import StoryState
from utils.replicate_video_gen import generate_kling_video
import os
import re

def video_design_agent(state: StoryState) -> StoryState:
    print("🎞️ Running Video Design Agent...")

    video_clips = []
    os.makedirs("videos", exist_ok=True)

    # Process scene_scripts (original functionality)
    if hasattr(state, "scene_scripts") and state.scene_scripts:
        ad_prompts = state.ad_prompts if hasattr(state, "ad_prompts") else {}
        ad_images = state.ad_images if hasattr(state, "ad_images") else {}
        
        for scene_key, shots in state.scene_scripts.items():
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
                    video_clips.append({
                        "path": video_file,
                        "scene_key": scene_key,
                        "shot_number": idx + 1,
                        "prompt": prompt
                    })
                except Exception as e:
                    print(f"❌ Error generating video for {unique_key}: {e}")
                    continue
    
    # Process storyboard items (new functionality)
    if hasattr(state, "storyboard") and state.storyboard:
        print(f"📋 Processing {len(state.storyboard)} storyboard items")
        
        for item in state.storyboard:
            ep_num = item.get("episode_number")
            scene_num = item.get("scene_number") 
            shot_num = item.get("shot_number")
            description = item.get("description", "")
            image_url = item.get("image_url", "")
            
            # Skip items without image URL
            if not image_url:
                print(f"⚠️ Skipping storyboard item ep{ep_num}_scene{scene_num}_shot{shot_num}: no image URL")
                continue
            
            # Extract title if available for more meaningful filename
            title = item.get("title", f"ep{ep_num}_scene{scene_num}")
            # Clean title for filename - remove special chars
            clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            
            # Create unique key and filename
            unique_key = f"{clean_title}_shot{shot_num}"
            output_path = f"videos/{unique_key}.mp4"
            
            print(f"🎬 Generating video for {unique_key}...")
            
            # Create prompt for video generation
            # Use shot description as primary prompt content
            prompt = f"Create a cinematic video showing: {description}"
            
            try:
                video_file = generate_kling_video(prompt, image_url, output_path=output_path)
                print(f"✅ Saved: {video_file}")
                
                # Add detailed metadata for frontend
                video_clips.append({
                    "path": video_file,
                    "url": f"/videos/{os.path.basename(video_file)}",
                    "episode_number": ep_num,
                    "scene_number": scene_num,
                    "shot_number": shot_num,
                    "description": description,
                    "title": title,
                    "source": "storyboard"
                })
            except Exception as e:
                print(f"❌ Error generating video for {unique_key}: {e}")
                continue

    return state.copy(update={
        "video_clips": video_clips
    })
