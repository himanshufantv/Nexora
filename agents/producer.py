# agents/producer.py

from utils.types import StoryState
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VALID_AGENTS = ["Writer", "Director", "Casting", "AD", "VideoDesign", "Editor"]
VALID_PROFILES = ["hindi_romantic", "hindi_action", "english_romantic", "english_action"]

def detect_writer_profile_gpt(user_input: str) -> str:
    profile_prompt = f"""
You are a style router in an AI filmmaking system.

Choose the most appropriate writer profile from the list below for this story request.

Available profiles:
- hindi_romantic
- hindi_action
- english_romantic
- english_action

Return ONLY one of the profile keys. Do not explain.

User request:
\"{user_input}\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": profile_prompt}],
            temperature=0
        )
        profile = response.choices[0].message.content.strip()
        if profile not in VALID_PROFILES:
            print(f"⚠️ Invalid profile returned: {profile}. Defaulting to english_romantic.")
            return "hindi_action"
        return profile
    except Exception as e:
        print(f"❌ GPT failed to select writer profile: {e}")
        return "english_action"

def producer_agent(state: StoryState, user_input: str) -> str:

    state_summary = f"""
    Current Story State:
    - Title: {state.title or 'N/A'}
    - Episodes: {len(state.episodes)}
    - Episode Scripts: {len(state.episode_scripts)}
    - Scene Scripts: {len(state.scene_scripts)}
    - Characters: {len(state.characters)}
    - Scene Images: {len(state.scene_image_prompts)}
    - Video Clips: {len(state.video_clips)}
    """

    # ⛔ Casting Keywords — override routing if user wants images/casting
    casting_keywords = [
        "generate images", "character images", "cast", "casting",
        "character photos", "faces", "actor look", "create images of characters",
        "character visuals", "generate character image"
    ]
    if any(keyword in user_input.lower() for keyword in casting_keywords):
        return "Casting"
        
    # ⛔ AD Keywords — override routing if user wants scene/setting images
    ad_keywords = [
        "scene images", "visualize scenes", "scene visuals", "generate scene images",
        "show me what the scene looks like", "create scene visuals", "scene photos",
        "episode visuals", "visualize episode", "show scenes", "generate visuals",
        "scene image", "episode image", "show episode", "episode visualization", 
        "scene visualization", "scene pictures", "episode pictures", "create scene images",
        "fix scene images", "generate image for scene", "generate scene images",
        "create visuals", "scene visual", "show me scene", "image for scene",
        "scene image", "scene", "image"  # Add very generic terms as a fallback
    ]
    if any(keyword in user_input.lower() for keyword in ad_keywords):
        print("✅ Scene image request detected, routing to AD agent")
        return "AD"
        
    # ⛔ VideoDesign Keywords — override routing for video requests
    video_keywords = [
        "video", "animate", "animation", "motion", "film", "movie", "clip", 
        "create video", "generate video", "make video", "video clip", "film clip"
    ]
    if any(keyword in user_input.lower() for keyword in video_keywords):
        print("✅ Video request detected, routing to VideoDesign agent")
        return "VideoDesign"

    print(f"❌ I AM HERE 232343 {state}")

    # ✅ Detect profile only if not set
    if isinstance(state, dict) and state.get("writer_profile"):
        print(f"❌ I AM HERE 232343")
        writer_profile = state.writer_profile
    else:
        print(f"❌ I AM HERE 32434234")
        writer_profile = detect_writer_profile_gpt(user_input)
        print(f"❌ I AM HERE 324342342324324 {writer_profile}")
        
    # 🎬 Ask GPT which agent to use
    prompt = f"""
You are the Producer Agent in a multi-agent AI filmmaking pipeline.

Based on the current state and user input, choose the next AGENT to run.

Available agents:
Writer, Director, Casting, AD, VideoDesign, Editor

Guidance:
- If the user asks about story, script, plot, episode, or scene → return Writer
- If the user asks about characters or visuals → return Casting
- Return ONLY one valid agent name from the list above. No explanations.

{state_summary}

User Message:
\"{user_input}\"
"""

    try:
        print(f"❌ I AM HERE write {state_summary}")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        agent = response.choices[0].message.content.strip()
        if agent not in VALID_AGENTS:
            print(f"⚠️ Invalid agent returned: {agent}. Defaulting to Writer.")
            agent = "Writer"

        if agent == "Writer":
            return f"Writer::{writer_profile}"
        return agent

    except Exception as e:
        print(f"❌ Producer Agent GPT call failed: {e}")
        return f"Writer::{writer_profile}"
