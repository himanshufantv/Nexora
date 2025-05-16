# agents/producer.py

from utils.types import StoryState
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VALID_AGENTS = ["Writer", "Director", "Casting", "AD", "VideoDesign", "Editor", "Storyboard"]
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
            print(f"⚠️ Invalid profile returned: {profile}. Defaulting to hindi_romantic.")
            return "hindi_romantic"
        return profile
    except Exception as e:
        print(f"❌ GPT failed to select writer profile: {e}")
        return "hindi_romantic"

# Add the missing ensure_character_consistency function
def ensure_character_consistency(state, user_input):
    """
    Ensure character information is consistent with synopsis by extracting 
    character names from synopsis if needed.
    
    Returns:
        Updated state with character information
    """
    try:
        # Only proceed if we don't have character information but have a synopsis or logline
        if (not state.characters or len(state.characters) == 0) and (state.logline or state.title):
            synopsis = state.logline if state.logline else state.title
            print(f"Ensuring character consistency using synopsis: {synopsis[:50]}...")
            
            # Extract character names from synopsis
            extract_prompt = f"""
Extract the names of the main characters mentioned in this synopsis. Return ONLY a JSON array of names, nothing else:

{synopsis}

Example: ["Name1", "Name2"]
"""
            extract_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": extract_prompt}],
                temperature=0.3
            )
            
            char_names_text = extract_response.choices[0].message.content.strip()
            # Clean up for valid JSON
            char_names_text = char_names_text.replace("```json", "").replace("```", "").strip()
            
            try:
                char_names = json.loads(char_names_text)
                print(f"Extracted character names from synopsis: {char_names}")
                
                # Generate basic character descriptions
                desc_prompt = f"""
Create basic character descriptions for these characters from the synopsis:
{", ".join(char_names)}

Synopsis: {synopsis}

For each character, provide their name followed by a brief description.
Format each as: "Character Name, brief description"
"""
                desc_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": desc_prompt}],
                    temperature=0.7
                )
                
                char_descriptions = desc_response.choices[0].message.content.strip().split("\n")
                # Filter out empty lines and quotes
                char_descriptions = [desc.strip(' "\'') for desc in char_descriptions if desc.strip()]
                
                # Add these characters to the state
                if not hasattr(state, "characters"):
                    state.characters = []
                    
                for desc in char_descriptions:
                    if desc and "," in desc:
                        state.characters.append(desc)
                
                print(f"Added {len(state.characters)} character descriptions to state")
            except Exception as parse_err:
                print(f"Error parsing character names: {parse_err}")
    except Exception as e:
        print(f"Error ensuring character consistency: {e}")
    
    return state

def producer_agent(state: StoryState, user_input: str) -> str:

    # First, ensure character consistency with the synopsis
    state = ensure_character_consistency(state, user_input)

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
        "shot image", "image for shot", "visualize shot", "generate shot image",
        "shots", "generate shots", "create shots", "shot visuals", "shot visualization",
        "shot pictures", "picture for shot", "image for shot", "show me shot",
        "show me the shots", "create image for shot", "generate visuals for shot"
    ]
    
    # ⛔ Storyboard Keywords — override routing for detailed scene shots with images
    storyboard_keywords = [
        "scene images", "storyboard", "scene storyboard",
        "visualize scene", "create shots for scene"
    ]
    
    # ✅ NEW: Detect explicit "generate storyboard" requests
    generate_storyboard_request = "generate storyboard" in user_input.lower() or "create storyboard" in user_input.lower()
    if generate_storyboard_request:
        print("✅ Explicit storyboard generation request detected, routing to Storyboard agent")
        return "Storyboard"
    
    # Check for specific scene creation requests (which should go to Storyboard for description only)
    scene_match = re.search(r'(create|make|generate)\s+scene\s+(\d+)', user_input.lower())
    scene_request = bool(scene_match) or any(keyword in user_input.lower() for keyword in storyboard_keywords)
    
    if scene_request:
        print("✅ Scene creation request detected, routing to Storyboard agent")
        return "Storyboard"
    
    # Check for more specific shot requests (which should go to AD)
    shot_image_request = any(keyword in user_input.lower() for keyword in ad_keywords)
    
    # Check for scene requests without images (which should go to Writer)
    scene_keywords = [
        "generate scene", "write scene", "scene for episode",
        "scene description", "scene breakdown", "scene details", "develop scene"
    ]
    scene_without_image_request = any(keyword in user_input.lower() for keyword in scene_keywords) and not shot_image_request and not scene_request
    
    if shot_image_request:
        print("✅ Shot image request detected, routing to AD agent")
        return "AD"
    
    if scene_without_image_request:
        print("✅ Scene description request detected, routing to Writer agent")
        writer_profile = detect_writer_profile_gpt(user_input)
        return f"Writer::{writer_profile}"
        
    # ⛔ VideoDesign Keywords — override routing for video requests
    video_keywords = [
        "video", "animate", "animation", "motion", "film", "movie", "clip", 
        "create video", "generate video", "make video", "video clip", "film clip"
    ]
    if any(keyword in user_input.lower() for keyword in video_keywords):
        print("✅ Video request detected, routing to VideoDesign agent")
        return "VideoDesign"

    # ✅ Detect profile only if not set
    if isinstance(state, dict) and state.get("writer_profile"):
        writer_profile = state.writer_profile
    else:
        writer_profile = detect_writer_profile_gpt(user_input)
        
    # 🎬 Ask GPT which agent to use
    prompt = f"""
You are the Producer Agent in a multi-agent AI filmmaking pipeline.

Based on the current state and user input, choose the next AGENT to run.

Available agents:
Writer, Director, Casting, AD, VideoDesign, Editor, Storyboard

Guidance:
- If the user asks about story, script, plot, episode, or scene → return Writer
- If the user asks about characters or visuals → return Casting
- Return ONLY one valid agent name from the list above. No explanations.

{state_summary}

User Message:
\"{user_input}\"
"""

    try:
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