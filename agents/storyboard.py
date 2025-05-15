# agents/storyboard.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from utils.types import StoryState
from utils.logger import log_agent_output
from utils.functional_logger import log_flow, log_api_call, log_error, log_entry_exit
from agents.ad import ad_agent
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@log_entry_exit
def storyboard_agent(state: StoryState, user_message: str) -> StoryState:
    """
    Generate detailed shot-by-shot storyboard for a scene, 
    with corresponding images for each shot.
    
    This agent:
    1. Extracts episode and scene numbers from the user message
    2. Generates detailed shot descriptions
    3. Requests images for each shot from the AD agent
    4. Updates the storyboard field with the shots and images
    """
    print("🎬 Running Storyboard Agent...")
    
    # Extract episode and scene numbers from user message
    episode_num, scene_num = extract_episode_scene_numbers(user_message)
    
    if episode_num is None or scene_num is None:
        print(f"❌ Could not extract episode and scene numbers from: {user_message}")
        return state
        
    print(f"📊 Generating storyboard for Episode {episode_num}, Scene {scene_num}")
    
    # Find the scene in structured_scenes or episode_scripts
    scene_description = find_scene_description(state, episode_num, scene_num)
    
    if not scene_description:
        print(f"❌ Could not find scene description for Episode {episode_num}, Scene {scene_num}")
        return state
        
    print(f"📜 Found scene description: {scene_description[:100]}...")
    
    # Initialize the scene_seeds storage if needed
    if not hasattr(state, "scene_seeds"):
        state.scene_seeds = {}
    
    # Check if there's a stored seed for this scene
    scene_key = f"ep{episode_num}_scene{scene_num}"
    scene_seed = None
    
    if scene_key in state.scene_seeds:
        scene_seed = state.scene_seeds[scene_key]
        print(f"Found existing seed {scene_seed} for {scene_key} - will use for consistent visuals")
    
    # Generate detailed shot descriptions
    shot_descriptions = generate_shot_descriptions(state, scene_description, episode_num, scene_num)
    
    if not shot_descriptions:
        print(f"❌ Failed to generate shot descriptions")
        return state
        
    print(f"✅ Generated {len(shot_descriptions)} shot descriptions")
    
    # Generate images for each shot using AD agent
    storyboard_items = []
    
    for i, shot in enumerate(shot_descriptions):
        print(f"🖼️ Generating image for Shot {i+1}")
        
        # Call AD agent to generate image for this shot
        # Note: The AD agent handles the actual image generation via Replicate
        shot_state = state.copy()
        shot_state.scene_scripts = {
            f"ep{episode_num}_scene{scene_num}": [{"shot": shot, "dialogue": ""}]
        }
        
        # Use the AD agent to generate the image with consistent seed
        try:
            if scene_seed:
                print(f"Passing seed {scene_seed} to AD agent for consistent visuals")
                updated_state = ad_agent(shot_state, episode_number=episode_num, scene_number=scene_num, seed=scene_seed)
            else:
                updated_state = ad_agent(shot_state, episode_number=episode_num, scene_number=scene_num)
                
                # If this is the first shot and a seed was generated, save it for consistency
                if i == 0 and hasattr(updated_state, "last_generated_seed") and updated_state.last_generated_seed:
                    scene_seed = updated_state.last_generated_seed
                    print(f"✅ Saved seed {scene_seed} for scene {scene_key}")
                    state.scene_seeds[scene_key] = scene_seed
            
            # Extract the generated image URL
            scene_key = f"ep{episode_num}_scene{scene_num}_shot1"  # AD agent uses this format
            image_url = ""
            
            if hasattr(updated_state, "ad_images") and scene_key in updated_state.ad_images:
                image_url = updated_state.ad_images[scene_key]
                print(f"✅ Generated image: {image_url[:50]}...")
            else:
                print("❌ Failed to generate image for shot")
                
            # Create storyboard item
            storyboard_item = {
                "episode_number": episode_num,
                "scene_number": scene_num,
                "shot_number": i+1,
                "description": shot,
                "image_url": image_url
            }
            
            storyboard_items.append(storyboard_item)
            
        except Exception as e:
            print(f"❌ Error generating image for shot: {e}")
    
    # Update scene_scripts with the detailed shots
    formatted_shots = []
    
    for i, shot in enumerate(shot_descriptions):
        formatted_shot = {
            "shot": shot,
            "dialogue": ""
        }
        formatted_shots.append(formatted_shot)
    
    # Create the script output in markdown format
    script_output = f"#### Scene {scene_num}: {get_scene_title(state, episode_num, scene_num)}\n\n"
    
    for i, shot in enumerate(shot_descriptions):
        script_output += f"**Shot {i+1}**: {shot}\n\n"
    
    # Update the state
    if not hasattr(state, "storyboard") or state.storyboard is None:
        state.storyboard = []
        
    state.storyboard.extend(storyboard_items)
    
    state.scene_scripts = {
        **state.scene_scripts,
        scene_key: formatted_shots
    }
    
    state.last_agent_output = script_output
    
    log_agent_output("Storyboard", state)
    return state

def extract_episode_scene_numbers(user_message: str):
    """Extract episode and scene numbers from the user message."""
    # Try to find numbers using regex
    episode_match = re.search(r'episode\s+(\d+)', user_message, re.IGNORECASE)
    scene_match = re.search(r'scene\s+(\d+)', user_message, re.IGNORECASE)
    
    # If not found, look for any numbers
    if not episode_match or not scene_match:
        nums = [int(s) for s in user_message.split() if s.isdigit()]
        
        if len(nums) == 1:
            # Assume it's scene number, episode 1
            return 1, nums[0]
        elif len(nums) >= 2:
            # Assume first is episode, second is scene
            return nums[0], nums[1]
        else:
            # No numbers found
            return None, None
    
    # Extract episode and scene numbers from matches
    episode_num = int(episode_match.group(1)) if episode_match else 1
    scene_num = int(scene_match.group(1)) if scene_match else 1
    
    return episode_num, scene_num

def find_scene_description(state: StoryState, episode_num: int, scene_num: int) -> str:
    """Find the scene description from state."""
    episode_str = str(episode_num)
    
    # Check in structured_scenes first
    if hasattr(state, "structured_scenes") and state.structured_scenes:
        if episode_str in state.structured_scenes:
            scenes = state.structured_scenes[episode_str]
            for scene in scenes:
                if scene.get("scene_number") == scene_num:
                    return scene.get("description", "")
    
    # Check in episode_scripts as fallback
    if hasattr(state, "episode_scripts") and state.episode_scripts:
        if episode_str in state.episode_scripts:
            scenes = state.episode_scripts[episode_str]
            if 0 <= scene_num - 1 < len(scenes):
                return scenes[scene_num - 1]
    
    return ""

def get_scene_title(state: StoryState, episode_num: int, scene_num: int) -> str:
    """Get a title for the scene, either from structured_scenes or a default."""
    episode_str = str(episode_num)
    
    # Check in structured_scenes
    if hasattr(state, "structured_scenes") and state.structured_scenes:
        if episode_str in state.structured_scenes:
            scenes = state.structured_scenes[episode_str]
            for scene in scenes:
                if scene.get("scene_number") == scene_num:
                    return scene.get("title", f"Scene {scene_num}")
    
    return f"Scene {scene_num}"

def generate_shot_descriptions(state: StoryState, scene_description: str, episode_num: int, scene_num: int) -> list:
    """Generate detailed shot descriptions for a scene."""
    # Get character information if available
    character_info = ""
    if hasattr(state, "character_profiles") and state.character_profiles:
        characters = []
        for profile in state.character_profiles:
            name = profile.get("name", "")
            desc = profile.get("description", "")[:100]  # Truncate long descriptions
            if name:
                characters.append(f"{name}: {desc}")
        
        if characters:
            character_info = "Characters in scene:\n" + "\n".join(characters)
    
    # Generate detailed shots
    prompt = f"""
You are a professional film storyboard artist. Create 6-8 detailed shot descriptions for this scene.

Scene Description:
{scene_description}

{character_info}

For each shot, provide a detailed visual description that includes:
- Camera angle (close-up, medium shot, wide shot, etc.)
- Character positions and actions
- Setting details
- Lighting and mood

Do NOT include dialogue in the shot descriptions, only visual elements.
Each shot should be a paragraph of 2-4 sentences.

Respond with ONLY the shot descriptions, numbered as "Shot 1", "Shot 2", etc.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        shot_text = response.choices[0].message.content.strip()
        
        # Parse shots from the response
        shots = []
        current_shot = ""
        
        for line in shot_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new shot
            shot_match = re.match(r'^Shot\s+\d+:', line, re.IGNORECASE)
            if shot_match:
                # Save previous shot if it exists
                if current_shot:
                    shots.append(current_shot.strip())
                
                # Start new shot, removing the "Shot X:" prefix
                shot_prefix_end = line.find(':') + 1
                current_shot = line[shot_prefix_end:].strip()
            else:
                # Continue current shot
                current_shot += " " + line
        
        # Add the last shot
        if current_shot:
            shots.append(current_shot.strip())
        
        return shots
    except Exception as e:
        print(f"❌ Error generating shot descriptions: {e}")
        return [] 