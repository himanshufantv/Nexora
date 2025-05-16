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
    3. Requests images for each shot
    
    The result is stored in state.storyboard as a list of dictionaries
    with episode_number, scene_number, shot_number, description, and image_url.
    """
    try:
        # Determine if this is a storyboard generation request or scene creation
        is_explicit_storyboard_request = "generate storyboard" in user_message.lower() or "create storyboard" in user_message.lower()
        
        # Extract episode and scene numbers from the user message
        episode_num, scene_num = extract_episode_scene_numbers(user_message)
        if episode_num is None or scene_num is None:
            return state
            
        print(f"Storyboard agent creating storyboard for Episode {episode_num}, Scene {scene_num}")
        print(f"Is explicit storyboard request: {is_explicit_storyboard_request}")
        
        # Find the scene description to generate shots from
        scene_description = find_scene_description(state, episode_num, scene_num)
        if not scene_description:
            print(f"No scene description found for Episode {episode_num}, Scene {scene_num}")
            return state
            
        print(f"Found scene description: {scene_description[:100]}...")
        
        # Generate detailed shot-by-shot breakdown
        prompt = f"""
You are a professional storyboard artist for a TV series.

Create a detailed shot-by-shot breakdown for the following scene:

Episode {episode_num}, Scene {scene_num}
{scene_description}

Return a JSON array with 5-8 shots. Each shot should have:
1. A shot number
2. A detailed visual description that includes camera angle, movement, composition, and action
3. Any essential dialogue or sound elements

Response should be ONLY the JSON array with this structure:
[
  {{
    "shot_number": 1,
    "description": "Detailed visual description...",
    "dialogue": "Any important dialogue or NONE if no dialogue"
  }},
  ...
]
"""
        
        shot_descriptions = []
        try:
            print("Calling GPT-4 to generate shot descriptions...")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            description_text = response.choices[0].message.content
            
            # Extract JSON if in code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', description_text)
            if json_match:
                description_text = json_match.group(1).strip()
                
            # Parse shot descriptions
            import json
            try:
                shot_descriptions = json.loads(description_text)
                print(f"Generated {len(shot_descriptions)} shot descriptions")
            except json.JSONDecodeError as e:
                print(f"Error parsing shot descriptions: {e}")
                # Fall back to regex extraction if JSON parse fails
                shot_descriptions = []
                shot_matches = re.findall(r'"shot_number":\s*(\d+)[^{]*"description":\s*"([^"]*)"', description_text)
                for shot_num, desc in shot_matches:
                    shot_descriptions.append({
                        "shot_number": int(shot_num),
                        "description": desc,
                        "dialogue": "NONE"
                    })
                print(f"Extracted {len(shot_descriptions)} shot descriptions using regex")
                
        except Exception as e:
            print(f"Error generating shot descriptions: {e}")
            return state
            
        # Prepare to store the results
        storyboard_items = []
        formatted_shots = []
        
        # Initialize the scene_seeds storage if needed
        if not hasattr(state, "scene_seeds"):
            state.scene_seeds = {}
        
        # Check if there's a stored seed for this scene
        scene_key = f"ep{episode_num}_scene{scene_num}"
        scene_seed = None
        
        if scene_key in state.scene_seeds:
            scene_seed = state.scene_seeds[scene_key]
            print(f"Found existing seed {scene_seed} for {scene_key} - will use for consistent visuals")
        
        # For each shot, generate an image if this is an explicit storyboard request
        for i, shot in enumerate(shot_descriptions):
            shot_num = shot.get("shot_number", i + 1)
            description = shot.get("description", "")
            dialogue = shot.get("dialogue", "")
            
            if dialogue == "NONE":
                dialogue = ""
                
            formatted_shot = {
                "shot": description,
                "dialogue": dialogue
            }
            formatted_shots.append(formatted_shot)
            
            # Create storyboard item with initial data
            storyboard_item = {
                "episode_number": episode_num,
                "scene_number": scene_num,
                "shot_number": shot_num,
                "description": description,
                "image_url": ""  # Will be populated later if needed
            }
            
            # Add title field with story title-episode-scene format
            story_title = state.title if hasattr(state, "title") and state.title else "Story"
            storyboard_item["title"] = f"{story_title}-Episode {episode_num}-Scene {scene_num}"
            
            # Only generate images for explicit storyboard requests
            if is_explicit_storyboard_request:
                try:
                    print(f"Generating image for Shot {shot_num} using AD agent with character LoRAs")
                    
                    # Create a temporary state with just this shot for the AD agent
                    shot_state = state.copy()
                    shot_state.scene_scripts = {
                        scene_key: [{"shot": description, "dialogue": dialogue}]
                    }
                    
                    # Use the AD agent to generate the image with consistent seed
                    if scene_seed:
                        print(f"Passing seed {scene_seed} to AD agent for consistent visuals")
                        updated_shot_state = ad_agent(shot_state, episode_number=episode_num, scene_number=scene_num, seed=scene_seed)
                    else:
                        # For first shot, generate without seed to get a new one
                        updated_shot_state = ad_agent(shot_state, episode_number=episode_num, scene_number=scene_num)
                        
                        # If this is the first shot and a seed was generated, save it for consistency
                        if hasattr(updated_shot_state, "last_generated_seed") and updated_shot_state.last_generated_seed:
                            scene_seed = updated_shot_state.last_generated_seed
                            state.scene_seeds[scene_key] = scene_seed
                            print(f"Saved new seed {scene_seed} for scene {scene_key} - will use for all shots in this scene")
                    
                    # Ensure the last generated seed is saved back to the main state
                    if hasattr(updated_shot_state, "last_generated_seed") and updated_shot_state.last_generated_seed:
                        state.last_generated_seed = updated_shot_state.last_generated_seed
                    
                    # Extract the generated image URL
                    ad_scene_key = f"ep{episode_num}_scene{scene_num}_shot1"  # AD agent uses this format
                    image_url = ""
                    
                    if hasattr(updated_shot_state, "ad_images") and ad_scene_key in updated_shot_state.ad_images:
                        image_url = updated_shot_state.ad_images[ad_scene_key]
                        print(f"Generated image: {image_url[:50]}...")
                    else:
                        print(f"No image found for shot {shot_num}, trying alternative key format")
                        # Try alternative key format
                        alt_scene_key = f"episode_{episode_num}_scene_{scene_num}"
                        if hasattr(updated_shot_state, "ad_images") and alt_scene_key in updated_shot_state.ad_images:
                            image_url = updated_shot_state.ad_images[alt_scene_key]
                            print(f"Found image with alternative key: {image_url[:50]}...")
                    
                    # Update the storyboard item with the image URL
                    storyboard_item["image_url"] = image_url
                    
                    # Also capture character information for consistent visuals
                    if hasattr(updated_shot_state, "ad_character_info"):
                        # Grab any character info associated with this shot's image
                        for info_key, char_info in updated_shot_state.ad_character_info.items():
                            if info_key.startswith(f"ep{episode_num}_scene{scene_num}") or info_key.startswith(f"episode_{episode_num}_scene_{scene_num}"):
                                storyboard_item["character_info"] = char_info
                                print(f"Captured character info: {char_info}")
                                break
                    
                except Exception as e:
                    print(f"Error generating image for shot {shot_num}: {e}")
            
            # Add the storyboard item to our list
            storyboard_items.append(storyboard_item)
        
        # Format output for UI
        script_output = f"# Episode {episode_num}, Scene {scene_num} Storyboard\n\n"
        
        for i, shot in enumerate(shot_descriptions):
            script_output += f"**Shot {i+1}**: {shot['description']}\n\n"
        
        # Update the state
        if not hasattr(state, "storyboard") or state.storyboard is None:
            state.storyboard = []
        
        if is_explicit_storyboard_request:
            # Add storyboard items to state only for explicit storyboard requests
            state.storyboard.extend(storyboard_items)
            
            # Copy AD images and prompts from any updated shot states into the main state
            if not hasattr(state, "ad_images"):
                state.ad_images = {}
            
            # Look for image URLs from both AD agent formats and store them
            for item in storyboard_items:
                if item["image_url"]:
                    scene_key = f"ep{item['episode_number']}_scene{item['scene_number']}_shot{item['shot_number']}"
                    alt_key = f"episode_{item['episode_number']}_scene_{item['scene_number']}"
                    
                    # Store the image URL under multiple keys to ensure it's found later
                    state.ad_images[scene_key] = item["image_url"]
                    state.ad_images[alt_key] = item["image_url"]
                    
                    # Also store character info if available
                    if "character_info" in item and item["character_info"]:
                        if not hasattr(state, "ad_character_info"):
                            state.ad_character_info = {}
                        state.ad_character_info[scene_key] = item["character_info"]
                        state.ad_character_info[alt_key] = item["character_info"]
                        print(f"Saved character info for {scene_key} and {alt_key}")
                    
                    print(f"Copied storyboard image to state.ad_images: {scene_key} and {alt_key}")
        else:
            # For scene creation, we don't update the storyboard array, leaving it empty
            print("📝 Not updating storyboard array as this is just scene creation")
        
        state.scene_scripts = {
            **state.scene_scripts,
            scene_key: formatted_shots
        }
        
        state.last_agent_output = script_output
        
        log_agent_output("Storyboard", state)
        return state
        
    except Exception as e:
        print(f"Error in storyboard_agent: {e}")
        log_error("Storyboard", str(e))
        return state

def extract_episode_scene_numbers(user_message: str):
    """Extract episode and scene numbers from the user message."""
    # Handle generic storyboard generation requests
    if "generate storyboard" in user_message.lower() or "create storyboard" in user_message.lower():
        # Check if the message contains specific episode and scene info
        episode_match = re.search(r'episode\s+(\d+)', user_message, re.IGNORECASE)
        scene_match = re.search(r'scene\s+(\d+)', user_message, re.IGNORECASE)
        
        if episode_match and scene_match:
            # User specified both episode and scene
            episode_num = int(episode_match.group(1))
            scene_num = int(scene_match.group(1))
            return episode_num, scene_num
        else:
            # If no episode/scene specified, default to episode 1, scene 1
            print("No specific episode/scene found in storyboard request. Using defaults: Episode 1, Scene 1")
            return 1, 1
    
    # Normal extraction for specific scene creation
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
    
    # If nothing is found, return a generic description
    return f"Scene {scene_num} of Episode {episode_num}" 