# agents/storyboard.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from utils.types import StoryState
from utils.logger import log_agent_output
from utils.functional_logger import log_flow, log_api_call, log_error, log_entry_exit
from agents.ad import ad_agent
import re
import traceback  # Add this import for traceback

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
    
    print(f"Storyboard agent creating storyboard from user message: {user_message}")
    
    # First, determine if this is an explicit storyboard generation request
    is_explicit_storyboard_request = "storyboard" in user_message.lower()
    is_scene_creation_request = "create scene" in user_message.lower() or "generate scene" in user_message.lower()
    print(f"Is explicit storyboard request: {is_explicit_storyboard_request}")
    print(f"Is scene creation request: {is_scene_creation_request}")

    # Extract episode and scene numbers from user message
    episode_num = 1
    scene_num = 1
    
    # Look for episode in format "episode X" or "ep X"
    episode_match = re.search(r'episode\s+(\d+)|ep\s+(\d+)', user_message.lower())
    if episode_match:
        # Use the first group that matched
        episode_num = int(episode_match.group(1) if episode_match.group(1) else episode_match.group(2))
    
    # Look for scene in format "scene X" or "sc X"
    scene_match = re.search(r'scene\s+(\d+)|sc\s+(\d+)', user_message.lower())
    if scene_match:
        # Use the first group that matched
        scene_num = int(scene_match.group(1) if scene_match.group(1) else scene_match.group(2))
    
    print(f"Storyboard agent creating storyboard for Episode {episode_num}, Scene {scene_num}")
    
    # If this is a scene creation request, we need to generate a proper scene description first
    if is_scene_creation_request:
        # Create a prompt for generating scene description
        episode_title = ""
        if hasattr(state, "episodes") and len(state.episodes) >= episode_num:
            episode_title = state.episodes[episode_num-1].get("episode_title", f"Episode {episode_num}")
        
        scene_creation_prompt = f"""
You are a skilled screenwriter for a film or TV series. Create a detailed scene description for a specific scene in an episode.

SCENE DETAILS:
Episode: {episode_num} - {episode_title}
Scene: {scene_num}

CONTEXT:
This is for a Hindi romantic drama series about a couple going on adventure dates in Delhi.

INSTRUCTIONS:
1. Create a rich, detailed scene description that sets the location, mood, and action
2. Include vivid details about the setting in Delhi
3. Include character interactions and emotions
4. If applicable, include snippets of important dialogue
5. Make the scene dramatically and emotionally meaningful
6. Give the scene a descriptive title

FORMAT:
Title: [Scene Title]

[2-3 paragraphs of detailed scene description including setting, characters, action, and dialogue]
"""
        
        # Generate the scene description
        try:
            print("Generating detailed scene description...")
            scene_desc_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": scene_creation_prompt}],
                temperature=0.7
            )
            
            scene_description = scene_desc_response.choices[0].message.content
            
            # Extract the title if available
            scene_title = "Scene " + str(scene_num)
            title_match = re.search(r'Title:\s*(.*?)$', scene_description, re.MULTILINE)
            if title_match:
                scene_title = title_match.group(1).strip()
                # Remove the title line from the description
                scene_description = re.sub(r'Title:\s*.*?$', '', scene_description, flags=re.MULTILINE).strip()
            
            print(f"Generated scene description: {scene_description[:100]}...")
            
            # Add to structured_scenes in state
            episode_str = str(episode_num)
            if not hasattr(state, "structured_scenes"):
                state.structured_scenes = {}
                
            if episode_str not in state.structured_scenes:
                state.structured_scenes[episode_str] = []
                
            # Check if this scene already exists
            scene_exists = False
            for i, scene in enumerate(state.structured_scenes[episode_str]):
                if scene.get("scene_number") == scene_num:
                    # Update the existing scene
                    state.structured_scenes[episode_str][i]["title"] = scene_title
                    state.structured_scenes[episode_str][i]["description"] = scene_description
                    scene_exists = True
                    break
                    
            if not scene_exists:
                # Add the new scene
                state.structured_scenes[episode_str].append({
                    "scene_number": scene_num,
                    "title": scene_title,
                    "description": scene_description
                })
                
            # Also update episode_scripts for backwards compatibility
            if not hasattr(state, "episode_scripts"):
                state.episode_scripts = {}
                
            if episode_str not in state.episode_scripts:
                state.episode_scripts[episode_str] = []
                
            # Ensure episode_scripts has enough entries
            while len(state.episode_scripts[episode_str]) < scene_num:
                state.episode_scripts[episode_str].append("")
                
            # Update the scene description
            state.episode_scripts[episode_str][scene_num - 1] = f"{scene_title} - {scene_description}"
            
            # If this is just a scene creation request (not a storyboard request),
            # we can return the state with just the scene description
            if not is_explicit_storyboard_request:
                # Generate the scene description as before
                scene_description_text = scene_description
                
                # Now, break it down into shots
                shot_prompt = f"""
You are a skilled film storyboard artist. Break down this scene description into 4-8 distinct cinematic shots.

SCENE DESCRIPTION:
{scene_description_text}

For each shot, describe:
1. Camera angle and framing (close-up, wide shot, etc.)
2. What's visible in the frame
3. Character actions and expressions
4. Any movement or transitions

Format your response as:

Shot 1: [Detailed description of the first shot]

Shot 2: [Detailed description of the second shot]

And so on.
"""
                
                try:
                    print("Breaking scene into shots...")
                    shot_response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": shot_prompt}],
                        temperature=0.7
                    )
                    
                    shots_text = shot_response.choices[0].message.content
                    
                    # Format the final output with the scene title and the shots
                    output = f"# Scene {scene_num} from Episode {episode_num}\n\n"
                    output += shots_text
                    
                    state.last_agent_output = output
                    log_agent_output("Storyboard", state)
                    return state
                except Exception as e:
                    print(f"Error breaking scene into shots: {e}")
                    # Fall back to just the scene description if shot breakdown fails
                    output = f"# Scene {scene_num} from Episode {episode_num}\n\n{scene_description}"
                    state.last_agent_output = output
                    log_agent_output("Storyboard", state)
                    return state
                
        except Exception as e:
            print(f"Error generating scene description: {e}")
            # Continue with existing scene description logic if this fails
    
    # Try to find the scene description from structured_scenes if available
    scene_description = ""
    if hasattr(state, "structured_scenes") and str(episode_num) in state.structured_scenes:
        for scene in state.structured_scenes[str(episode_num)]:
            if scene.get("scene_number") == scene_num:
                scene_description = scene.get("description", "")
                print(f"Found scene description: {scene_description[:50]}...")
                break
    
    # If not found, try to find from episode_scripts (older format)
    if not scene_description and hasattr(state, "episode_scripts") and str(episode_num) in state.episode_scripts:
        scenes = state.episode_scripts[str(episode_num)]
        if scene_num <= len(scenes):
            scene_description = scenes[scene_num - 1]
            print(f"Found scene description from episode_scripts: {scene_description[:50]}...")
            
    # If still not found, construct a generic scene description
    if not scene_description:
        scene_description = f"# Scene {scene_num} from Episode {episode_num}\n\n## Shot 1\n\nA wide establishing shot of the exterior of a bustling city din..."
        print(f"No scene description found, using generic: {scene_description[:50]}...")
    
    # Create a prompt for the storyboard
    prompt = f"""
You are a skilled film storyboard artist. Create a detailed shot-by-shot storyboard for a specific scene in a film.

SCENE DETAILS:
Episode: {episode_num}
Scene: {scene_num}

SCENE DESCRIPTION:
{scene_description}

INSTRUCTIONS:
1. Create 8 distinct shots that effectively tell the visual story of this scene
2. For each shot, provide a shot number and detailed description
3. The descriptions should include camera angles, movements, and composition
4. Include details on what characters are in frame and any dialogue (if applicable)
5. Format the output as valid JSON with this exact structure:

[
  {{
    "shot_number": 1,
    "description": "Wide establishing shot of...",
    "dialogue": "Character Name: 'Any dialogue here.'" (or "NONE" if no dialogue)
  }},
  ... and so on for all shots
]

IMPORTANT GUIDELINES:
- Make each shot visually distinctive
- Vary camera angles and shot types (wide, medium, close-up)
- Consider interesting camera movements
- Focus on creating a compelling visual narrative
- Use cinematographic terms and techniques
"""
    
    # List to store shot descriptions
    shot_descriptions = []
    
    # Generate shot descriptions using GPT-4
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
                shot_key = f"ep{episode_num}_scene{scene_num}_shot{shot_num}"  # Include shot_num in the key
                shot_state.scene_scripts = {
                    shot_key: [{"shot": description, "dialogue": dialogue}]
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
                    if not hasattr(state, "last_generated_seed"):
                        state.last_generated_seed = None
                    state.last_generated_seed = updated_shot_state.last_generated_seed
                
                # Debug logs to help identify issues
                if hasattr(updated_shot_state, "ad_images"):
                    print(f"AD images keys: {list(updated_shot_state.ad_images.keys())}")
                else:
                    print("No ad_images in updated_shot_state")
                    
                # Check if we got an image - look for exact shot_key first
                found_image = False
                if hasattr(updated_shot_state, "ad_images"):
                    # Try exact matching key
                    if shot_key in updated_shot_state.ad_images:
                        original_image_url = updated_shot_state.ad_images[shot_key]
                        found_image = True
                    else:
                        # Try matching with any additional suffixes
                        for key in updated_shot_state.ad_images.keys():
                            if key.startswith(shot_key):
                                original_image_url = updated_shot_state.ad_images[key]
                                found_image = True
                                print(f"Found image URL with key {key} for shot {shot_num}")
                                break
                
                if found_image:
                    # Use the original Replicate URL directly
                    print(f"Setting image_url for shot {shot_num} to: {original_image_url[:50]}...")
                    storyboard_item["image_url"] = original_image_url
                    
                    # Copy to the state's ad_images as well
                    if not hasattr(state, "ad_images"):
                        state.ad_images = {}
                    state.ad_images[shot_key] = original_image_url
                else:
                    print(f"No image found for shot {shot_num} (key: {shot_key})")
                
                # Copy any other AD image outputs from the shot state to the main state
                if hasattr(updated_shot_state, "ad_images"):
                    for k, v in updated_shot_state.ad_images.items():
                        if k != shot_key:  # Skip the one we just handled
                            state.ad_images[k] = v
                
                # Copy AD prompts from the shot state to the main state
                if hasattr(updated_shot_state, "ad_prompts"):
                    if not hasattr(state, "ad_prompts"):
                        state.ad_prompts = {}
                    for k, v in updated_shot_state.ad_prompts.items():
                        state.ad_prompts[k] = v
                
                # Copy AD character info from the shot state to the main state
                if hasattr(updated_shot_state, "ad_character_info"):
                    if not hasattr(state, "ad_character_info"):
                        state.ad_character_info = {}
                    for k, v in updated_shot_state.ad_character_info.items():
                        state.ad_character_info[k] = v
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
        if not hasattr(state, "ad_prompts"):
            state.ad_prompts = {}
        if not hasattr(state, "ad_character_info"):
            state.ad_character_info = {}
    
    # Create a pretty markdown output for the UI
    output = f"# Storyboard: Scene {scene_num} from Episode {episode_num}\n\n"
    
    for i, shot in enumerate(shot_descriptions):
        shot_num = shot.get("shot_number", i + 1)
        description = shot.get("description", "")
        
        output += f"## Shot {shot_num}\n\n{description}\n\n"
        
        # Add image reference if available
        corresponding_item = next((item for item in storyboard_items if item["shot_number"] == shot_num), None)
        if corresponding_item and corresponding_item["image_url"]:
            output += f"![Shot {shot_num}]({corresponding_item['image_url']})\n\n"
    
    # Save the output to the state
    state.last_agent_output = output
    
    # Return the updated state
    log_agent_output("Storyboard", state)
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
    
    # Handle scene creation requests
    if "create scene" in user_message.lower() or "generate scene" in user_message.lower():
        episode_match = re.search(r'episode\s+(\d+)', user_message, re.IGNORECASE)
        scene_match = re.search(r'scene\s+(\d+)', user_message, re.IGNORECASE)
        
        if episode_match and scene_match:
            # User specified both episode and scene
            episode_num = int(episode_match.group(1))
            scene_num = int(scene_match.group(1))
            return episode_num, scene_num
    
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