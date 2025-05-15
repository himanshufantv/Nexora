# agents/ad.py

import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from utils.types import StoryState
from utils.logger import log_agent_output
from utils.functional_logger import log_flow, log_api_call, log_db_operation, log_error, log_entry_exit
from utils.replicate_image_gen import generate_flux_image
from db.models.casting_characters import get_candidate_characters
from db.models.story_projects import update_story_project_by_session

# First make sure replicate is imported
import replicate

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@log_entry_exit
def detect_characters_in_text(text, character_names):
    """
    Detect which characters appear in a text description.
    
    Args:
        text (str): The text to analyze
        character_names (list): List of character names to look for
    
    Returns:
        list: Names of characters found in the text
    """
    print(f"Detecting characters in text from {len(character_names)} possible characters")
    found_characters = []
    lowered_text = text.lower()
    
    # Enhanced detection to work with various forms of character names
    for name in character_names:
        # Skip if no name
        if not name or len(name) < 2:
            continue
            
        # Convert name to lowercase for case-insensitive matching
        name_lower = name.lower()
        
        # 1. Handle multi-word names (e.g., "John Smith")
        if " " in name:
            if name_lower in lowered_text:
                found_characters.append(name)
                print(f"Found multi-word character (exact match): {name}")
                continue
                
            # Try first name / last name separately
            name_parts = name_lower.split()
            for part in name_parts:
                if len(part) >= 3:  # Only match on meaningful parts (not "a", "the", etc.)
                    pattern = rf'\b{re.escape(part)}\b'
                    if re.search(pattern, lowered_text):
                        found_characters.append(name)
                        print(f"Found multi-word character (partial match on '{part}'): {name}")
                        break
        
        # 2. Handle single-word names with word boundaries
        else:
            pattern = rf'\b{re.escape(name_lower)}\b'
            if re.search(pattern, lowered_text):
                found_characters.append(name)
                print(f"Found single-word character: {name}")
                
    # Remove duplicates while preserving order
    unique_found = []
    for char in found_characters:
        if char not in unique_found:
            unique_found.append(char)
    
    print(f"Detected {len(unique_found)} characters: {', '.join(unique_found) if unique_found else 'none'}")
    return unique_found

@log_entry_exit
def select_best_lora(character_profiles, detected_characters):
    """
    Select the best LoRA based on characters in the scene.
    
    Args:
        character_profiles (list): List of character profile objects
        detected_characters (list): List of character names detected in the scene
    
    Returns:
        tuple: (lora_type, lora_value, character_pair) where:
            - lora_type is "combined" or "individual"
            - lora_value is the LoRA file to use
            - character_pair is a list of the character names (if combined)
    """
    if not detected_characters:
        log_flow("No characters detected in scene, using default LoRA", level="warning")
        if character_profiles:
            # Default to first character's LoRA
            return "individual", character_profiles[0].get("hf_lora", ""), [character_profiles[0].get("name", "")]
        return "none", "", []
    
    # If only one character detected, use their individual LoRA
    if len(detected_characters) == 1:
        char_name = detected_characters[0]
        profile = next((p for p in character_profiles if p.get("name") == char_name), None)
        if profile and profile.get("hf_lora"):
            log_flow(f"Using individual LoRA for character: {char_name}")
            return "individual", profile.get("hf_lora"), [char_name]
    
    # If two or more characters detected, try to find a compatible pair with combined_lora
    if len(detected_characters) >= 2:
        log_flow(f"Multiple characters detected ({len(detected_characters)}), looking for combined LoRA")
        # Check each possible pair
        for i, char1_name in enumerate(detected_characters):
            for char2_name in detected_characters[i+1:]:
                char1 = next((p for p in character_profiles if p.get("name") == char1_name), None)
                char2 = next((p for p in character_profiles if p.get("name") == char2_name), None)
                
                # Check if either character has a combined_lora
                if char1 and char2:
                    if char1.get("combined_lora"):
                        log_flow(f"Using combined LoRA for pair: {char1_name} and {char2_name}")
                        return "combined", char1.get("combined_lora"), [char1_name, char2_name]
                    if char2.get("combined_lora"):
                        log_flow(f"Using combined LoRA for pair: {char1_name} and {char2_name}")
                        return "combined", char2.get("combined_lora"), [char1_name, char2_name]
    
    # Fallback: use the first detected character's individual LoRA
    char_name = detected_characters[0]
    profile = next((p for p in character_profiles if p.get("name") == char_name), None)
    if profile and profile.get("hf_lora"):
        log_flow(f"Fallback: Using individual LoRA for first detected character: {char_name}")
        return "individual", profile.get("hf_lora"), [char_name]
    
    # Last resort: use any available LoRA from character profiles
    for profile in character_profiles:
        if profile.get("hf_lora"):
            log_flow("Last resort: Using individual LoRA from available profiles", level="warning")
            return "individual", profile.get("hf_lora"), [profile.get("name", "Unknown")]
    
    log_flow("No suitable LoRA found", level="warning")
    return "none", "", []

@log_entry_exit
def ensure_character_profiles(state):
    """
    Ensure character profiles exist in the state, fetching from DB if needed.
    
    Args:
        state (StoryState): The current state
        
    Returns:
        tuple: (updated_state, characters_finalized)
        - updated_state is the state with character profiles
        - characters_finalized is True if character profiles were created/updated
    """
    # If we already have character profiles with images, we're good
    if state.character_profiles and any(p.get("reference_image") for p in state.character_profiles):
        log_flow(f"Found {len(state.character_profiles)} existing character profiles with images")
        return state, False
    
    # If we have character descriptions but no profiles, we need to create them
    if state.characters and not state.character_profiles:
        log_flow("Found character descriptions but no profiles - need to create them", level="warning")
        
        # Check if casting_agent module is available
        try:
            from agents.casting import casting_agent
            log_flow("Using casting agent to generate character profiles...")
            
            # Run the casting agent to create profiles
            updated_state = casting_agent(state)
            
            if updated_state.character_profiles:
                log_flow(f"Successfully created {len(updated_state.character_profiles)} character profiles")
                return updated_state, True
            else:
                log_error("Failed to create character profiles with casting agent")
        except ImportError:
            log_error("Could not import casting_agent module")
    
    # If we have neither character profiles nor descriptions, check if we can extract from episodes
    if not state.character_profiles and not state.characters and state.episodes:
        log_flow("No character information found, attempting to extract from episodes...")
        character_names = set()
        
        # Extract character names from episode summaries
        for episode in state.episodes:
            summary = episode.get("summary", "")
            # Use GPT to extract character names from summaries
            try:
                prompt = f"""
Extract the names of all characters mentioned in this text. 
Return only a JSON array of names, nothing else.

Text: {summary}
"""
                log_api_call("OpenAI GPT-3.5-Turbo", {"purpose": "character extraction"})
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                try:
                    import json
                    names = json.loads(response.choices[0].message.content)
                    if isinstance(names, list):
                        for name in names:
                            if isinstance(name, str) and len(name) > 1:
                                character_names.add(name)
                except Exception as e:
                    log_error(f"Error parsing character names", e)
            except Exception as e:
                log_error(f"Error extracting character names from episode", e)
        
        if character_names:
            log_flow(f"Extracted {len(character_names)} character names from episodes: {', '.join(character_names)}")
            # Create simple character descriptions
            state.characters = [f"{name}, A character in the story" for name in character_names]
            
            # Try to use casting agent again with these descriptions
            try:
                from agents.casting import casting_agent
                updated_state = casting_agent(state)
                
                if updated_state.character_profiles:
                    log_flow(f"Successfully created {len(updated_state.character_profiles)} character profiles")
                    return updated_state, True
            except ImportError:
                log_error("Could not import casting_agent module for second attempt")
    
    # Fallback: try to get character profiles from the DB directly
    if state.session_id:
        try:
            log_flow(f"Fetching character profiles from DB for session: {state.session_id}")
            log_db_operation("find", "story_projects", {"session_id": state.session_id})
            from db.models.story_projects import find_project_by_session_id
            
            project = find_project_by_session_id(state.session_id)
            if project and "story_data" in project:
                story_data = project["story_data"]
                if "character_profiles" in story_data and story_data["character_profiles"]:
                    # Update state with profiles from DB
                    state.character_profiles = story_data["character_profiles"]
                    state.character_map = story_data.get("character_map", {})
                    log_flow(f"Found {len(state.character_profiles)} character profiles in DB")
                    return state, False
        except Exception as e:
            log_error("Error fetching from DB", e)
    
    # If all else fails, create dummy character profiles based on raw character descriptions
    if state.characters and not state.character_profiles:
        log_flow("Creating basic character profiles as fallback", level="warning")
        log_db_operation("get", "candidate_characters")
        candidates = get_candidate_characters()
        
        if candidates:
            # Use random candidates for characters
            import random
            profiles = []
            
            for i, char_desc in enumerate(state.characters):
                parts = char_desc.split(",", 1)
                name = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else "A character in the story"
                
                # Find a random candidate with image
                valid_candidates = [c for c in candidates if c.get("image_url")]
                if valid_candidates:
                    candidate = random.choice(valid_candidates)
                    profile = {
                        "name": name,
                        "description": description,
                        "token": f"character-{i+1}",
                        "reference_image": candidate.get("image_url", ""),
                        "hf_lora": candidate.get("hf_lora", ""),
                        "combined_lora": candidate.get("combined_lora", ""),
                        "gender": candidate.get("gender", "")
                    }
                    profiles.append(profile)
            
            if profiles:
                state.character_profiles = profiles
                log_flow(f"Created {len(profiles)} basic profiles with existing candidates")
                return state, True
    
    # No character profiles could be created
    log_flow("No character profiles could be created", level="warning")
    return state, False

@log_entry_exit
def ad_agent(state: StoryState, episode_number=None, scene_number=None, seed=None) -> StoryState:
    print("🎬 Running AD Agent to generate all shot images...")
    print(f"✅ REPLICATE_API_TOKEN is {'SET' if os.getenv('REPLICATE_API_TOKEN') else 'NOT SET'}")
    
    # Track the last seed used for returning to caller
    last_generated_seed = None
    
    # Check if we're targeting a specific episode/scene
    is_specific_request = episode_number is not None and scene_number is not None
    if is_specific_request:
        print(f"🎯 Targeting specific episode {episode_number}, scene {scene_number}" + 
              (f" with seed {seed}" if seed else ""))
    
    # Ensure character profiles exist before proceeding
    state, characters_updated = ensure_character_profiles(state)
    
    # If characters were updated, save them to the database
    if characters_updated and state.session_id:
        try:
            # Update the story project with the new character profiles
            update_story_project_by_session(
                state.session_id, 
                {
                    "character_profiles": state.character_profiles,
                    "character_map": state.character_map or {},
                }
            )
            print("✅ Updated character profiles in database")
        except Exception as e:
            print(f"❌ Error updating character profiles in DB: {e}")
    
    # Initialize storage dictionaries
    ad_prompts = state.ad_prompts.copy() if hasattr(state, "ad_prompts") else {}
    ad_images = state.ad_images.copy() if hasattr(state, "ad_images") else {}
    ad_character_info = state.ad_character_info.copy() if hasattr(state, "ad_character_info") else {}

    scene_background_seeds = {}  # {scene_key: seed}
    finalized_characters = state.character_profiles

    if not finalized_characters:
        print("⚠️ No finalized characters available. Creating scenes without character LoRAs.")
        # Continue with the function, but with no specific LoRAs
    else:
        # Print finalized characters for debug
        print("🧩 Finalized Character Profiles:")
        for char in finalized_characters:
            print(f" - {char['name']} | image: {char.get('reference_image', '')[:30]}... | hf_lora: {char.get('hf_lora', '')} | combined_lora: {char.get('combined_lora', '')}")

    # Get all character names for detection
    character_names = [profile.get("name", "") for profile in finalized_characters]
    print(f"👥 Looking for these characters in scenes: {', '.join(character_names)}")

    # Handle scene scripts if they exist
    if state.scene_scripts:
        print(f"📜 Found {len(state.scene_scripts)} scene scripts to process")
        
        # If targeting a specific scene, only process that one
        if is_specific_request:
            target_scene_key = f"ep{episode_number}_scene{scene_number}"
            if target_scene_key in state.scene_scripts:
                scenes_to_process = {target_scene_key: state.scene_scripts[target_scene_key]}
                print(f"🎯 Found target scene {target_scene_key}, processing only this scene")
            else:
                print(f"⚠️ Target scene {target_scene_key} not found in state.scene_scripts")
                scenes_to_process = {}
        else:
            scenes_to_process = state.scene_scripts

        for scene_key, shots in scenes_to_process.items():
            if not shots:
                continue

            # Initialize first background seed for the scene
            scene_seed = seed if is_specific_request and scene_key == target_scene_key else None

            for idx, shot_info in enumerate(shots):
                shot_text = shot_info.get("shot", "")
                dialogue_text = shot_info.get("dialogue", "")

                if not shot_text:
                    continue

                # Detect characters in this shot
                shot_full_text = f"{shot_text} {dialogue_text}".strip()
                detected_chars = detect_characters_in_text(shot_full_text, character_names)
                print(f"🧠 Scene {scene_key}, Shot {idx+1}: Detected characters: {detected_chars}")
                
                # Select best LoRA based on detected characters
                lora_type, lora_to_use, char_pair = select_best_lora(finalized_characters, detected_chars) 
                
                # Create prompt for GPT to create visual prompt
                prompt_to_gpt = f"""
You are an expert cinematic visual storyteller.

Given this shot description, create a 9:16 image generation prompt in Flux style.

Constraints:
- Ultra-realistic texture
- Cinematic lighting with volumetric shadows
- Soft focus background
- Do NOT mention camera brands

Shot Description:
"{shot_text}"

Dialogue (if any):
"{dialogue_text}"

Characters in shot: {', '.join(detected_chars) if detected_chars else 'None specified'}

Respond ONLY with the image prompt.
"""

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt_to_gpt}],
                    temperature=0.7
                )

                image_prompt = response.choices[0].message.content.strip()

                # Build unique key
                unique_key = f"{scene_key}_shot{idx+1}"
                ad_prompts[unique_key] = image_prompt
                
                # Store character information for this shot
                ad_character_info[unique_key] = {
                    "detected_characters": detected_chars,
                    "lora_type": lora_type,
                    "lora_value": lora_to_use,
                    "character_pair": char_pair
                }

                print(f"🖼️ Generating image for {unique_key}" + 
                      (f" using {lora_type} LoRA: {lora_to_use}" if lora_to_use else " without LoRA"))

                # Set seed: reuse same seed within the scene
                if scene_seed is None:
                    print(f"========== CHARACTER SCENE IMAGE GENERATION ==========")
                    print(f"Shot key: {unique_key}")
                    print(f"Image prompt: {image_prompt}")
                    print(f"Characters in shot: {detected_chars}")
                    print(f"LoRA type: {lora_type}")
                    print(f"LoRA value: {lora_to_use}")
                    print(f"Character pair: {char_pair}")
                    print(f"Generating first image in scene (seed will be auto-generated)")
                    image_url, used_seed = generate_flux_image(image_prompt, hf_lora=lora_to_use)
                    if used_seed is not None:
                        scene_seed = used_seed
                        last_generated_seed = used_seed  # Store for returning to caller
                        print(f"Generated seed {scene_seed} will be used for subsequent shots in this scene")
                else:
                    print(f"========== CHARACTER SCENE IMAGE GENERATION ==========")
                    print(f"Shot key: {unique_key}")
                    print(f"Image prompt: {image_prompt}")
                    print(f"Characters in shot: {detected_chars}")
                    print(f"LoRA type: {lora_type}")
                    print(f"LoRA value: {lora_to_use}")
                    print(f"Character pair: {char_pair}")
                    print(f"Reusing scene seed: {scene_seed}")
                    image_url, _ = generate_flux_image(image_prompt, hf_lora=lora_to_use, seed=scene_seed)
                    last_generated_seed = scene_seed  # Store for returning to caller

                if not image_url:
                    print(f"❌ Failed to generate image for {unique_key}.")
                else:
                    print(f"✅ {unique_key} → {image_url[:50]}...")

                ad_images[unique_key] = image_url
    
    # If no scene_scripts or if we need more images, generate from episodes
    if not ad_images and state.episodes:
        print("📝 Generating episode-based scene images")
        
        # Track seeds for episodes to maintain visual consistency across related images
        episode_seeds = {}  # {episode_num: seed}
        
        # If targeting a specific episode, only process that one
        if is_specific_request:
            if episode_number <= len(state.episodes):
                episodes_to_process = [state.episodes[episode_number-1]]
                episodes_indices = [episode_number-1]
                print(f"🎯 Found target episode {episode_number}, processing only this episode")
            else:
                print(f"⚠️ Target episode {episode_number} not found in state.episodes")
                episodes_to_process = []
                episodes_indices = []
        else:
            episodes_to_process = state.episodes[:3]  # Limit to first 3 episodes
            episodes_indices = range(len(episodes_to_process))
            
        for i, episode in zip(episodes_indices, episodes_to_process):
            episode_title = episode.get("episode_title", f"Episode {i+1}")
            summary = episode.get("summary", "")
            
            # Detect characters in the episode summary
            print(f"\n========== CHARACTER DETECTION FOR EPISODE {i+1} ==========")
            print(f"Episode title: {episode_title}")
            print(f"Episode summary: {summary}")
            print(f"Available character names: {character_names}")
            
            detected_chars = detect_characters_in_text(summary, character_names)
            print(f"🧠 Episode {i+1}: Detected characters: {detected_chars}")
            
            # If no characters detected, try to extract character names directly
            if not detected_chars and finalized_characters:
                print("No characters detected by name matching - trying direct extraction")
                # Use the first 1-2 characters as a fallback
                detected_chars = [finalized_characters[0]['name']]
                if len(finalized_characters) > 1:
                    detected_chars.append(finalized_characters[1]['name'])
                print(f"Using fallback characters: {detected_chars}")
            
            # Select best LoRA for this episode
            print(f"Character profiles available: {len(finalized_characters)}")
            for idx, char in enumerate(finalized_characters):
                print(f"  {idx+1}. {char.get('name')}: LoRA = {char.get('hf_lora', 'None')}, combined = {char.get('combined_lora', 'None')}")
                
            lora_type, lora_to_use, char_pair = select_best_lora(finalized_characters, detected_chars)
            
            # Debug LoRA selection
            print(f"LoRA selection result - Type: {lora_type}, Value: {lora_to_use}, Pair: {char_pair}")
            
            # Ensure we have a valid LoRA to use
            if not lora_to_use and finalized_characters:
                # Force use of first character's LoRA as fallback
                for char in finalized_characters:
                    if char.get('hf_lora'):
                        lora_to_use = char.get('hf_lora')
                        lora_type = "individual"
                        char_pair = [char.get('name')]
                        print(f"No LoRA selected - forcing use of {char.get('name')}'s LoRA: {lora_to_use}")
                        break
            
            # Generate image prompt for this episode
            prompt_to_gpt = f"""
You are an expert visual artist creating scene descriptions for image generation.
Create a detailed visual description for this scene that would be good for generating an image.
Focus on setting, lighting, characters, mood, and visual elements.

Episode: {episode_title}
Summary: {summary}

Characters detected: {', '.join(detected_chars) if detected_chars else 'None specified'}

Keep your description under 100 words and make it visually rich.
"""
            # Get a scene description
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt_to_gpt}],
                temperature=0.7,
                max_tokens=150
            )
            
            scene_description = response.choices[0].message.content.strip()
            
            # Generate the actual image
            if is_specific_request:
                scene_key = f"episode_{episode_number}_scene_{scene_number}"
                print(f"🔑 Using specific scene key format: {scene_key}")
            else:
                scene_key = f"episode_{i+1}_main"  # Adding underscore for consistency
                print(f"🔑 Using general scene key format: {scene_key}")
                
            ad_prompts[scene_key] = scene_description
            
            # Store character information for this episode image
            ad_character_info[scene_key] = {
                "detected_characters": detected_chars,
                "lora_type": lora_type,
                "lora_value": lora_to_use,
                "character_pair": char_pair
            }
            
            # Print lora info
            print(f"🔍 Calling generate_flux_image with scene description: {scene_description[:50]}...")
            print(f"   Using {lora_type} LoRA: {lora_to_use}")
            
            # FORCE IMAGE GENERATION: Use the direct replicate_image_gen without fallbacks
            try:
                # First try standard flux image generation
                print(f"========== EPISODE IMAGE GENERATION ==========")
                print(f"Episode: {episode_title}")
                print(f"Scene description: {scene_description}")
                print(f"Characters: {detected_chars}")
                print(f"LoRA type: {lora_type}")
                print(f"LoRA value: {lora_to_use}")
                print(f"Character pair: {char_pair}")
                
                # Check if we should use a provided seed or a consistent seed for this episode
                if is_specific_request and seed:
                    seed_value = seed
                    print(f"Using provided seed {seed_value} for specific episode/scene request")
                    image_url, _ = generate_flux_image(scene_description, hf_lora=lora_to_use, seed=seed_value)
                    last_generated_seed = seed_value
                elif i in episode_seeds:
                    seed_value = episode_seeds[i]
                    print(f"Using consistent seed {seed_value} for episode {i+1}")
                    image_url, _ = generate_flux_image(scene_description, hf_lora=lora_to_use, seed=seed_value)
                    last_generated_seed = seed_value
                else:
                    print(f"Generating first image for episode {i+1} - seed will be auto-generated")
                    image_url, used_seed = generate_flux_image(scene_description, hf_lora=lora_to_use)
                    # Save seed for consistency if one was returned
                    if used_seed is not None:
                        episode_seeds[i] = used_seed
                        last_generated_seed = used_seed  # Store for returning to caller
                        print(f"Saving seed {used_seed} for episode {i+1} consistency")
                
                if not image_url:
                    # Fallback to manual direct call
                    print("⚠️ Primary image generation failed, trying direct call...")
                    print(f"========== FALLBACK EPISODE IMAGE GENERATION ==========")
                    print(f"Episode: {episode_title}")
                    print(f"Scene description: {scene_description}")
                    print(f"Note: Direct fallback doesn't support LoRA")
                    
                    fallback_input = {
                        "prompt": scene_description,
                        "aspect_ratio": "9:16",
                        "output_format": "jpg",
                        "prompt_strength": 0.8
                    }
                    
                    # Use the same seed if we have one for consistency
                    if is_specific_request and seed:
                        seed_value = seed
                        fallback_input["seed"] = seed_value
                        print(f"Using provided seed {seed_value} in fallback")
                        last_generated_seed = seed_value
                    elif i in episode_seeds:
                        seed_value = episode_seeds[i]
                        fallback_input["seed"] = seed_value
                        print(f"Using consistent seed {seed_value} in fallback")
                        last_generated_seed = seed_value
                        
                    print(f"Fallback input: {fallback_input}")
                    output = replicate.run(
                        "black-forest-labs/flux-1.1-pro",
                        input=fallback_input
                    )
                    
                    print(f"Fallback output: {output}")
                    
                    if isinstance(output, list) and len(output) > 0:
                        image_url = str(output[0])
                        print(f"Using first image from fallback: {image_url}")
                    elif isinstance(output, str):
                        image_url = output
                        print(f"Using string output from fallback: {image_url}")
            except Exception as e:
                print(f"❌ All image generation attempts failed: {e}")
                # Final fallback to placeholder
                image_url = f"https://placehold.co/600x400/png?text={episode_title.replace(' ', '+')}"
                print(f"Using placeholder image due to all failures: {image_url}")
            
            if not image_url:
                print(f"❌ Failed to generate image for {scene_key}.")
            else:
                print(f"✅ {scene_key} → {image_url[:50]}...")
                ad_images[scene_key] = image_url

    new_state = state.copy(update={
        "ad_prompts": ad_prompts,
        "ad_images": ad_images,
        "ad_character_info": ad_character_info,
        "last_generated_seed": last_generated_seed  # Add the last used seed to state
    })

    log_agent_output("AD", new_state)
    return new_state
