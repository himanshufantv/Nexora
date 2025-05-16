# agents/writer.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.types import StoryState
from utils.parser import safe_parse_json_string
from utils.logger import log_agent_output
from typing import Any
from datetime import datetime
from utils.db import get_writer_profiles_collection

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load writer profile from JSON
def load_writer_profile(profile_key: str) -> dict:
    try:
        collection = get_writer_profiles_collection()
        profile = collection.find_one({"profile_key": profile_key})
        if not profile:
            print(f"Writer profile '{profile_key}' not found in database, using default profile")
            return get_default_profile(profile_key)
        
        # Clean Mongo _id field
        profile.pop("_id", None)
        return profile
    except Exception as e:
        print(f"Error loading writer profile from database: {e}")
        return get_default_profile(profile_key)

def get_default_profile(profile_key: str) -> dict:
    """Return a default writer profile when database access fails"""
    default_profiles = {
        "english_romantic": {
            "profile_key": "english_romantic",
            "language": "English",
            "tone": "Emotional, heartfelt, and intimate",
            "style_note": "Focus on character relationships and emotional resonance",
            "genre": "Romantic drama",
            "system_prompt": "You are a talented screenwriter specializing in romantic dramas. Create compelling, emotionally resonant stories with complex characters and meaningful relationship arcs."
        },
        "hindi_thriller": {
            "profile_key": "hindi_thriller",
            "language": "Hindi (with English script)",
            "tone": "Suspenseful, tense, and mysterious",
            "style_note": "Focus on plot twists and gradual revelation of secrets",
            "genre": "Thriller",
            "system_prompt": "You are a renowned Hindi thriller writer known for creating suspenseful narratives with unexpected twists. Your stories keep audiences on the edge of their seats."
        },
        "english_action": {
            "profile_key": "english_action",
            "language": "English",
            "tone": "Dynamic, intense, and exciting",
            "style_note": "Focus on fast-paced action sequences and high-stakes scenarios",
            "genre": "Action",
            "system_prompt": "You are an accomplished screenwriter specializing in action sequences. Create thrilling stories with dynamic characters, elaborate action set pieces, and high-stakes conflicts."
        }
    }
    
    # Return the requested profile or english_romantic as fallback
    return default_profiles.get(profile_key, default_profiles["english_romantic"])

# Main Writer Agent
def writer_agent(state: StoryState, user_message: str, profile_key: str = "english_romantic") -> StoryState:
    try:
        # Ensure state has necessary attributes
        if not hasattr(state, "episodes") or state.episodes is None:
            state.episodes = []
        
        if not hasattr(state, "episode_scripts") or state.episode_scripts is None:
            state.episode_scripts = {}
            
        if not hasattr(state, "scene_scripts") or state.scene_scripts is None:
            state.scene_scripts = {}
            
        if not hasattr(state, "characters") or state.characters is None:
            state.characters = []
        
        # Load profile safely
        profile = load_writer_profile(profile_key)
        system_prompt = profile["system_prompt"]
        language = profile["language"]
        tone = profile["tone"]
        style = profile["style_note"]
        genre = profile["genre"]

        # Debug logging
        print(f"Writer agent processing request: '{user_message}'")
        print(f"Current state has {len(state.episodes)} episodes, {len(state.episode_scripts)} episode scripts, {len(state.scene_scripts)} scene scripts")

        # Determine which function to call based on the user message
        if "scene" in user_message.lower():
            print(f"Routing to generate_scene_script")
            return generate_scene_script(state, user_message, system_prompt)
        elif "episode" in user_message.lower():
            print(f"Routing to generate_episode_script")
            return generate_episode_script(state, user_message, system_prompt)
        else:
            print(f"Routing to generate_series_synopsis")
            return generate_series_synopsis(state, user_message, system_prompt, language, tone, style, genre)
    except Exception as e:
        print(f"Error in writer_agent: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Create a minimal valid response with string keys for MongoDB compatibility
        if not hasattr(state, "episode_scripts") or state.episode_scripts is None:
            state.episode_scripts = {"1": ["Default scene created due to an error."]}
        if not hasattr(state, "last_agent_output"):
            state.last_agent_output = "Error in writer agent, created default state."
        
        print(f"Returning default state due to error")
        return state

# Level 1: Series Synopsis
def generate_series_synopsis(state, user_message, system_prompt, language, tone, style, genre):
    prompt = f"""
{system_prompt}

Language: {language}
Genre: {genre}
Tone: {tone}
Style Guide: {style}
Instructions:- 
"Only 2-3 characters per episode",
"Only 1-2 locations per episode",
Write a 10-episode series synopsis:
"{user_message}"

Respond in this exact JSON format with no additional text or explanation:
```json
{{
  "series_title": "Title of the Series",
  "logline": "One-sentence description of the series",
  "characters": ["Name, description", "..."],
  "episodes": [
    {{ "episode_number": 1, "episode_title": "Title of episode 1", "summary": "Detailed summary of episode 1" }},
    {{ "episode_number": 2, "episode_title": "Title of episode 2", "summary": "Detailed summary of episode 2" }},
    ...and so on for all 10 episodes
  ]
}}
```

Ensure each episode has:
1. A properly numbered episode_number field (1-10)
2. A clear, distinct episode_title
3. Spaces between words in titles (not camelCase or runTogether)
4. A complete summary
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    
    content = response.choices[0].message.content
    
    # Extract JSON from code blocks if present
    import re
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if json_match:
        content_to_parse = json_match.group(1).strip()
    else:
        content_to_parse = content
    
    parsed = safe_parse_json_string(content_to_parse)
    
    # Ensure each episode has a proper episode_number
    if "episodes" in parsed:
        for i, episode in enumerate(parsed["episodes"]):
            if "episode_number" not in episode:
                episode["episode_number"] = i + 1
    
    new_state = state.copy(update={
        "story_prompt": user_message,
        "title": parsed.get("series_title", ""),
        "logline": parsed.get("logline", ""),
        "characters": parsed.get("characters", []),
        "episodes": parsed.get("episodes", [])
    })
    new_state.last_agent_output = parsed
    log_agent_output("Writer", new_state)
    return new_state

# Level 2: Episode Breakdown
def generate_episode_script(state, user_message, system_prompt):
    try:
        # Import needed modules
        import re
        from utils.parser import safe_parse_json_string
        
        # Check if user is requesting just episodes (without scenes)
        is_episodes_only_request = "episodes" in user_message.lower() and not "scene" in user_message.lower()
        
        # If this is a general "generate episodes" request, create all 10 episodes
        if is_episodes_only_request:
            print(f"Generating all episodes without scenes")
            
            # Get synopsis and character information if available to ensure consistency
            synopsis = state.logline if hasattr(state, "logline") and state.logline else ""
            
            # If no synopsis in logline, check title as well
            if not synopsis and hasattr(state, "title") and state.title:
                synopsis = state.title
            
            # Extract character information from state if available
            character_info = []
            if hasattr(state, "characters") and state.characters:
                character_info = state.characters
                print(f"Using {len(character_info)} existing characters")
                
            # Modify the episodes prompt to include character information
            character_names = []
            if character_info:
                for char in character_info:
                    if isinstance(char, str) and ',' in char:
                        name = char.split(',')[0].strip()
                        character_names.append(name)
            
            # If we don't have character names but have a synopsis, try to extract them
            if not character_names and synopsis:
                try:
                    # Try to extract character names from the synopsis
                    import json
                    from openai import OpenAI
                    
                    extract_prompt = f"""
Extract the main characters' names from this synopsis. Return ONLY a JSON array of names, nothing else:

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
                        character_names = json.loads(char_names_text)
                        print(f"Extracted character names from synopsis: {character_names}")
                    except:
                        # If JSON parsing fails, try a simple split
                        character_names = [name.strip(' "\'[]') for name in char_names_text.split(",")]
                except Exception as e:
                    print(f"Error extracting character names from synopsis: {e}")
            
            # Define character constraint if we have character names
            character_constraint = ""
            if character_names:
                character_constraint = f"""
IMPORTANT: Use ONLY these character names in your episodes: {', '.join(character_names)}
DO NOT invent or use any other main character names not in this list.
"""
            
            episodes_prompt = f"""
{system_prompt}

Create a 10-episode series for a dramatic series. For each episode, provide:
1. The episode number (1-10)
2. A compelling episode title
3. A detailed summary (1-2 paragraphs) of what happens in the episode

Synopsis of the story: {synopsis}
{character_constraint}

Episode summaries should be self-contained but connect to form a cohesive season arc.
DO NOT include detailed scenes - just the high-level episode summaries.

Return your response in this JSON format:
{{
  "episodes": [
    {{
      "episode_number": 1,
      "episode_title": "Title of episode 1",
      "summary": "Detailed summary of episode 1"
    }},
    {{
      "episode_number": 2,
      "episode_title": "Title of episode 2",
      "summary": "Detailed summary of episode 2"
    }},
    ...and so on for all 10 episodes
  ]
}}
"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": episodes_prompt}],
                temperature=0.7
            )
            
            response_content = response.choices[0].message.content
            
            # Extract JSON if in code block
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
            if json_match:
                episodes_json = json_match.group(1).strip()
            else:
                episodes_json = response_content
                
            parsed_response = safe_parse_json_string(episodes_json)
            
            # Update the episodes in the state
            if isinstance(parsed_response, dict) and "episodes" in parsed_response:
                new_episodes = parsed_response["episodes"]
                
                # Ensure all episodes have proper episode_number
                for i, episode in enumerate(new_episodes):
                    if "episode_number" not in episode:
                        episode["episode_number"] = i + 1
                
                # Update state with all episodes
                new_state = state.copy(update={
                    "episodes": new_episodes
                })
                
                # Set the output for downstream processes
                new_state.last_agent_output = f"# Series Episodes\n\n" + "\n\n".join([
                    f"## Episode {ep['episode_number']}: {ep['episode_title']}\n\n{ep['summary']}"
                    for ep in new_episodes
                ])
                
                log_agent_output("Writer", new_state)
                return new_state
        
        # If it's a request for specific episode script with scenes...
        # Try to extract episode number from user message
        episode_digits = "".join(filter(str.isdigit, user_message))
        episode_number = int(episode_digits) if episode_digits else 1
        
        # Safety check: ensure episode number is valid
        if not state.episodes or episode_number < 1 or (episode_number > len(state.episodes) if state.episodes else True):
            print(f"Episode {episode_number} not found, generating a new episode")
            
            # Generate a new episode if none exist
            episode_prompt = f"""
{system_prompt}

Create a detailed episode for a dramatic series. 
This should be Episode {episode_number}.

Return your response in this JSON format:
{{
  "episode_number": {episode_number},
  "episode_title": "Title of the episode",
  "summary": "Detailed summary of the episode"
}}
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": episode_prompt}],
                temperature=0.7
            )
            
            episode_content = response.choices[0].message.content
            
            # Extract JSON if in code block
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', episode_content)
            if json_match:
                episode_json = json_match.group(1).strip()
            else:
                episode_json = episode_content
            
            # Parse the episode
            episode = safe_parse_json_string(episode_json)
            
            # Add the new episode
            state.episodes.append(episode)
            episode_number = len(state.episodes)  # Update episode number
            
            title = episode.get("episode_title", f"Episode {episode_number}")
            summary = episode.get("summary", "A new episode in the series.")
        else:
            # Use existing episode
            episode = state.episodes[episode_number - 1] if state.episodes and len(state.episodes) >= episode_number else None
            
            if not episode:
                # Fallback if episode is still not found
                episode = {
                    "episode_number": episode_number,
                    "episode_title": f"Episode {episode_number}",
                    "summary": "A new adventure begins."
                }
                
            title = episode.get("episode_title", f"Episode {episode_number}")
            summary = episode.get("summary", "A new episode in the series.")

        # Generate scenes for the episode
        scenes_prompt = f"""
{system_prompt}

Break this episode into 6-8 detailed scenes for a Hindi romantic drama series.
Focus on the adventures of the couple in Delhi.
For each scene, provide:
1. A descriptive title
2. A detailed description (2-3 paragraphs) showing the couple's journey and emotions

Episode Title: {title}
Episode Summary: {summary}

Format each scene like this:
Scene 1: [Scene Title] - [Full detailed description that includes setting, characters, actions, and dialogue]
Scene 2: [Scene Title] - [Full detailed description]
etc.

Each scene description should be rich, evocative, and complete - no placeholders.
"""

        # Add character information to the prompt if available
        character_names = []
        
        # Try to extract character information from character_profiles
        if hasattr(state, "character_profiles") and state.character_profiles:
            print(f"Found {len(state.character_profiles)} character profiles to use in episode script")
            for profile in state.character_profiles:
                name = profile.get("name", "")
                if name:
                    character_names.append(name)
                    scenes_prompt += f"\nCharacter: {name}"
        
        # If no profiles found, try to get from characters list
        elif hasattr(state, "characters") and state.characters:
            print(f"Found {len(state.characters)} characters to use in episode script")
            for char in state.characters:
                if isinstance(char, str) and ',' in char:
                    name = char.split(',')[0].strip()
                    character_names.append(name)
                    scenes_prompt += f"\nCharacter: {name}"
        
        # If any character names were found, add a constraint to the prompt
        if character_names:
            print(f"Using character names in episode script: {', '.join(character_names)}")
            scenes_prompt += f"""

IMPORTANT: Use ONLY these character names in your scenes: {', '.join(character_names)}
"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": scenes_prompt}],
            temperature=0.7
        )
        
        # Parse the response - expect a list of strings or simple text format with scene markers
        response_content = response.choices[0].message.content
        
        # We'll handle both structured and unstructured formats
        # First try to extract scenes based on "Scene X:" markers
        import re
        scenes = re.findall(r'Scene\s+\d+:\s*(.*?)(?=Scene\s+\d+:|$)', response_content, re.DOTALL)
        
        if not scenes:
            # Try to parse JSON if the response is in that format
            try:
                parsed_response = safe_parse_json_string(response_content)
                if isinstance(parsed_response, list):
                    scenes = parsed_response
                elif isinstance(parsed_response, dict) and "scenes" in parsed_response:
                    scenes = [f"{scene.get('title', '')}: {scene.get('description', '')}" 
                             for scene in parsed_response["scenes"]]
            except:
                # Fallback - use the original response as a single scene
                scenes = [response_content]
        
        # Clean up the scenes (remove extra whitespace, etc.)
        scenes = [scene.strip() for scene in scenes]
        
        # Create structured scenes with rich content
        structured_scenes = []
        for i, scene_text in enumerate(scenes):
            scene_num = i + 1
            
            # Try to extract a title if the scene has a format like "Title - Description"
            title_match = re.match(r'^([^-:]+)[-:](.*)', scene_text)
            if title_match:
                scene_title = title_match.group(1).strip()
                scene_desc = title_match.group(2).strip()
            else:
                # Just use the first few words as title
                words = scene_text.split()
                scene_title = " ".join(words[:3]) + "..."
                scene_desc = scene_text
            
            structured_scenes.append({
                "scene_number": scene_num,
                "title": scene_title,
                "description": scene_desc
            })
        
        # Update both formats in the state
        episode_str = str(episode_number)
        structured_scenes_dict = {}
        if hasattr(state, "structured_scenes"):
            structured_scenes_dict = state.structured_scenes
            
        # Format the scenes for the old format (string list)
        old_format_scenes = []
        for scene in structured_scenes:
            scene_num = scene.get("scene_number", 0)
            scene_title = scene.get("title", "")
            scene_desc = scene.get("description", "")
            old_format_scenes.append(f"Scene {scene_num}: {scene_title} - {scene_desc}")
            
        new_state = state.copy(update={
            "episode_scripts": {
                **state.episode_scripts,
                episode_str: old_format_scenes
            },
            "structured_scenes": {
                **structured_scenes_dict,
                episode_str: structured_scenes
            }
        })
        
        # Set the output for downstream processes
        new_state.last_agent_output = structured_scenes
        log_agent_output("Writer", new_state)
        return new_state
    except Exception as e:
        print(f"Error in generate_episode_script: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Create a minimal valid response
        if not hasattr(state, "episode_scripts"):
            state.episode_scripts = {}
            
        episode_number = 1
        state.episode_scripts["1"] = ["Default scene created due to an error."]  # Use string key
        state.last_agent_output = "Error generating episode script, created a default scene instead."
        
        return state

# Level 3: Scene Breakdown
def generate_scene_script(state, user_message, system_prompt):
    try:
        # Import needed modules
        import re
        from utils.parser import safe_parse_json_string
        
        # Attempt to extract numbers
        nums = [int(s) for s in user_message.split() if s.isdigit()]
        
        if len(nums) == 1:
            episode_number = 1
            scene_number = nums[0]
        elif len(nums) >= 2:
            episode_number, scene_number = nums[0], nums[1]
        else:
            # Default values if no numbers found
            episode_number = 1
            scene_number = 1
            print(f"No scene/episode numbers specified, defaulting to episode 1, scene 1")

        # Convert to string keys for MongoDB compatibility
        episode_str = str(episode_number)
        
        # Initialize episode_scripts if not present
        if not hasattr(state, "episode_scripts") or not state.episode_scripts:
            state.episode_scripts = {"1": ["Default scene for testing purposes."]}
        
        # Ensure the episode exists
        if episode_str not in state.episode_scripts:
            print(f"Episode {episode_number} not found in episode_scripts, defaulting to episode 1")
            episode_number = 1
            episode_str = "1"
            # If no episodes at all, create a default
            if not state.episode_scripts:
                state.episode_scripts = {"1": ["Default scene for testing purposes."]}
        
        scene_list = state.episode_scripts.get(episode_str, [])
        
        # Ensure the scene exists
        if scene_number > len(scene_list) or scene_number < 1:
            print(f"Scene {scene_number} does not exist in Episode {episode_number}, defaulting to scene 1")
            scene_number = 1
            # If the episode exists but has no scenes, add a default scene
            if not scene_list:
                state.episode_scripts[episode_str] = ["Default scene for testing purposes."]
                scene_list = state.episode_scripts[episode_str]

        scene_text = scene_list[scene_number - 1]

        prompt = f"""
{system_prompt}

Break this scene into cinematic shots:
only 1-2 characters per shot
Scene: {scene_text}

Return format:
[
  {{ "shot": "...", "dialogue": "..." }},
  ...
]
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85
        )
        
        # Parse the shot list
        shot_list = safe_parse_json_string(response.choices[0].message.content)
        scene_key = f"ep{episode_number}_scene{scene_number}"
        new_state = state.copy(update={
            "scene_scripts": {
                **state.scene_scripts,
                scene_key: shot_list
            }
        })
        new_state.last_agent_output = shot_list
        log_agent_output("Writer", new_state)
        return new_state
    except Exception as e:
        print(f"Error in generate_scene_script: {e}")
        # Return a default response with the error message
        state.last_agent_output = f"Error generating scene script: {str(e)}"
        return state
