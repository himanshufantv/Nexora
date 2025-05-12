import asyncio
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os
import time
import random
import re
import json
from openai import OpenAI

from engine.runner import run_agent
from utils.types import StoryState
from utils.parser import safe_parse_json_string
from utils.functional_logger import log_api_call, log_db_operation, log_error, log_entry_exit

from agents.writer import writer_agent
from agents.director import director_agent
from agents.casting import casting_agent
from agents.ad import ad_agent
from agents.video_design import video_design_agent
from agents.editor import editor_agent
from agents.producer import producer_agent

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["StudioNexora"]
projects = db["story_projects"]

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

agent_map = {
    "Writer": writer_agent,
    "Director": director_agent,
    "Casting": casting_agent,
    "AD": ad_agent,
    "VideoDesign": video_design_agent,
    "Editor": editor_agent,
}

async def run_producer_stream(state: StoryState, session_id: str, user_message: str):
    print(f"Starting producer processing for session {session_id}")
    project = projects.find_one({"session_id": session_id})
    if not project:
        print(f"Session not found: {session_id}")
        return "Error: session not found."      
    print(f"Found project for session {session_id}")
    
    if project.get("story_data"):
        story_data = project.get("story_data")
        print(f"Loaded story data from project")
        state = StoryState(**story_data)
        state.session_id = session_id
    
    # Ensure all StoryState attributes are properly initialized
    state = ensure_state_attributes(state)
    print(f"State initialized with {len(state.episodes)} episodes, {len(state.episode_scripts)} episode scripts")
       
    try:
        print(f"Determining agent to run based on user message")
        agent_to_run = producer_agent(state, user_message)
        print(f"Producer selected agent: {agent_to_run}")

        if not agent_to_run or agent_to_run not in agent_map:
            print(f"Invalid agent {agent_to_run}, defaulting to Writer")
            agent_to_run = "Writer"
        
        # If user is asking for scene images but was routed to VideoDesign, redirect to AD
        if "scene" in user_message.lower() and "image" in user_message.lower() and agent_to_run == "VideoDesign":
            print("Redirecting from VideoDesign to AD for scene images")
            agent_to_run = "AD"
            
        # Special handling for Casting agent to get images
        if agent_to_run == "Casting":
            print("Using Casting agent for image generation")
            
            # Import the character utilities
            try:
                from utils.character_utils import save_characters_to_db
                print(f"Successfully imported character_utils")
            except ImportError:
                print(f"Could not import character_utils")
            
            # Use the actual Casting agent implementation
            print("Calling casting agent")
            updated_state = casting_agent(state)
            
            # Check if we got character profiles with images
            if updated_state.character_profiles:
                print(f"Casting agent returned {len(updated_state.character_profiles)} character profiles with images")
                
                # Format the response as markdown
                response_text = "# Character Visualizations\n\n"
                
                # For each character, add to the markdown response
                for profile in updated_state.character_profiles:
                    char_name = profile.get("name", "Character")
                    image_url = profile.get("reference_image", "")
                    description = profile.get("description", "")
                    
                    print(f"Adding character info for {char_name}")
                    response_text += f"## {char_name}\n\n"
                    
                    if image_url:
                        response_text += f"![{char_name}]({image_url})\n\n"
                    
                    response_text += f"{description}\n\n"
                
                # Additional: Save character profiles to database
                try:
                    if 'save_characters_to_db' in locals():
                        print(f"Saving character profiles to database")
                        log_db_operation("save", "characters", {"session_id": session_id})
                        character_save_result = save_characters_to_db(
                            session_id, 
                            updated_state.character_profiles, 
                            updated_state.character_map
                        )
                        print(f"Character save result: {character_save_result}")
                except Exception as char_save_err:
                    print(f"Error saving characters", char_save_err)
                
                # Update the state in the database
                try:
                    log_db_operation("update", "projects", {"session_id": session_id})
                    projects.update_one(
                        {"session_id": session_id},
                        {"$set": {
                            "story_data": updated_state.model_dump(),
                            "updated_at": datetime.utcnow()
                        }}
                    )
                    print(f"Updated state in database with character images and details")
                except Exception as db_err:
                    print(f"Database update error", db_err)
                
                return response_text
            else:
                # Fallback to description generation if no images are found
                print(f"WARNING: No character profiles with images generated, falling back to text descriptions")
                prompt = f"""
You are helping visualize characters for a story.

Based on these character descriptions, provide detailed visual descriptions:
{chr(10).join(state.characters)}

For each character, describe their physical appearance, age, clothing style, and distinguishing features.
Format your response in markdown with appropriate headers.
"""
                log_api_call("OpenAI Chat Completion")
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content
                return response_text
        
        # Special handling for AD agent to generate scene images
        elif agent_to_run == "AD":
            print(f"Using AD agent for scene image generation")
            
            # Check if the request is for a specific scene
            is_specific_scene = False
            scene_number = 1
            episode_number = 1
            
            # Parse the user message to extract scene and episode numbers
            if "scene" in user_message.lower():
                is_specific_scene = True
                # Try to extract episode and scene numbers
                episode_match = re.search(r'episode\s*(\d+)', user_message.lower())
                scene_match = re.search(r'scene\s*(\d+)', user_message.lower())
                
                if episode_match:
                    episode_number = int(episode_match.group(1))
                    print(f"Detected request for episode {episode_number}")
                
                if scene_match:
                    scene_number = int(scene_match.group(1))
                    print(f"Detected request for scene {scene_number}")
                
                print(f"Will generate images for episode {episode_number}, scene {scene_number}")
                
                # Create a scene key to track seeds
                scene_key = f"episode_{episode_number}_scene_{scene_number}"
                
                # Check if we already have a seed for this scene
                if hasattr(state, "scene_seeds") and scene_key in state.scene_seeds:
                    existing_seed = state.scene_seeds[scene_key]
                    print(f"Found existing seed {existing_seed} for {scene_key}")
                else:
                    print(f"No existing seed found for {scene_key}, will generate a new one")
            
            # Check if we have episode scripts or scene scripts
            has_episode_scripts = hasattr(state, "episode_scripts") and state.episode_scripts
            has_scene_scripts = hasattr(state, "scene_scripts") and state.scene_scripts
            
            if not has_episode_scripts and not has_scene_scripts:
                # If no existing scenes, create quick scene prompts from episodes
                has_episodes = hasattr(state, "episodes") and state.episodes and len(state.episodes) > 0
                if has_episodes:
                    # Create scene images based on episode summaries
                    print(f"Generating scene visualizations from episode summaries")
                    response_text = "# Scene Visualizations\n\n"
                    
                    # Call AD agent to generate images
                    try:
                        # If specific scene requested, pass scene info to AD agent
                        if is_specific_scene:
                            print(f"Calling AD agent with specific scene request: episode {episode_number}, scene {scene_number}")
                            # Check if we have a seed to pass
                            if hasattr(state, "scene_seeds") and scene_key in state.scene_seeds:
                                seed_value = state.scene_seeds[scene_key]
                                print(f"Passing existing seed {seed_value} for consistent imagery")
                                updated_state = ad_agent(state, episode_number=episode_number, scene_number=scene_number, seed=seed_value)
                            else:
                                updated_state = ad_agent(state, episode_number=episode_number, scene_number=scene_number)
                                # Check if a seed was generated and save it
                                if hasattr(updated_state, "ad_images") and scene_key in updated_state.ad_images:
                                    # Try to extract the seed used from the agent's response
                                    if not hasattr(updated_state, "scene_seeds"):
                                        updated_state.scene_seeds = {}
                                    if hasattr(updated_state, "last_generated_seed") and updated_state.last_generated_seed:
                                        updated_state.scene_seeds[scene_key] = updated_state.last_generated_seed
                                        print(f"Saved new seed {updated_state.last_generated_seed} for {scene_key}")
                        else:
                            updated_state = ad_agent(state)
                            
                        # Format the AD agent response as markdown
                        if is_specific_scene:
                            # For specific scene request, only show that scene
                            scene_key = f"episode_{episode_number}"
                            if hasattr(updated_state, "ad_images") and scene_key in updated_state.ad_images:
                                ep_title = f"Episode {episode_number}, Scene {scene_number}"
                                response_text += f"## Scene from {ep_title}\n\n"
                                response_text += f"![Scene from {ep_title}]({updated_state.ad_images[scene_key]})\n\n"
                                
                                if hasattr(updated_state, "ad_prompts") and scene_key in updated_state.ad_prompts:
                                    response_text += updated_state.ad_prompts[scene_key] + "\n\n"
                        else:
                            # For general request, show multiple episodes/scenes
                            episode_count = min(len(state.episodes), 3) if state.episodes else 0
                            for j in range(episode_count):
                                if j < len(state.episodes):  # Ensure we don't go out of bounds
                                    ep = state.episodes[j]
                                    ep_title = ep.get("episode_title", f"Episode {j+1}")
                                    scene_key = f"episode_{j+1}"
                                    
                                    if hasattr(updated_state, "ad_images") and scene_key in updated_state.ad_images:
                                        response_text += f"## Scene from {ep_title}\n\n"
                                        response_text += f"![Scene from {ep_title}]({updated_state.ad_images[scene_key]})\n\n"
                                    
                                    if hasattr(updated_state, "ad_prompts") and scene_key in updated_state.ad_prompts:
                                        response_text += updated_state.ad_prompts[scene_key] + "\n\n"
                    except Exception as e:
                        print(f"Error calling AD agent: {e}")
                        response_text += f"Error generating scene visualizations: {str(e)}\n\n"
                        updated_state = state  # Use original state if there's an error
                        
                        # Save updated state to the database
                        try:
                            projects.update_one(
                                {"session_id": session_id},
                                {"$set": {
                                    "story_data": updated_state.model_dump(),
                                    "updated_at": datetime.utcnow()
                                }}
                            )
                            print(f"Updated state in database with scene images")
                        except Exception as db_err:
                            print(f"Database update error: {db_err}")
                            
                        return response_text
                    else:
                        # No episodes to generate scene images from
                        return "No episodes found to generate scene visualizations. Please create a story outline first."
            else:
                # Use existing scenes to generate images
                if is_specific_scene:
                    print(f"Calling AD agent with specific scene request: episode {episode_number}, scene {scene_number}")
                    # Check if we have a seed to pass
                    scene_key = f"episode_{episode_number}_scene_{scene_number}"
                    if hasattr(state, "scene_seeds") and scene_key in state.scene_seeds:
                        seed_value = state.scene_seeds[scene_key]
                        print(f"Passing existing seed {seed_value} for consistent imagery")
                        updated_state = ad_agent(state, episode_number=episode_number, scene_number=scene_number, seed=seed_value)
                    else:
                        updated_state = ad_agent(state, episode_number=episode_number, scene_number=scene_number)
                        # Check if a seed was generated and save it
                        if hasattr(updated_state, "ad_images") and scene_key in updated_state.ad_images:
                            # Save the seed if it was returned
                            if not hasattr(updated_state, "scene_seeds"):
                                updated_state.scene_seeds = {}
                            if hasattr(updated_state, "last_generated_seed") and updated_state.last_generated_seed:
                                updated_state.scene_seeds[scene_key] = updated_state.last_generated_seed
                                print(f"Saved new seed {updated_state.last_generated_seed} for {scene_key}")
                else:
                    updated_state = ad_agent(state)
                
                # Format response as markdown
                response_text = "# Scene Visualizations\n\n"
                
                # Add existing scene images if available
                if is_specific_scene:
                    # For specific scene request, only show that scene
                    scene_key = f"episode_{episode_number}_scene_{scene_number}"
                    if hasattr(updated_state, "ad_images") and scene_key in updated_state.ad_images:
                        scene_title = f"Episode {episode_number}, Scene {scene_number}"
                        response_text += f"## {scene_title}\n\n"
                        response_text += f"![{scene_title}]({updated_state.ad_images[scene_key]})\n\n"
                        
                        if scene_key in updated_state.ad_prompts:
                            response_text += updated_state.ad_prompts[scene_key] + "\n\n"
                else:
                    # Show all available scene images
                    for scene_key, image_url in updated_state.ad_images.items():
                        scene_title = scene_key.replace("_", " ").title()
                        response_text += f"## {scene_title}\n\n"
                        response_text += f"![{scene_title}]({image_url})\n\n"
                        
                        if scene_key in updated_state.ad_prompts:
                            response_text += updated_state.ad_prompts[scene_key] + "\n\n"
                
                return response_text
        
        # For Writer agent and other agents
        else:
            # Handle different agent types
            agent_name = agent_to_run
            profile_key = "english_romantic"
            
            if "::" in agent_to_run:
                agent_name, profile_key = agent_to_run.split("::")
            
            # Get the agent function
            agent_fn = agent_map.get(agent_name)
            
            if not agent_fn:
                return f"Error: Agent {agent_name} not found."
            
            # Call the agent function with appropriate arguments
            if agent_name == "Writer":
                updated_state = agent_fn(state, user_message, profile_key)
            else:
                updated_state = agent_fn(state)
            
            # Process agent response based on agent type
            if agent_name == "Writer":
                # Format Writer agent response as markdown
                # Only display the title without extra headers
                response_text = f"# {updated_state.title}\n\n"
                
                if updated_state.episodes and not updated_state.episode_scripts and not updated_state.scene_scripts:
                    # Series overview - but skip the title and logline headers since we already included the title
                    if updated_state.characters:
                        response_text += "## Characters\n\n"
                        for char in updated_state.characters:
                            response_text += f"- {char}\n"
                        response_text += "\n"
                    
                    response_text += "## Episodes\n\n"
                    for ep in updated_state.episodes:
                        ep_num = ep.get("episode_number", 0)
                        ep_title = ep.get("episode_title", f"Episode {ep_num}")
                        ep_summary = ep.get("summary", "No summary available")
                        
                        response_text += f"### Episode {ep_num}: {ep_title}\n\n"
                        response_text += f"{ep_summary}\n\n"
                
                elif updated_state.episode_scripts:
                    # Episode scenes
                    response_text += "## Episode Scenes\n\n"
                    
                    # Check if we have the new structured format
                    if hasattr(updated_state, "structured_scenes") and updated_state.structured_scenes:
                        for ep_num, scenes in updated_state.structured_scenes.items():
                            response_text += f"### Episode {ep_num}\n\n"
                            
                            for scene in scenes:
                                scene_num = scene.get("scene_number", 0)
                                scene_title = scene.get("title", f"Scene {scene_num}")
                                scene_description = scene.get("description", "")
                                
                                # Remove the redundant "Scene X:" prefix if it exists
                                scene_prefix_pattern = re.match(r'^Scene\s*\d+\s*:\s*(.*)', scene_description, re.IGNORECASE)
                                if scene_prefix_pattern:
                                    scene_description = scene_prefix_pattern.group(1).strip()
                                
                                response_text += f"#### Scene {scene_num}: {scene_title}\n\n"
                                response_text += f"{scene_description}\n\n"
                    # Fall back to old format if structured scenes not available
                    else:
                        for ep_num, scenes in updated_state.episode_scripts.items():
                            response_text += f"### Episode {ep_num}\n\n"
                            
                            for i, scene in enumerate(scenes):
                                response_text += f"#### Scene {i+1}\n\n"
                                response_text += f"{scene}\n\n"
                
                elif updated_state.scene_scripts:
                    # Scene details
                    response_text += "## Scene Breakdown\n\n"
                    
                    for scene_key, shots in updated_state.scene_scripts.items():
                        match = re.match(r'ep(\d+)_scene(\d+)', scene_key)
                        if match:
                            ep_num, scene_num = match.groups()
                            response_text += f"### Episode {ep_num}, Scene {scene_num}\n\n"
                        else:
                            response_text += f"### {scene_key}\n\n"
                        
                        for i, shot in enumerate(shots):
                            response_text += f"#### Shot {i+1}\n\n"
                            
                            if "shot" in shot:
                                response_text += f"**Visual:** {shot['shot']}\n\n"
                            
                            if "dialogue" in shot:
                                response_text += f"**Dialogue:** {shot['dialogue']}\n\n"
                else:
                    # Default response for other cases
                    if hasattr(updated_state, 'last_agent_output') and updated_state.last_agent_output:
                        response_text += updated_state.last_agent_output
                    else:
                        response_text += f"The {agent_name} agent has processed your request.\n\n"
            
            # Save updated state to the database
            try:
                projects.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "story_data": updated_state.model_dump(),
                        "updated_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                print(f"Updated state saved to database for {agent_name} agent")
            except Exception as db_err:
                print(f"Database update error: {db_err}")
            
            return response_text

    except Exception as e:
        print(f"Error in run_producer_stream: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error processing request: {str(e)}"

@log_entry_exit
def run_engine(state: StoryState, user_message: str):
    print(f"Starting engine run with message: {user_message[:50]}...")
    
    # Determine which agent to run using the producer
    print("Calling producer agent to determine next agent")
    agent_to_run = producer_agent(state, user_message)
    
    if not agent_to_run or agent_to_run not in agent_map:
        print(f"Invalid agent '{agent_to_run}', defaulting to Writer")
        agent_to_run = "Writer"
    
    print(f"Producer selected agent: {agent_to_run}")
    
    # Run the selected agent
    updated_state = agent_map[agent_to_run](state)
    
    return updated_state, agent_to_run

def ensure_state_attributes(state: StoryState) -> StoryState:
    """Ensure all StoryState attributes are properly initialized to avoid None errors"""
    # User context
    if not hasattr(state, "user_prompt") or state.user_prompt is None:
        state.user_prompt = ""
    if not hasattr(state, "story_prompt") or state.story_prompt is None:
        state.story_prompt = ""
    if not hasattr(state, "producer_notes") or state.producer_notes is None:
        state.producer_notes = ""
    if not hasattr(state, "session_memory") or state.session_memory is None:
        state.session_memory = []
    if not hasattr(state, "last_agent_output") or state.last_agent_output is None:
        state.last_agent_output = None
    if not hasattr(state, "ad_prompts") or state.ad_prompts is None:
        state.ad_prompts = {}
    if not hasattr(state, "ad_images") or state.ad_images is None:
        state.ad_images = {}
    if not hasattr(state, "ad_character_info") or state.ad_character_info is None:
        state.ad_character_info = {}

    # High-level metadata
    if not hasattr(state, "title") or state.title is None:
        state.title = ""
    if not hasattr(state, "logline") or state.logline is None:
        state.logline = ""
    if not hasattr(state, "genre") or state.genre is None:
        state.genre = ""
    if not hasattr(state, "style") or state.style is None:
        state.style = ""
    if not hasattr(state, "writer_profile") or state.writer_profile is None:
        state.writer_profile = ""

    # Characters
    if not hasattr(state, "characters") or state.characters is None:
        state.characters = []
    if not hasattr(state, "character_map") or state.character_map is None:
        state.character_map = {}
    if not hasattr(state, "character_profiles") or state.character_profiles is None:
        state.character_profiles = []
    if not hasattr(state, "structured_characters") or state.structured_characters is None:
        state.structured_characters = []

    # Series Structure
    if not hasattr(state, "script_outline") or state.script_outline is None:
        state.script_outline = ""
    if not hasattr(state, "three_act_structure") or state.three_act_structure is None:
        state.three_act_structure = {}

    # Episode-level story
    if not hasattr(state, "episodes") or state.episodes is None:
        state.episodes = []
    if not hasattr(state, "episode_scripts") or state.episode_scripts is None:
        state.episode_scripts = {}
    if not hasattr(state, "scene_scripts") or state.scene_scripts is None:
        state.scene_scripts = {}
    if not hasattr(state, "structured_scenes") or state.structured_scenes is None:
        state.structured_scenes = {}

    # Visual Assets
    if not hasattr(state, "scenes") or state.scenes is None:
        state.scenes = []
    if not hasattr(state, "scene_image_prompts") or state.scene_image_prompts is None:
        state.scene_image_prompts = []
    if not hasattr(state, "video_clips") or state.video_clips is None:
        state.video_clips = []
    if not hasattr(state, "session_id") or state.session_id is None:
        state.session_id = ""

    # Structured Data Storage
    if not hasattr(state, "structured_data") or state.structured_data is None:
        state.structured_data = {}

    return state
