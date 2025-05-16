from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from utils.types import StoryState
from engine.runner import run_agent
from openai import OpenAI
from typing import Optional, Dict, Any, List

import os
import uuid
import json
import re
import asyncio

from nexora_engine import run_producer_stream
from agents.producer import producer_agent
from agents.writer import writer_agent
from agents.casting import casting_agent
from agents.ad import ad_agent
from agents.director import director_agent
from agents.video_design import video_design_agent
from agents.editor import editor_agent
from agents.storyboard import storyboard_agent

load_dotenv()

router = APIRouter()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["StudioNexora"]
story_projects = db["story_projects"]
chat_sessions = db["chat_sessions"]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Mapping of agent names to their functions
agent_map = {
    "Writer": writer_agent,
    "Director": director_agent,
    "Casting": casting_agent,
    "AD": ad_agent,
    "VideoDesign": video_design_agent,
    "Editor": editor_agent,
    "Storyboard": storyboard_agent,
}

class StartChatRequest(BaseModel):
    prompt: str

class SendChatRequest(BaseModel):
    session_id: str
    prompt: Optional[str] = None

class EditMessageRequest(BaseModel):
    session_id: str
    message_id: str
    new_message: str

# Helper function to determine response type
def determine_response_type(text: str) -> str:
    """
    Determine the type of response based on the content.
    """
    if not text:
        return "thinking"
        
    text_lower = text.lower()
    if "creating character" in text_lower or "character profile" in text_lower:
        return "character"
    elif "scene" in text_lower or "setting" in text_lower:
        return "scene"
    elif "plot" in text_lower or "story" in text_lower:
        return "plot"
    else:
        return "general"

# Generate thinking process explanation
async def generate_thinking_process(user_message: str) -> str:
    """Generate a thinking process explanation for how GPT will approach the task"""
    try:
        prompt = f"""
You are an AI assistant that explains its thinking process.
A user has asked: "{user_message}"

Write a clear explanation of how you would approach this task, including the steps you would take.
FORMAT YOUR RESPONSE IN VALID MARKDOWN following these guidelines:
1. Use proper heading structure (# for main headings).
2. Use proper subheading structure (## for subheadings).
3. Use PROPERLY FORMATTED bullet points with complete sentences.
4. Ensure all markdown syntax is valid (no incomplete bullet points, etc.).
5. Properly format any lists with complete bullets.
6. Keep your explanation focused and structured.

Start your explanation with a heading "## My Approach" and organize your thoughts with properly formatted bullet points.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller, faster model for the thinking part
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        
        thinking = response.choices[0].message.content.strip()
        return thinking
    except Exception as e:
        print(f"Error generating thinking process: {e}")
        return f"## My Approach\n\nI will help you with: '{user_message}'"

# Generate chat suggestions
async def generate_chat_suggestions(session_id: str, recent_messages: list = None) -> list:
    """Generate action-oriented suggestions based on the user's progression through the story creation process"""
    try:
        # If no recent messages provided, fetch from database
        if not recent_messages:
            session = chat_sessions.find_one({"session_id": session_id})
            if not session or "messages" not in session:
                return ["Generate episodes", "Improve the story"]
            
            # Get only user and nexora messages (not thinking messages)
            recent_messages = [
                msg for msg in session["messages"] 
                if msg.get("sender") in ["user", "nexora"]
            ]
        
        # Get the most recent user message to determine context
        user_messages = [msg for msg in recent_messages if msg.get("sender") == "user"]
        last_user_message = user_messages[-1]["message"].lower() if user_messages else ""
        
        # Get project data to check state
        project = story_projects.find_one({"session_id": session_id})
        if not project or not project.get("story_data"):
            # Initial state - no story data yet
            return ["Generate episodes", "Improve the story"]
            
        story_data = project.get("story_data", {})
        
        # Check if "create episodes" was the last command
        if "create episodes" in last_user_message or "generate episodes" in last_user_message:
            return ["Finalise characters", "Improve episodes"]
            
        # Check if "finalise characters" was the last command
        if "finalise characters" in last_user_message or "finalize characters" in last_user_message:
            suggestions = ["Change characters"]
            
            # Add episode creation suggestions
            if "episodes" in story_data and story_data["episodes"]:
                for i, episode in enumerate(story_data["episodes"]):
                    episode_num = episode.get("episode_number", i+1)
                    suggestions.append(f"Create episode {episode_num}")
            return suggestions
            
        # Check if creating a specific episode
        episode_match = re.search(r"create episode (\d+)", last_user_message)
        if episode_match:
            episode_num = int(episode_match.group(1))
            # Generate scene suggestions for this episode
            suggestions = []
            
            # Check if we have scene information for this episode
            episode_str = str(episode_num)
            if "structured_scenes" in story_data and episode_str in story_data["structured_scenes"]:
                # Get number of scenes
                scenes = story_data["structured_scenes"][episode_str]
                for scene in scenes:
                    scene_num = scene.get("scene_number", 0)
                    suggestions.append(f"Create episode {episode_num} scene {scene_num}")
            else:
                # Fallback to episode_scripts
                if "episode_scripts" in story_data and episode_str in story_data["episode_scripts"]:
                    scenes = story_data["episode_scripts"][episode_str]
                    for i in range(len(scenes)):
                        suggestions.append(f"Create episode {episode_num} scene {i+1}")
                else:
                    # No scenes yet, suggest creating some default ones
                    for i in range(1, 6):  # Suggest 5 scenes
                        suggestions.append(f"Create episode {episode_num} scene {i}")
            
            return suggestions

        # Check if creating a specific scene
        scene_match = re.search(r"create episode (\d+) scene (\d+)", last_user_message)
        if scene_match:
            episode_num = int(scene_match.group(1))
            scene_num = int(scene_match.group(2))
            return [f"Generate storyboard for episode {episode_num} scene {scene_num}"]
            
        # Default behavior based on project state
        has_episodes = "episodes" in story_data and story_data["episodes"]
        has_characters = ("character_profiles" in story_data and story_data["character_profiles"]) or ("characters" in story_data and story_data["characters"])
            
        if not has_episodes:
            return ["Generate episodes", "Create characters", "Improve the story"]
        elif not has_characters:
            return ["Finalise characters", "Improve episodes", "Change story"]
        else:
            # Both episodes and characters exist
            suggestions = ["Generate scene images", "Change story"]
            
            # Add first episode scene creation suggestion
            if has_episodes and story_data["episodes"]:
                first_episode = story_data["episodes"][0]
                ep_num = first_episode.get("episode_number", 1)
                suggestions.insert(0, f"Create episode {ep_num} scene 1")
                
            return suggestions
            
    except Exception as e:
        print(f"Error generating chat suggestions: {e}")
        import traceback
        print(traceback.format_exc())
        return ["Generate episodes", "Create characters", "Change story"]

# Helper functions for formatting markdown responses
def format_episodes_as_markdown(episodes):
    try:
        markdown = "# Series Episodes\n\n"
        if not episodes:
            return "No episodes available."
            
        for episode in episodes:
            episode_number = episode.get("episode_number", 0)
            title = episode.get("episode_title", f"Episode {episode_number}")
            summary = episode.get("summary", "No summary available")
            
            markdown += f"## Episode {episode_number}: {title}\n\n"
            markdown += f"{summary}\n\n"
        
        return markdown
    except Exception as e:
        print(f"Error formatting episodes: {e}")
        return "Error formatting episodes."

def format_episode_scripts_as_markdown(episode_scripts, structured_scenes=None):
    try:
        markdown = "# Episode Scenes\n\n"
        if not episode_scripts and not structured_scenes:
            return "No episode scripts available."
        
        # Use structured format if available
        if structured_scenes:
            for episode_num, scenes in structured_scenes.items():
                markdown += f"## Episode {episode_num}\n\n"
                
                for scene in scenes:
                    scene_num = scene.get("scene_number", 0)
                    scene_title = scene.get("title", f"Scene {scene_num}")
                    scene_description = scene.get("description", "")
                    
                    # Remove the redundant "Scene X:" prefix if it exists
                    scene_prefix_pattern = re.match(r'^Scene\s*\d+\s*:\s*(.*)', scene_description, re.IGNORECASE)
                    if scene_prefix_pattern:
                        scene_description = scene_prefix_pattern.group(1).strip()
                    
                    markdown += f"### Scene {scene_num}: {scene_title}\n\n"
                    markdown += f"{scene_description}\n\n"
        # Fall back to old format
        else:
            for episode_num, scenes in episode_scripts.items():
                markdown += f"## Episode {episode_num}\n\n"
                
                for i, scene in enumerate(scenes):
                    markdown += f"### Scene {i+1}\n\n"
                    
                    # Remove the "Scene X:" prefix if it exists
                    scene_content = scene
                    # Using a more flexible regex that captures the scene number pattern
                    scene_prefix_pattern = re.match(r'^Scene\s*\d+\s*:\s*(.*)', scene, re.IGNORECASE)
                    if scene_prefix_pattern:
                        scene_content = scene_prefix_pattern.group(1).strip()
                    
                    markdown += f"{scene_content}\n\n"
        
        return markdown
    except Exception as e:
        print(f"Error formatting episode scripts: {e}")
        return "Error formatting episode scripts."

def format_scene_scripts_as_markdown(scene_scripts):
    try:
        markdown = "# Scene Details\n\n"
        if not scene_scripts:
            return "No scene scripts available."
            
        for scene_key, shots in scene_scripts.items():
            match = re.match(r'ep(\d+)_scene(\d+)', scene_key)
            if match:
                episode_num, scene_num = match.groups()
                markdown += f"## Episode {episode_num}, Scene {scene_num}\n\n"
            else:
                markdown += f"## {scene_key}\n\n"
            
            for i, shot in enumerate(shots):
                markdown += f"### Shot {i+1}\n\n"
                
                if "shot" in shot:
                    markdown += f"**Visual:** {shot['shot']}\n\n"
                
                if "dialogue" in shot:
                    markdown += f"**Dialogue:** {shot['dialogue']}\n\n"
        
        return markdown
    except Exception as e:
        print(f"Error formatting scene scripts: {e}")
        return "Error formatting scene scripts."

# Add this helper function after the existing helper functions (around line 240)
async def refresh_project_data(session_id: str):
    """
    Refresh project data from the database, extracting relevant information for the response.
    
    Args:
        session_id (str): The session ID to look up
        
    Returns:
        tuple: (refreshed_project, refreshed_character_data, refreshed_episodes_data)
    """
    try:
        # Query the project from the database
        refreshed_project = story_projects.find_one({"session_id": session_id})
        
        if not refreshed_project:
            print(f"Project not found for session: {session_id}")
            return None, [], []
            
        # Extract character information
        refreshed_character_data = []
        refreshed_episodes_data = []
        
        # Load the story data
        refreshed_story_data = refreshed_project.get("story_data", {})
        
        print(f"DEBUG: Refreshed story data keys: {list(refreshed_story_data.keys() if refreshed_story_data else [])}")
        
        # Process character data if available
        if refreshed_story_data and "character_profiles" in refreshed_story_data:
            char_profiles = refreshed_story_data["character_profiles"]
            print(f"DEBUG: Found {len(char_profiles)} characters in refreshed data")
            
            character_map = refreshed_story_data.get("character_map", {})
            
            # Process each character from story_data
            for profile in char_profiles:
                # Handle both string and dictionary formats (for backward compatibility)
                if isinstance(profile, str):
                    # Try to parse JSON string
                    try:
                        import json
                        parsed_profile = json.loads(profile)
                        if isinstance(parsed_profile, dict):
                            profile = parsed_profile
                    except:
                        # If parsing fails, use the string as description
                        continue
                
                # Extract character information
                if isinstance(profile, dict):
                    name = profile.get("name", "Unknown")
                    desc = profile.get("description", "")
                    
                    # Determine token for this character
                    token = None
                    for char_name, char_token in character_map.items():
                        if char_name == name:
                            token = char_token
                            break
                    
                    # Default token if not found
                    if not token:
                        token = f"character-{len(refreshed_character_data) + 1}"
                    
                    # Create character entry in the format expected by the frontend
                    char_entry = {
                        "name": name,
                        "token": token,
                        "description": desc
                    }
                    
                    # Add image URL if available
                    if "reference_image" in profile:
                        char_entry["reference_image"] = profile["reference_image"]
                    elif "image_url" in profile:
                        char_entry["reference_image"] = profile["image_url"]
                    
                    # Add gender if available
                    if "gender" in profile:
                        char_entry["gender"] = profile["gender"]
                    
                    refreshed_character_data.append(char_entry)
        
        # Process episode data if available
        if refreshed_story_data and "episodes" in refreshed_story_data and refreshed_story_data["episodes"]:
            print(f"DEBUG: Found {len(refreshed_story_data['episodes'])} episodes in refreshed data")
            # Process each episode from refreshed story_data
            for episode in refreshed_story_data["episodes"]:
                episode_number = episode.get("episode_number", 0)
                episode_title = episode.get("episode_title", f"Episode {episode_number}")
                
                # Create the episode entry with child scenes
                episode_entry = {
                    "title": episode_title,
                    "prompt": f"create episode {episode_number}",
                    "child": []
                }
                
                # Check if structured scenes exist for this episode
                episode_str = str(episode_number)
                has_scenes = False
                
                # First check structured_scenes
                if "structured_scenes" in refreshed_story_data and episode_str in refreshed_story_data["structured_scenes"]:
                    scenes = refreshed_story_data["structured_scenes"][episode_str]
                    print(f"DEBUG: Found {len(scenes)} structured scenes for episode {episode_number}")
                    # Process each scene for this episode
                    for scene in scenes:
                        scene_number = scene.get("scene_number", 0)
                        scene_title = scene.get("title", f"Scene {scene_number}")
                        
                        # Add scene to episode's children
                        scene_entry = {
                            "title": scene_title,
                            "prompt": f"create scene {scene_number} episode {episode_number}"
                        }
                        episode_entry["child"].append(scene_entry)
                    has_scenes = True
                
                # Then check episode_scripts as fallback and add any missing scenes
                if "episode_scripts" in refreshed_story_data and episode_str in refreshed_story_data["episode_scripts"]:
                    scenes = refreshed_story_data["episode_scripts"][episode_str]
                    print(f"DEBUG: Found {len(scenes)} scenes in episode_scripts for episode {episode_number}")
                    
                    # Get existing scene numbers to avoid duplicates
                    existing_scene_numbers = set()
                    for child in episode_entry["child"]:
                        prompt = child.get("prompt", "")
                        scene_match = re.search(r"create scene (\d+)", prompt)
                        if scene_match:
                            existing_scene_numbers.add(int(scene_match.group(1)))
                    
                    # Add scenes that don't exist yet
                    for i, scene_desc in enumerate(scenes):
                        scene_number = i + 1
                        if scene_number in existing_scene_numbers:
                            continue  # Skip scenes that already exist
                            
                        scene_title = f"Scene {scene_number}"
                        
                        # Try to extract a title from the scene description
                        scene_title_match = re.search(r'^Scene\s+\d+:\s*(.+?)[\.\n]', scene_desc)
                        if scene_title_match:
                            scene_title = scene_title_match.group(1).strip()
                        
                        scene_entry = {
                            "title": scene_title,
                            "prompt": f"create scene {scene_number} episode {episode_number}"
                        }
                        episode_entry["child"].append(scene_entry)
                    has_scenes = True
                
                # Check scene_scripts for any scene-specific scripts that might not be in structured_scenes or episode_scripts
                if "scene_scripts" in refreshed_story_data:
                    # Look for keys matching this episode
                    scene_script_pattern = f"ep{episode_number}_scene"
                    found_scenes = []
                    for key in refreshed_story_data["scene_scripts"].keys():
                        if key.startswith(scene_script_pattern):
                            try:
                                # Extract scene number from key (e.g., "ep3_scene4" -> 4)
                                scene_match = re.search(r"scene(\d+)", key)
                                if scene_match:
                                    scene_number = int(scene_match.group(1))
                                    
                                    # Check if this scene is already in our list
                                    if scene_number not in existing_scene_numbers:
                                        found_scenes.append(scene_number)
                            except:
                                continue
                    
                    # For each scene found, add to the episode's children if not already present
                    for scene_number in found_scenes:
                        if scene_number in existing_scene_numbers:
                            continue
                            
                        scene_entry = {
                            "title": f"Scene {scene_number}",
                            "prompt": f"create scene {scene_number} episode {episode_number}"
                        }
                        episode_entry["child"].append(scene_entry)
                        existing_scene_numbers.add(scene_number)
                        has_scenes = True
                
                # Also check storyboard data for any additional scenes that might be missing
                if "storyboard" in refreshed_story_data and refreshed_story_data["storyboard"]:
                    for item in refreshed_story_data["storyboard"]:
                        if item.get("episode_number") == episode_number:
                            scene_number = item.get("scene_number")
                            if scene_number and scene_number not in existing_scene_numbers:
                                scene_entry = {
                                    "title": f"Scene {scene_number}",
                                    "prompt": f"create scene {scene_number} episode {episode_number}"
                                }
                                episode_entry["child"].append(scene_entry)
                                existing_scene_numbers.add(scene_number)
                                has_scenes = True
                
                # Sort the children by scene number for consistency
                episode_entry["child"].sort(key=lambda x: int(re.search(r"scene (\d+)", x.get("prompt", "scene 1000")).group(1)))
                
                refreshed_episodes_data.append(episode_entry)
        else:
            print(f"DEBUG: No episodes found in refreshed data. Keys: {list(refreshed_story_data.keys() if refreshed_story_data else [])}")
        
        print(f"Refreshed data: {len(refreshed_character_data)} characters, {len(refreshed_episodes_data)} episodes")
        return refreshed_project, refreshed_character_data, refreshed_episodes_data
    except Exception as e:
        print(f"Error refreshing project data: {e}")
        import traceback
        print(traceback.format_exc())
        return None, [], []

@router.post("/chat/start")
async def start_chat(req: StartChatRequest):
    print(f"Creating new chat session")
    session_id = str(uuid.uuid4())
    user_id = "user123"  # In real system, this would come from auth
    
    # Generate synopsis from the initial prompt
    try:
        synopsis_prompt = f"""
Generate a brief, engaging synopsis (1-2 paragraphs) for a story based on this prompt:
"{req.prompt}"

The synopsis should capture the essence of what this story would be about.
Keep it under 200 words and make it compelling.
"""
        
        synopsis_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": synopsis_prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        synopsis = synopsis_response.choices[0].message.content.strip()
        print(f"Generated synopsis: {synopsis[:50]}...")
    except Exception as e:
        print(f"Error generating synopsis: {e}")
        synopsis = ""
    
    # Create initial document in story_projects with the synopsis
    story_projects.insert_one({
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "synopsis": synopsis,  # Store the synopsis separately at root level
        "story_data": {
            "session_id": session_id
        }
    })
    
    # Store initial user message
    chat_sessions.insert_one({
        "session_id": session_id,
        "messages": [{
            "sender": "user",
            "message": req.prompt,
            "timestamp": datetime.utcnow(),
            "response_type": "string",
            "message_id": str(uuid.uuid4())
        }]
    })
    
    print(f"Created new session with ID: {session_id}")
    
    # Return only the session ID
    return {"session_id": session_id}

@router.post("/chat/send")
async def send_chat_message(req: SendChatRequest):
    print(f"\n==== /chat/send API CALLED ====")
    print(f"Request: session_id={req.session_id}, prompt={req.prompt if req.prompt else 'None'}")
    
    try:
        session = chat_sessions.find_one({"session_id": req.session_id})
        if not session:
            print(f"ERROR: Session {req.session_id} not found in database")
            raise HTTPException(status_code=404, detail="Session not found")

        print(f"Found session in database with {len(session.get('messages', []))} existing messages")

        # Initialize chat history first
        print(f"Building chat history for left_section")
        chat_history = []
        if session and "messages" in session:
            print(f"Processing {len(session['messages'])} messages for chat history")
            for msg in session["messages"]:
                if msg.get("sender") == "user":
                    chat_history.append({
                        "sender": {
                            "time": str(msg.get("timestamp", "")),
                            "message": msg.get("message", "")
                        }
                    })
                elif msg.get("sender") == "nexora":
                    chat_history.append({
                        "receiver": {
                            "time": str(msg.get("timestamp", "")),
                            "message": msg.get("message", "")
                        }
                    })
            print(f"Built chat history with {len(chat_history)} entries")
        else:
            print(f"No messages found in session, chat history will be empty")

        # Get the messages
        messages = session.get('messages', [])
        
        # Check if this is the first call to /chat/send
        receiver_messages = [msg for msg in messages if msg.get('sender') == 'nexora']
        is_first_call = len(receiver_messages) == 0
        
        # Get initial prompt only if this is the first call
        initial_prompt = None
        if is_first_call and messages and messages[0].get('sender') == 'user':
            initial_prompt = messages[0].get('message')
            print(f"First call to /chat/send, using initial prompt: {initial_prompt}")
        
        # Use provided prompt or initial prompt (only on first call) or empty string
        prompt = req.prompt if req.prompt else initial_prompt if is_first_call else ""
        
        # Early error detection - check if prompt is None or empty when creating an episode
        if not prompt or prompt.strip() == "":
            print("WARNING: Empty prompt received")
            # Create an appropriate response for an empty prompt
            if is_first_call:
                # This is the first call after chat/start, probably just getting initial data
                print("Empty prompt on first call, returning initial response")
            else:
                print("ERROR: Empty prompt on non-first call")
        
        if prompt:
            print(f"Processing prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Processing prompt: {prompt}")
            
            # Check if this is a create episode request for debugging
            episode_match = re.search(r'create\s+episode\s+(\d+)', prompt.lower())
            if episode_match:
                episode_number = int(episode_match.group(1))
                print(f"DEBUG: Detected create episode {episode_number} request")
                
            # Store the user message only if it's a new prompt (not the initial one being reused)
            if prompt != initial_prompt:
                user_message = {
                    "sender": "user",
                    "message": prompt,
                    "timestamp": datetime.utcnow(),
                    "response_type": "string",
                    "message_id": str(uuid.uuid4())
                }
                print(f"Storing new user message in database with message_id: {user_message['message_id']}")
                chat_sessions.update_one(
                    {"session_id": req.session_id},
                    {"$push": {"messages": user_message}}
                )
                print(f"User message stored successfully")
                
                # Add the sender message to chat history
                chat_history.append({
                    "sender": {
                        "time": str(user_message["timestamp"]),
                        "message": user_message["message"]
                    }
                })
            else:
                print(f"First call to chat/send after chat/start - will not generate full content")
        
        # Create state object for processing
        print(f"Creating StoryState object")
        state = StoryState()
        
        # Try to load any existing story data
        print(f"Attempting to load story data from database")
        project = story_projects.find_one({"session_id": req.session_id})
        
        # Get the stored synopsis (which should never change after initial creation)
        stored_synopsis = ""
        if project:
            stored_synopsis = project.get("synopsis", "")
            print(f"Retrieved stored synopsis: {stored_synopsis[:50]}..." if stored_synopsis else "No synopsis found")
            
            if project.get("story_data"):
                print(f"Found story data in project, loading into state")
                
                # Add data cleansing before loading into StoryState
                story_data = project.get("story_data", {})
                
                # Fix character_profiles if they are strings instead of dictionaries
                if "character_profiles" in story_data and story_data["character_profiles"]:
                    try:
                        fixed_profiles = []
                        for profile in story_data["character_profiles"]:
                            if isinstance(profile, str):
                                # Convert string to dictionary with basic fields
                                name = profile.split(',')[0].strip() if ',' in profile else profile
                                
                                # Extract description if possible (after first comma)
                                description = profile[profile.find(',')+1:].strip() if ',' in profile else ""
                                
                                # Create a dictionary with minimum required fields
                                fixed_profile = {
                                    "name": name,
                                    "description": description,
                                    "gender": "male" if name.lower() in ["aman", "rajiv", "vikram"] else "female",
                                    "reference_image": ""
                                }
                                # Apply field validation and defaults
                                fixed_profile = _ensure_character_profile_fields(fixed_profile)
                                fixed_profiles.append(fixed_profile)
                            else:
                                # Already a dictionary, just ensure it has all needed fields
                                fixed_profile = _ensure_character_profile_fields(profile)
                                fixed_profiles.append(fixed_profile)
                        
                        # Replace with fixed profiles
                        story_data["character_profiles"] = fixed_profiles
                        print(f"Fixed {len(fixed_profiles)} character profiles from string to dictionary format")
                    except Exception as e:
                        print(f"Error fixing character profiles: {e}")
                        # Remove problematic character_profiles entirely if they can't be fixed
                        story_data.pop("character_profiles", None)
                        print("Removed problematic character_profiles field")
                
                # Try loading StoryState with fixed data
                try:
                    state = StoryState(**story_data)
                    print(f"Story data loaded successfully")
                except Exception as e:
                    print(f"Error loading story data into StoryState: {e}")
                    # If loading fails, create a clean state
                    state = StoryState()
                    state.session_id = req.session_id
                    
                    # Copy over any non-problematic fields
                    for key, value in story_data.items():
                        if key != "character_profiles" and hasattr(state, key):
                            try:
                                setattr(state, key, value)
                                print(f"Copied {key} to clean state")
                            except Exception as e:
                                print(f"Could not copy {key}: {e}")
                    
                    print("Created clean state with safe fields")
            else:
                print("No story_data found in project")
                state = StoryState()
        else:
            print(f"No story data found for this session or empty data")
            state = StoryState()
        
        # Prepare response structure
        initial_response = {
            "prompt": prompt,
            "left_section": chat_history[:-1] if prompt and chat_history and len(chat_history) > 1 else chat_history,
            "tabs": [],
            "synopsis": stored_synopsis,  # Always use the stored synopsis instead of logline
            "script": "",
            "character": [],
            "storyboard": state.storyboard if hasattr(state, "storyboard") and state.storyboard else [],
            "episodes": []
        }
        
        # If there's no prompt, just return the initial response with existing data
        if not prompt:
            initial_response["episodes"] = []
            return initial_response
        
        # For the first call after chat/start, we want to return just the synopsis without script content
        if is_first_call:
            try:
                # Generate thinking process asynchronously
                thinking_process = await generate_thinking_process(prompt)
                print(f"Generated thinking process: {len(thinking_process)} chars")
                
                # Create a better initial message instead of showing thinking process
                initial_message = f"## {stored_synopsis.split('\"')[1] if '\"' in stored_synopsis else 'New Story'}\n\n"
                initial_message += "I'll help you create this romantic drama series. You can start by:\n\n"
                initial_message += "- Creating episodes for the series\n"
                initial_message += "- Developing the main characters\n"
                initial_message += "- Setting up specific scenes\n\n"
                initial_message += "What would you like to do first?"
                
                # Create response with better initial message
                streamed_left_section = chat_history.copy()
                streamed_left_section.append({
                    "receiver": {
                        "time": str(datetime.utcnow()),
                        "message": initial_message
                    }
                })
                
                # Get suggestions for the UI
                suggestions = await generate_chat_suggestions(req.session_id)
                
                # Create Nexora response message with thinking process and overview
                nexora_message = {
                    "sender": "nexora",
                    "message": thinking_process,  # Only storing thinking process, not full content
                    "timestamp": datetime.utcnow(),
                    "response_type": "thinking",
                    "message_id": str(uuid.uuid4()),
                    "thinking_process": thinking_process,
                    "script_overview": initial_message  # Add the new initial message
                }
                
                # Store in database
                chat_sessions.update_one(
                    {"session_id": req.session_id},
                    {"$push": {"messages": nexora_message}}
                )
                
                # First call response - only includes synopsis, no script or episodes
                first_call_response = {
                    "prompt": prompt,
                    "left_section": streamed_left_section,
                    "tabs": suggestions,
                    "synopsis": stored_synopsis,
                    "script": "",  # Empty script - no episode generation
                    "character": [],
                    "storyboard": state.storyboard if hasattr(state, "storyboard") and state.storyboard else [],
                    "episodes": []  # Empty episodes array for first call
                }
                
                print(f"Returning first call response with synopsis only")
                return first_call_response
            except Exception as e:
                print(f"Error in send_chat_message first call: {e}")
                import traceback
                print(traceback.format_exc())
                # Use HTTPException instead of JSONResponse for consistency
                raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
        # Normal processing for non-first calls
        try:
            # Generate thinking process asynchronously
            thinking_process = await generate_thinking_process(prompt)
            print(f"Generated thinking process: {len(thinking_process)} chars")
            
            # Create response with thinking process
            streamed_left_section = chat_history.copy()
            streamed_left_section.append({
                "receiver": {
                    "time": str(datetime.utcnow()),
                    "message": thinking_process
                }
            })
            
            # Get the complete response from run_producer_stream
            response_text = await run_producer_stream(state, req.session_id, prompt)
            print(f"Got complete response: {len(response_text)} chars")
            
            # Ensure we never have an empty response to display
            if not response_text or response_text.strip() == "":
                print("Warning: Empty response received from run_producer_stream")
                
                # Create a meaningful fallback response based on the prompt
                if "create episodes" in prompt.lower():
                    response_text = "Episodes have been generated successfully. View them in the episode list."
                elif "create episode" in prompt.lower() and "scene" not in prompt.lower():
                    # Extract episode number if possible
                    episode_match = re.search(r'episode\s+(\d+)', prompt.lower())
                    episode_number = episode_match.group(1) if episode_match else ""
                    response_text = f"Episode {episode_number} has been created successfully."
                elif "create" in prompt.lower() and "scene" in prompt.lower():
                    # Extract episode and scene numbers if possible
                    episode_match = re.search(r'episode\s+(\d+)', prompt.lower())
                    scene_match = re.search(r'scene\s+(\d+)', prompt.lower())
                    
                    episode_number = episode_match.group(1) if episode_match else ""
                    scene_number = scene_match.group(1) if scene_match else ""
                    
                    response_text = f"Scene {scene_number} for Episode {episode_number} has been created successfully."
                else:
                    response_text = "Your request has been processed successfully."
            
            # Get suggestions for the UI
            suggestions = await generate_chat_suggestions(req.session_id)
            
            # Create Nexora response message
            # Store the full response in the database but generate an overview for UI display
            script_overview = await generate_script_overview(response_text, prompt)
            
            nexora_message = {
                "sender": "nexora",
                "message": response_text,  # Store full response in DB
                "timestamp": datetime.utcnow(),
                "response_type": determine_response_type(response_text),
                "message_id": str(uuid.uuid4()),
                "thinking_process": thinking_process,  # Keep thinking process in DB
                "script_overview": script_overview    # Add the new script overview
            }
            
            # Store in database
            chat_sessions.update_one(
                {"session_id": req.session_id},
                {"$push": {"messages": nexora_message}}
            )
            
            # Initialize episodes data structure
            episodes_data = []
            
            # For the left_section in the UI, we'll only show the thinking process, not the full response
            # This is the key change being made - we're not adding the response_text to the left_section
            
            # Get character data if available
            character_data = []
            if project and project.get("story_data"):
                story_data = project.get("story_data", {})
                
                # Attempt to extract character data from multiple sources in order of preference
                if "character_profiles" in story_data and isinstance(story_data["character_profiles"], list):
                    print(f"Found character_profiles in story_data, attempting to extract")
                    
                    # Process character profiles to ensure they're properly formatted
                    for profile in story_data["character_profiles"]:
                        # If profile is a string, convert to a dictionary
                        if isinstance(profile, str):
                            # Extract name and description from string
                            name = profile.split(',')[0].strip() if ',' in profile else profile
                            description = profile[profile.find(',')+1:].strip() if ',' in profile else profile
                            
                            # Create properly formatted character object for frontend
                            character_obj = {
                                "name": name,
                                "description": description,
                                "reference_image": "",  # Default empty image URL
                                "gender": "male" if name.lower() in ["aman", "rajiv", "vikram"] else "female",
                                "token": f"character-{len(character_data)+1}"
                            }
                            character_data.append(character_obj)
                            print(f"Converted string profile to object: {name}")
                        elif isinstance(profile, dict):
                            # If it's already a dictionary, ensure it has all required fields
                            char_obj = profile.copy()
                            if "name" not in char_obj or not char_obj["name"]:
                                # Try to extract name from description if no name field
                                if "description" in char_obj and char_obj["description"]:
                                    desc = char_obj["description"]
                                    name_match = re.match(r'^([^,]+)', desc)
                                    if name_match:
                                        char_obj["name"] = name_match.group(1).strip()
                            
                            # Ensure image URL is set
                            if "reference_image" not in char_obj or not char_obj["reference_image"]:
                                char_obj["reference_image"] = ""
                            
                            # Ensure token exists
                            if "token" not in char_obj or not char_obj["token"]:
                                char_obj["token"] = f"character-{len(character_data)+1}"
                            
                            character_data.append(char_obj)
                            print(f"Processed dictionary profile: {char_obj.get('name', 'Unknown')}")
                
                # If still no character data, try extracting from 'characters' field
                if not character_data and "characters" in story_data and isinstance(story_data["characters"], list):
                    print(f"No character_profiles found, using characters field")
                    
                    for idx, char_desc in enumerate(story_data["characters"]):
                        if isinstance(char_desc, str):
                            # Extract name from the beginning of the description
                            name_match = re.match(r'^([^,]+)', char_desc)
                            name = name_match.group(1).strip() if name_match else f"Character {idx+1}"
                            
                            # Create character object
                            char_obj = {
                                "name": name,
                                "description": char_desc,
                                "reference_image": "",  # Default empty image URL
                                "gender": "male" if name.lower() in ["aman", "rajiv", "vikram"] else "female",
                                "token": f"character-{idx+1}"
                            }
                            
                            # Look for image URL in character_map if available
                            if "character_map" in story_data and isinstance(story_data["character_map"], dict):
                                if name in story_data["character_map"]:
                                    char_obj["token"] = story_data["character_map"][name]
                            
                            character_data.append(char_obj)
                            print(f"Created character from characters array: {name}")
                
                # Only keep character data from characters field if at least one character has a reference image
                has_images = False
                for char in character_data:
                    if char.get("reference_image"):
                        has_images = True
                        break
                
                if not has_images:
                    print("No character images found, clearing character data")
                    character_data = []

                # If we found character data, look for image URLs in the script content
                if character_data and response_text:
                    print("Looking for character images in script content")
                    
                    # Check for image markdown in the format: ![CharacterName](image_url)
                    image_matches = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', response_text)
                    
                    if image_matches:
                        print(f"Found {len(image_matches)} image references in script")
                        
                        # Match images to characters by name
                        for char_name, image_url in image_matches:
                            # Clean up character name (remove "Character " prefix if present)
                            clean_name = re.sub(r'^Character\s+', '', char_name).strip()
                            
                            # Find corresponding character and update its image URL
                            for char in character_data:
                                if clean_name.lower() in char["name"].lower() or char["name"].lower() in clean_name.lower():
                                    char["reference_image"] = image_url
                                    print(f"Updated image URL for {char['name']}")
                                    break
                    
                    # Also look for structured character sections
                    char_sections = re.findall(r'# Character ([^\n]+)\s+\*\*Gender:\*\* ([^\n]+)\s+!\[([^\]]+)\]\(([^)]+)\)', response_text, re.DOTALL)
                    
                    if char_sections:
                        print(f"Found {len(char_sections)} structured character sections")
                        
                        for name, gender, img_alt, img_url in char_sections:
                            # Find matching character
                            for char in character_data:
                                if name.lower() in char["name"].lower() or char["name"].lower() in name.lower():
                                    char["reference_image"] = img_url
                                    char["gender"] = gender.strip().lower()
                                    print(f"Updated character info from structured section: {name}")
                                    break
                
                # If still no complete character data, create from the script directly
                if (not character_data or not any(c.get("reference_image") for c in character_data)) and "Character" in response_text and "![" in response_text:
                    print("Creating character data directly from script content")
                    
                    # Extract character blocks from the script
                    char_blocks = re.findall(r'# Character ([^\n]+)\s+\*\*Gender:\*\* ([^\n]+)\s+!\[([^\]]+)\]\(([^)]+)\)', response_text, re.DOTALL)
                    
                    if char_blocks:
                        print(f"Found {len(char_blocks)} character blocks in script")
                        
                        # Create character objects from blocks
                        character_data = []
                        for idx, (name, gender, img_alt, img_url) in enumerate(char_blocks):
                            char_obj = {
                                "name": name.strip(),
                                "description": f"Character from script: {name}",
                                "reference_image": img_url.strip(),
                                "gender": gender.strip().lower(),
                                "token": f"character-{idx+1}"
                            }
                            character_data.append(char_obj)
                            print(f"Created character from script block: {name}")
                    
                    # Also try the simpler format in the example
                    if not char_blocks:
                        char_blocks = re.findall(r'# Character ([^\n]+)\s+\*\*Gender:\*\* ([^\n]+)\s+!\[.*?\]\(([^)]+)\)', response_text, re.DOTALL)
                        if char_blocks:
                            print(f"Found {len(char_blocks)} character blocks in alternate format")
                            character_data = []
                            for idx, (name, gender, img_url) in enumerate(char_blocks):
                                char_obj = {
                                    "name": name.strip(),
                                    "description": f"Character from script: {name}",
                                    "reference_image": img_url.strip(),
                                    "gender": gender.strip().lower(),
                                    "token": f"character-{idx+1}"
                                }
                                character_data.append(char_obj)
                                print(f"Created character from alternate format: {name}")
                
                # Special handling for the example format with "character" array of strings
                if not character_data and isinstance(story_data.get("characters"), list) and len(story_data.get("characters", [])) > 0:
                    print("Checking if script has character images that match the characters array")
                    
                    # First extract the character names from the strings
                    char_names = []
                    for char_str in story_data["characters"]:
                        if isinstance(char_str, str):
                            name_match = re.match(r'^([^,]+)', char_str)
                            if name_match:
                                char_names.append(name_match.group(1).strip())
                    
                    # Then find image URLs in the script
                    image_matches = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', response_text)
                    
                    if char_names and image_matches:
                        print(f"Found {len(char_names)} character names and {len(image_matches)} image references")
                        
                        # If we have the same number, assume they match in order
                        if len(char_names) == len(image_matches):
                            character_data = []
                            for idx, (name, (img_alt, img_url)) in enumerate(zip(char_names, image_matches)):
                                char_str = story_data["characters"][idx]
                                char_obj = {
                                    "name": name,
                                    "description": char_str,
                                    "reference_image": img_url,
                                    "gender": "male" if idx == 0 else "female",  # Simple alternating gender
                                    "token": f"character-{idx+1}"
                                }
                                character_data.append(char_obj)
                                print(f"Matched character {name} with image {img_url[:30]}...")
                        else:
                            # Try to match by name similarity
                            character_data = []
                            for idx, name in enumerate(char_names):
                                char_str = story_data["characters"][idx]
                                char_obj = {
                                    "name": name,
                                    "description": char_str,
                                    "reference_image": "",
                                    "gender": "male" if idx == 0 else "female",
                                    "token": f"character-{idx+1}"
                                }
                                
                                # Try to find a matching image
                                for img_alt, img_url in image_matches:
                                    # Clean the image alt text
                                    clean_alt = re.sub(r'^Character\s+', '', img_alt).strip()
                                    if clean_alt.lower() == name.lower() or name.lower() in clean_alt.lower() or clean_alt.lower() in name.lower():
                                        char_obj["reference_image"] = img_url
                                        print(f"Matched {name} with image: {img_url[:30]}...")
                                        break
                                
                                character_data.append(char_obj)
                
                # If we still don't have character data with images but script contains them, extract directly
                if (not character_data or not any(c.get("reference_image") for c in character_data)) and "![" in response_text:
                    print("Extracting character images directly from script markdown")
                    image_matches = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', response_text)
                    
                    if image_matches:
                        character_data = []
                        for idx, (img_alt, img_url) in enumerate(image_matches):
                            # Try to extract a clean name
                            clean_name = re.sub(r'^Character\s+', '', img_alt).strip()
                            if not clean_name:
                                clean_name = f"Character {idx+1}"
                            
                            char_obj = {
                                "name": clean_name,
                                "description": f"Character from script: {clean_name}",
                                "reference_image": img_url,
                                "gender": "male" if idx == 0 else "female",  # Simple gender alternation
                                "token": f"character-{idx+1}"
                            }
                            character_data.append(char_obj)
                            print(f"Created character directly from image: {clean_name}")
                
                print(f"Final character data for response: {len(character_data)} characters")
                for char in character_data:
                    print(f"  - {char.get('name', 'Unknown')}: {char.get('reference_image', 'No image')[:30]}...")
                    
                # Final validation to ensure all characters have required fields
                for char in character_data:
                    if not char.get("name"):
                        char["name"] = "Unknown Character"
                    if not char.get("description"):
                        char["description"] = f"Character: {char['name']}"
                    if not char.get("reference_image"):
                        char["reference_image"] = ""
                    if not char.get("gender"):
                        char["gender"] = "unknown"
                    if not char.get("token"):
                        char["token"] = f"character-{uuid.uuid4().hex[:8]}"
                
                # Filter character data to only include specific fields for frontend
                filtered_character_data = []
                for char in character_data:
                    filtered_char = {
                        "name": char.get("name", ""),
                        "token": char.get("token", ""),
                        "description": char.get("description", ""),
                        "reference_image": char.get("reference_image", ""),
                        "gender": char.get("gender", "unknown")
                    }
                    # Remove any other fields that might be present
                    filtered_character_data.append(filtered_char)
                
                character_data = filtered_character_data

                # For character creation/finalization requests, ensure character data is included in the response
                is_character_request = any(term in prompt.lower() for term in ["create characters", "finalize characters", "finalise characters"])
                
                # Check if characters array contains string descriptions
                if is_character_request and not character_data and "character" in story_data:
                    if isinstance(story_data["character"], list) and len(story_data["character"]) > 0:
                        print("Found character array in response - converting to objects")
                        
                        character_data = []
                        for idx, char_str in enumerate(story_data["character"]):
                            if isinstance(char_str, str):
                                # Extract name from beginning of string
                                name_match = re.match(r'^([^,]+)', char_str) 
                                name = name_match.group(1).strip() if name_match else f"Character {idx+1}"
                                
                                # Create character object
                                char_obj = {
                                    "name": name,
                                    "description": char_str,
                                    "reference_image": "",
                                    "gender": "male" if idx == 0 else "female",
                                    "token": f"character-{idx+1}"
                                }
                                character_data.append(char_obj)
                                print(f"Converted character string to object: {name}")
                
                # If characters have image information in script content
                if is_character_request and character_data and "![" in response_text:
                    print("Looking for character images in script")
                    
                    # Extract image URLs
                    image_matches = re.findall(r'!\[([^\]]+)\]\(([^)]+)\)', response_text)
                    
                    if len(image_matches) == len(character_data):
                        print(f"Found matching number of images ({len(image_matches)}) and characters ({len(character_data)})")
                        
                        # Update character images
                        for idx, (img_name, img_url) in enumerate(image_matches):
                            if idx < len(character_data):
                                character_data[idx]["reference_image"] = img_url
                                print(f"Updated image for {character_data[idx]['name']}: {img_url[:30]}...")
                    else:
                        print(f"Image count ({len(image_matches)}) doesn't match character count ({len(character_data)})")
                        
                        # Try to match by name
                        for img_name, img_url in image_matches:
                            clean_name = re.sub(r'^Character\s+', '', img_name).strip()
                            
                            for char in character_data:
                                if clean_name.lower() in char["name"].lower() or char["name"].lower() in clean_name.lower():
                                    char["reference_image"] = img_url
                                    print(f"Matched image for {char['name']}: {img_url[:30]}...")
                                    break
                
                # Special handling for the exact format in the example
                special_format = re.search(r'# Character ([^\n]+)\s+\*\*Gender:\*\* ([^\n]+)\s+!\[.*?\]\(([^)]+)\)', response_text)
                if special_format and not any(c.get("reference_image") for c in character_data):
                    print("Detected special character format - extracting directly")
                    
                    special_blocks = re.findall(r'# Character ([^\n]+)\s+\*\*Gender:\*\* ([^\n]+)\s+!\[.*?\]\(([^)]+)\)', response_text)
                    
                    if special_blocks:
                        # Override character data with directly extracted info
                        character_data = []
                        for idx, (name, gender, img_url) in enumerate(special_blocks):
                            char_obj = {
                                "name": name.strip(),
                                "description": f"Character: {name}",
                                "reference_image": img_url.strip(),
                                "gender": gender.strip().lower(),
                                "token": f"character-{idx+1}"
                            }
                            character_data.append(char_obj)
                            print(f"Created character from special format: {name}")

                # Check if this is an episode request
                is_episode_request = "episode" in prompt.lower()
                
                # Handle the case when no proper character objects have been created yet
                # Only include characters in the response if they have proper image URLs
                has_proper_characters = False
                if character_data and isinstance(character_data, list):
                    # Check if any character has an image URL
                    for char in character_data:
                        if isinstance(char, dict) and char.get("reference_image"):
                            has_proper_characters = True
                            break
                
                # For requests that don't specifically create/finalize characters, only include
                # characters in the response if they have been properly processed with images
                if not is_character_request and not has_proper_characters:
                    print("No fully processed characters with images yet - returning empty character array")
                    character_data = []
                
                # Extract episode information from response text if an episode is requested
                if is_episode_request and ("Episode" in response_text and "Scene" in response_text):
                    print("Detected episode request with scene information in response")
                    
                    # Extract episode number from prompt
                    episode_match = re.search(r'episode\s*(\d+)', prompt.lower())
                    episode_number = int(episode_match.group(1)) if episode_match else 1
                    print(f"Detected Episode {episode_number} in request")
                    
                    # For episode creation, ensure we're only returning character data if it has images
                    if not has_proper_characters:
                        print("Episode creation request - returning empty character array since no character images")
                        character_data = []
                    
                    # Create episode entry
                    episode_entry = {
                        "title": f"Episode {episode_number}",
                        "prompt": f"create episode {episode_number}",
                        "child": []
                    }
                    
                    # Find scenes in the response text
                    scene_matches = re.findall(r'Scene\s*(\d+):\s*([^\n]+)', response_text)
                    if scene_matches:
                        print(f"Found {len(scene_matches)} scenes in response text")
                        
                        for scene_num_str, scene_title in scene_matches:
                            try:
                                scene_number = int(scene_num_str)
                            except ValueError:
                                scene_number = len(episode_entry["child"]) + 1
                                
                            scene_entry = {
                                "title": scene_title.strip(),
                                "prompt": f"create scene {scene_number} episode {episode_number}"
                            }
                            episode_entry["child"].append(scene_entry)
                    
                    # Add the episode to the episodes_data list
                    episodes_data = [episode_entry]
                    print(f"Created episode entry with {len(episode_entry['child'])} scenes")
                    
                    # Also update the database with this information
                    if not "episodes" in story_data or not story_data["episodes"]:
                        story_data["episodes"] = []
                    
                    # Check if this episode already exists
                    episode_exists = False
                    for i, ep in enumerate(story_data["episodes"]):
                        if ep.get("episode_number") == episode_number:
                            episode_exists = True
                            break
                    
                    # Add the episode if it doesn't exist
                    if not episode_exists:
                        story_data["episodes"].append({
                            "episode_number": episode_number,
                            "episode_title": f"Episode {episode_number}",
                            "summary": "Auto-extracted from response"
                        })
                    
                    # Create structured scenes
                    if "structured_scenes" not in story_data:
                        story_data["structured_scenes"] = {}
                    
                    # Add scenes for this episode
                    structured_scenes = []
                    for i, scene in enumerate(episode_entry["child"]):
                        scene_number = i + 1
                        scene_title = scene.get("title", f"Scene {scene_number}")
                        
                        # Extract scene description from the script
                        scene_desc_pattern = rf'Scene\s*{scene_number}:[^\n]*\n\n(.*?)(?=Scene\s*\d+:|$)'
                        scene_desc_match = re.search(scene_desc_pattern, response_text, re.DOTALL)
                        scene_desc = scene_desc_match.group(1).strip() if scene_desc_match else "Auto-extracted from script"
                        
                        structured_scenes.append({
                            "scene_number": scene_number,
                            "title": scene_title,
                            "description": scene_desc
                        })
                    
                    # Update database with the new episode and scenes
                    update_data = {}
                    update_data["story_data.episodes"] = story_data["episodes"]
                    update_data[f"story_data.structured_scenes.{episode_number}"] = structured_scenes
                    update_data["updated_at"] = datetime.utcnow()
                    
                    # Also save to episode_scripts for backwards compatibility
                    if "episode_scripts" not in story_data:
                        story_data["episode_scripts"] = {}
                    
                    # Format the scene descriptions for episode_scripts
                    episode_scenes = []
                    for scene in structured_scenes:
                        scene_num = scene.get("scene_number", 0)
                        scene_title = scene.get("title", "")
                        scene_desc = scene.get("description", "")
                        episode_scenes.append(f"Scene {scene_num}: {scene_title}\n\n{scene_desc}")
                    
                    # Save to episode_scripts
                    story_data["episode_scripts"][str(episode_number)] = episode_scenes
                    update_data[f"story_data.episode_scripts.{episode_number}"] = episode_scenes
                    
                    story_projects.update_one(
                        {"session_id": req.session_id},
                        {"$set": update_data}
                    )
                    print(f"Updated database with episode {episode_number} and {len(structured_scenes)} scenes")
                    
                    # Create detailed episode script for the response
                    episode_script = f"# Episode {episode_number}: {episode_entry['title']}\n\n"
                    
                    # Add each scene with its description
                    for scene in structured_scenes:
                        scene_num = scene.get("scene_number", 0)
                        scene_title = scene.get("title", f"Scene {scene_number}")
                        scene_desc = scene.get("description", "")
                        
                        episode_script += f"## Scene {scene_num}: {scene_title}\n\n{scene_desc}\n\n"
                    
                    # Explicitly set the response_text for episode creation requests
                    # to ensure the script field gets the formatted episode content
                    if "create episode" in prompt.lower():
                        response_text = episode_script
                        print(f"Set script content for episode {episode_number} with {len(structured_scenes)} scenes")
                    
                    # Also update the scenes in the episodes_data 
                    for ep_entry in episodes_data:
                        # Check if this is the episode we're processing
                        title_match = re.search(r'Episode\s*(\d+)', ep_entry.get("title", ""))
                        if title_match and int(title_match.group(1)) == episode_number:
                            # Clear any existing child entries
                            ep_entry["child"] = []
                            
                            # Add the extracted scenes as children
                            for scene in structured_scenes:
                                scene_num = scene.get("scene_number", 0)
                                scene_title = scene.get("title", f"Scene {scene_num}")
                                
                                scene_entry = {
                                    "title": scene_title,
                                    "prompt": f"create scene {scene_num} episode {episode_number}"
                                }
                                ep_entry["child"].append(scene_entry)
                            
                            print(f"Added {len(structured_scenes)} scene entries to episode {episode_number} in response")
                            break
                            
                    # Make sure the script field has detailed episode information
                    # Format the script to show episode scenes
                    if "create episode" in prompt.lower():
                        # Format episode and scenes as markdown
                        episode_script_markdown = f"# Episode {episode_number}\n\n"
                        
                        # Add each scene with its description
                        for scene in structured_scenes:
                            scene_num = scene.get("scene_number", 0)
                            scene_title = scene.get("title", f"Scene {scene_num}")
                            scene_desc = scene.get("description", "")
                            
                            episode_script_markdown += f"## Scene {scene_num}: {scene_title}\n\n"
                            episode_script_markdown += f"{scene_desc}\n\n"
                        
                        # Override the response_text for this specific case
                        response_text = episode_script_markdown
                        print(f"Created detailed episode script markdown for Episode {episode_number} with {len(structured_scenes)} scenes")
                
                # Refresh from database to ensure we have the most up-to-date episode information
                print("Episode request detected - refreshing data from database")
                refreshed_project, refreshed_character_data, refreshed_episodes_data = await refresh_project_data(req.session_id)
                
                if refreshed_episodes_data:
                    print(f"Using refreshed episodes data: {len(refreshed_episodes_data)} episodes with scenes")
                    episodes_data = refreshed_episodes_data
                
                if refreshed_character_data and not character_data:
                    print(f"Using refreshed character data: {len(refreshed_character_data)} characters")
                    character_data = refreshed_character_data
                    
            # Handle storyboard and scene creation requests specifically
            # This ensures that scenes are properly added to episodes even if they weren't detected earlier
            is_scene_request = "create scene" in prompt.lower() or "scene" in prompt.lower() and "episode" in prompt.lower()
            if is_scene_request and "Scene" in response_text:
                print("Detected scene creation in response")
                
                # Extract episode and scene numbers
                episode_match = re.search(r'episode\s*(\d+)', prompt.lower())
                scene_match = re.search(r'scene\s*(\d+)', prompt.lower())
                
                if episode_match and scene_match:
                    episode_number = int(episode_match.group(1))
                    scene_number = int(scene_match.group(1))
                    print(f"Detected Scene {scene_number} for Episode {episode_number}")
                    
                    # Try to extract scene title from response
                    scene_title = f"Scene {scene_number}"
                    title_pattern = rf'Scene\s*{scene_number}.*?from\s*Episode\s*{episode_number}(.*?)(?=\n)'
                    scene_title_match = re.search(title_pattern, response_text)
                    if scene_title_match and scene_title_match.group(1).strip():
                        scene_title = scene_title_match.group(1).strip()
                    
                    # Ensure we have structured_scenes
                    if "structured_scenes" not in story_data:
                        story_data["structured_scenes"] = {}
                    
                    # Create scene data
                    episode_str = str(episode_number)
                    if episode_str not in story_data["structured_scenes"]:
                        story_data["structured_scenes"][episode_str] = []
                    
                    # Check if this scene already exists
                    scene_exists = False
                    for i, scene in enumerate(story_data["structured_scenes"].get(episode_str, [])):
                        if scene.get("scene_number") == scene_number:
                            # Update existing scene with new details
                            story_data["structured_scenes"][episode_str][i]["title"] = scene_title
                            story_data["structured_scenes"][episode_str][i]["description"] = response_text
                            scene_exists = True
                            break
                    
                    # If scene doesn't exist, add it
                    if not scene_exists:
                        story_data["structured_scenes"][episode_str].append({
                            "scene_number": scene_number,
                            "title": scene_title,
                            "description": response_text
                        })
                        print(f"Added Scene {scene_number} to Episode {episode_number}")
                    
                    # Make sure the episode exists
                    if "episodes" not in story_data:
                        story_data["episodes"] = []
                    
                    # Check if this episode exists
                    episode_exists = False
                    for i, ep in enumerate(story_data["episodes"]):
                        if ep.get("episode_number") == episode_number:
                            episode_exists = True
                            break
                    
                    # Add episode if it doesn't exist
                    if not episode_exists:
                        story_data["episodes"].append({
                            "episode_number": episode_number,
                            "episode_title": f"Episode {episode_number}",
                            "summary": f"Episode {episode_number} with Scene {scene_number}"
                        })
                        print(f"Added Episode {episode_number} to database")
                    
                    # Update the database
                    update_data = {}
                    if not episode_exists:
                        update_data["story_data.episodes"] = story_data["episodes"]
                    update_data[f"story_data.structured_scenes.{episode_str}"] = story_data["structured_scenes"][episode_str]
                    update_data["updated_at"] = datetime.utcnow()
                    
                    story_projects.update_one(
                        {"session_id": req.session_id},
                        {"$set": update_data}
                    )
                    print(f"Updated database with Scene {scene_number} for Episode {episode_number}")
                    
                    # Refresh data again to make sure episodes include this scene
                    refreshed_project, refreshed_character_data, refreshed_episodes_data = await refresh_project_data(req.session_id)
                    
                    if refreshed_episodes_data:
                        print(f"Final refresh: {len(refreshed_episodes_data)} episodes with scenes")
                        episodes_data = refreshed_episodes_data
            
            # Create proper left_section with latest user message and response
            left_section_chat_history = []
            
            # Get the full chat history from the database to properly deduplicate
            try:
                chat_session = chat_sessions.find_one({"session_id": req.session_id})
                if chat_session and "messages" in chat_session:
                    # Process messages to prevent duplicates
                    messages = chat_session["messages"]
                    seen_user_messages = set()
                    
                    for i, msg in enumerate(messages):
                        if msg.get("sender") == "user":
                            # For user messages, prevent duplicates by tracking content
                            msg_content = msg.get("message", "").strip()
                            if msg_content and hash(msg_content) not in seen_user_messages:
                                seen_user_messages.add(hash(msg_content))
                                left_section_chat_history.append({
                                    "sender": {
                                        "time": str(msg.get("timestamp", "")),
                                        "message": msg_content
                                    }
                                })
                                
                                # Find corresponding nexora response
                                if i < len(messages) - 1 and messages[i+1].get("sender") == "nexora":
                                    nexora_msg = messages[i+1]
                                    response_content = nexora_msg.get("message", "")
                                    
                                    # Don't add this message if it's the same as our current prompt
                                    # (prevents duplicating the last message when refreshing)
                                    if hash(msg_content) != hash(prompt.strip()):
                                        left_section_chat_history.append({
                                            "receiver": {
                                                "time": str(nexora_msg.get("timestamp", "")),
                                                "message": nexora_msg.get("script_overview", nexora_msg.get("thinking_process", response_content))
                                            }
                                        })
                else:
                    # If no chat history, use the existing chat_history
                    left_section_chat_history = chat_history.copy()
            except Exception as e:
                print(f"Error processing chat history: {e}")
                # Fallback to original chat history
                left_section_chat_history = chat_history.copy()
            
            # Now add the current request and response
            # First check if the current prompt is already the last sender message
            current_prompt = prompt.strip()
            if not left_section_chat_history or (
                left_section_chat_history and 
                ("sender" not in left_section_chat_history[-1] or 
                 left_section_chat_history[-1]["sender"].get("message") != current_prompt)
            ):
                # Add current prompt if not already the last message
                left_section_chat_history.append({
                    "sender": {
                        "time": str(datetime.utcnow()),
                        "message": current_prompt
                    }
                })
            
            # Add a response that's different from the script content
            if response_text and response_text.strip():
                # Check if response text looks like script content that we should summarize
                script_patterns = ["# Episode", "## Episode", "# Series Episodes", "## Scene"]
                is_likely_script = any(pattern in response_text for pattern in script_patterns)
                
                if is_likely_script:
                    # Generate a script overview instead of generic summary
                    script_overview = await generate_script_overview(response_text, prompt)
                    
                    left_section_chat_history.append({
                        "receiver": {
                            "time": str(datetime.utcnow()),
                            "message": script_overview
                        }
                    })
                else:
                    # Use the original response if it doesn't look like script content
                    left_section_chat_history.append({
                        "receiver": {
                            "time": str(datetime.utcnow()),
                            "message": response_text
                        }
                    })
            elif thinking_process and thinking_process.strip():
                # Generate script overview even for empty responses if possible
                if "create episode" in prompt.lower() or "scene" in prompt.lower():
                    script_overview = await generate_script_overview("", prompt)
                    left_section_chat_history.append({
                        "receiver": {
                            "time": str(datetime.utcnow()),
                            "message": script_overview
                        }
                    })
                else:
                    # Use thinking process as fallback only if not a script-related request
                    left_section_chat_history.append({
                        "receiver": {
                            "time": str(datetime.utcnow()),
                            "message": thinking_process
                        }
                    })
            
            # Create final response format with updated left_section
            response = {
                "prompt": req.prompt,
                "left_section": left_section_chat_history,
                "tabs": suggestions,
                "synopsis": project.get("synopsis", "") if project else "",
                "script": response_text,
                "character": character_data,
                "storyboard": refreshed_project.get("story_data", {}).get("storyboard", []) if refreshed_project else [],
                "episodes": episodes_data
            }
            
            # Check if this is specifically an episode request and we don't have script content
            if "create episode" in prompt.lower() and not response_text.strip():
                # Extract episode number from the prompt
                episode_match = re.search(r'episode\s*(\d+)', prompt.lower())
                if episode_match:
                    episode_number = int(episode_match.group(1))
                    print(f"Empty script for episode {episode_number} request, attempting to retrieve from database")
                    
                    # Try to get episode details from the database
                    episode_str = str(episode_number)
                    episode_script = ""
                    
                    # First check if we have structured_scenes
                    if refreshed_project and "story_data" in refreshed_project and "structured_scenes" in refreshed_project["story_data"]:
                        scenes = refreshed_project["story_data"]["structured_scenes"].get(episode_str, [])
                        if scenes:
                            print(f"Found {len(scenes)} structured scenes for episode {episode_number}")
                            
                            # Find the episode title
                            episode_title = f"Episode {episode_number}"
                            if "episodes" in refreshed_project["story_data"]:
                                for ep in refreshed_project["story_data"]["episodes"]:
                                    if ep.get("episode_number") == episode_number:
                                        episode_title = ep.get("episode_title", f"Episode {episode_number}")
                                        break
                            
                            episode_script = f"# Episode {episode_number}: {episode_title}\n\n"
                            
                            # Generate descriptions if they're missing or placeholders
                            for scene in scenes:
                                scene_num = scene.get("scene_number", 0)
                                scene_title = scene.get("title", f"Scene {scene_num}")
                                scene_desc = scene.get("description", "")
                                
                                # If missing or placeholder, generate a scene description
                                if not scene_desc or scene_desc == "Auto-extracted from script" or "In this scene, Aman and Priya explore" in scene_desc:
                                    # Get character names from the state
                                    character_names = []
                                    if "character_profiles" in refreshed_project["story_data"] and refreshed_project["story_data"]["character_profiles"]:
                                        for profile in refreshed_project["story_data"]["character_profiles"]:
                                            if isinstance(profile, dict) and "name" in profile:
                                                character_names.append(profile["name"])
                                    
                                    # If no character profiles, use character string descriptions
                                    if not character_names and "characters" in refreshed_project["story_data"] and refreshed_project["story_data"]["characters"]:
                                        for char_desc in refreshed_project["story_data"]["characters"]:
                                            if isinstance(char_desc, str) and "," in char_desc:
                                                char_name = char_desc.split(",")[0].strip()
                                                character_names.append(char_name)
                                    
                                    # Use default names if none found
                                    if not character_names:
                                        character_names = ["Aman", "Priya"]
                                    
                                    # Extract location from scene title if possible
                                    location_keywords = ["cafe", "market", "temple", "garden", "park", "restaurant", "museum", 
                                                        "fort", "palace", "street", "hotel", "home", "apartment", "river", "mall"]
                                    location = ""
                                    for keyword in location_keywords:
                                        if keyword in scene_title.lower():
                                            location = keyword
                                            break
                                    
                                    if not location:
                                        location = "Delhi"
                                    
                                    # Create a detailed scene description
                                    scene_desc = f"In this captivating scene at the {location.title()} in Delhi, {' and '.join(character_names)} experience a pivotal moment in their journey. "
                                    scene_desc += f"The vibrant atmosphere of Delhi comes alive with rich sensory details—the mingling of spices in the air, colorful textiles adorning nearby stalls, and the melodic sounds of street musicians. "
                                    scene_desc += f"As they explore {scene_title.lower()}, their relationship deepens through meaningful conversation and shared experiences. "
                                    scene_desc += f"\n\n{character_names[0]} notices how the warm afternoon light catches in {character_names[1]}'s eyes, creating a moment of connection amidst the bustling city. "
                                    scene_desc += f"They navigate through the crowds, occasionally stopping to admire the intricate architecture or sample local delicacies from street vendors. "
                                    scene_desc += f"Their adventure allows them to discover not just the beauty of Delhi, but new dimensions of their relationship. "
                                    scene_desc += f"\n\n\"This is exactly what we needed,\" {character_names[0]} says, taking {character_names[1]}'s hand as they continue their exploration. "
                                    scene_desc += f"The scene concludes with a meaningful exchange of glances that speaks volumes about their evolving connection and the memories they're creating together in this enchanting city."
                                    
                                    # Update the scene in the database
                                    try:
                                        story_projects.update_one(
                                            {"session_id": req.session_id},
                                            {"$set": {
                                                f"story_data.structured_scenes.{episode_str}.{scene_num-1}.description": scene_desc
                                            }}
                                        )
                                        print(f"Updated scene {scene_num} description in database")
                                    except Exception as e:
                                        print(f"Error updating scene description: {e}")
                                
                                episode_script += f"## Scene {scene_num}: {scene_title}\n\n{scene_desc}\n\n"
                    
                    # If no structured scenes, try episode_scripts
                    if not episode_script and refreshed_project and "story_data" in refreshed_project and "episode_scripts" in refreshed_project["story_data"]:
                        scenes = refreshed_project["story_data"]["episode_scripts"].get(episode_str, [])
                        if scenes:
                            print(f"Found {len(scenes)} scenes in episode_scripts for episode {episode_number}")
                            episode_script = f"# Episode {episode_number}\n\n"
                            
                            # Format scenes into a script
                            for i, scene_text in enumerate(scenes):
                                episode_script += f"{scene_text}\n\n"
                    
                    # Use the constructed episode script if we found scenes
                    if episode_script:
                        print(f"Setting script content from database for episode {episode_number}")
                        response["script"] = episode_script
            
            # Special case for "create episodes" (plural) - show all episode summaries
            elif prompt.lower() in ["create episodes", "generate episodes"] and not response_text.strip():
                print("Processing 'create episodes' command - generating episode list for script field")
                
                if refreshed_project and "story_data" in refreshed_project and "episodes" in refreshed_project["story_data"]:
                    episodes = refreshed_project["story_data"]["episodes"]
                    if episodes:
                        print(f"Found {len(episodes)} episodes in database, formatting for display")
                        episodes_script = "# Series Episodes\n\n"
                        
                        # Sort episodes by episode number
                        sorted_episodes = sorted(episodes, key=lambda ep: ep.get("episode_number", 99))
                        
                        # Format each episode with its title and summary
                        for episode in sorted_episodes:
                            ep_num = episode.get("episode_number", 0)
                            ep_title = episode.get("episode_title", f"Episode {ep_num}")
                            ep_summary = episode.get("summary", "No summary available")
                            
                            episodes_script += f"## Episode {ep_num}: {ep_title}\n\n{ep_summary}\n\n"
                        
                        # Add the formatted episodes to the response
                        response["script"] = episodes_script
                        print("Set script field with episode summaries")
                    else:
                        print("No episodes found in database")
                else:
                    print("No episode data found in refreshed project")
            
            # Only include character data in the response if it has proper image URLs or is explicitly a character request
            is_character_request = any(term in req.prompt.lower() for term in ["create characters", "finalize characters", "finalise characters"])
            has_proper_characters = False
            
            if character_data and isinstance(character_data, list):
                # First check if we have an array of objects (not strings)
                objects_only = True
                for char in character_data:
                    if not isinstance(char, dict):
                        objects_only = False
                        break
                
                # Only proceed if we have an array of objects
                if objects_only:
                    # Check if any characters have image URLs
                    for char in character_data:
                        if char.get("reference_image"):
                            has_proper_characters = True
                            break
                    
                    # Only include characters if they have images OR this is a character request
                    if has_proper_characters or is_character_request:
                        response["character"] = character_data
                        print(f"Including {len(character_data)} characters in response")
                    else:
                        print("No character images found and not a character request, using empty array")
                else:
                    print("Character data contains non-dictionary elements, using empty array")
            
            return response
            
        except Exception as e:
            print(f"Error in send_chat_message (normal processing): {e}")
            import traceback
            print(traceback.format_exc())
            # Use HTTPException instead of JSONResponse for consistency
            raise HTTPException(status_code=500, detail=f"An error occurred during message processing: {str(e)}")

    except Exception as e:
        print(f"Error in send_chat_message: {e}")
        import traceback
        print(traceback.format_exc())
        # Use HTTPException instead of JSONResponse for consistency
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.patch("/chat/message")
async def edit_message(req: EditMessageRequest):
    # Find the session
    session = chat_sessions.find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find the message to update
    message_found = False
    updated_message = None
    
    for i, message in enumerate(session["messages"]):
        # Check if this message has a message_id that matches
        if message.get("message_id") == req.message_id:
            # Update the message
            message["message"] = req.new_message
            message["edited_at"] = datetime.utcnow()
            message["was_edited"] = True
            
            # If this is a Nexora message, also update thinking_process if available
            if message.get("sender") == "nexora" and "thinking_process" in message:
                # For now, we'll keep the thinking_process unchanged
                # In a future version, you might want to regenerate the thinking process or allow editing it
                pass
            
            updated_message = message
            message_found = True
            break
    
    if not message_found:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Update the message in the database
    chat_sessions.update_one(
        {"session_id": req.session_id},
        {"$set": {"messages": session["messages"]}}
    )
    
    # Return the updated message
    return {
        "success": True,
        "message": "Message updated successfully",
        "updated_message": updated_message
    }

@router.post("/refresh_session")
async def refresh_session(req: SendChatRequest):
    """
    Special endpoint to force a refresh of session data.
    This can be called when the frontend detects a synchronization issue.
    """
    try:
        # Use the refresh_project_data function to get latest data
        refreshed_project, refreshed_characters, refreshed_episodes = await refresh_project_data(req.session_id)
        
        if not refreshed_project:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get stored synopsis
        stored_synopsis = refreshed_project.get("synopsis", "")
        
        # Get chat history
        session = chat_sessions.find_one({"session_id": req.session_id})
        chat_history = []
        script_content = ""
        
        # Apply improved deduplication logic for chat history
        if session and "messages" in session:
            # Process messages to prevent duplicates
            messages = session["messages"]
            seen_user_messages = set()
            
            for i, msg in enumerate(messages):
                if msg.get("sender") == "user":
                    # For user messages, prevent duplicates by tracking content
                    msg_content = msg.get("message", "").strip()
                    if msg_content and hash(msg_content) not in seen_user_messages:
                        seen_user_messages.add(hash(msg_content))
                        chat_history.append({
                            "sender": {
                                "time": str(msg.get("timestamp", "")),
                                "message": msg_content
                            }
                        })
                        
                        # Find corresponding nexora response
                        if i < len(messages) - 1 and messages[i+1].get("sender") == "nexora":
                            nexora_msg = messages[i+1]
                            response_content = nexora_msg.get("message", "")
                            
                            # Save script content from the most recent response
                            script_content = response_content
                            
                            # Use thinking process for display if available
                            display_content = nexora_msg.get("thinking_process", response_content)
                            
                            # For episode or scene creation responses, use a shorter confirmation
                            if response_content.startswith("# Episode") or "# Series Episodes" in response_content:
                                # Detect what was created
                                if "create episodes" in msg_content.lower():
                                    display_content = "All episodes have been created successfully."
                                elif "create episode" in msg_content.lower():
                                    episode_match = re.search(r'episode\s+(\d+)', msg_content.lower())
                                    episode_num = episode_match.group(1) if episode_match else ""
                                    display_content = f"Episode {episode_num} has been created successfully."
                                elif "scene" in msg_content.lower():
                                    scene_match = re.search(r'scene\s+(\d+)', msg_content.lower())
                                    scene_num = scene_match.group(1) if scene_match else ""
                                    display_content = f"Scene {scene_num} has been created successfully."
                            
                            chat_history.append({
                                "receiver": {
                                    "time": str(nexora_msg.get("timestamp", "")),
                                    "message": display_content
                                }
                            })
        
        # If no episodes found but script has episode content, try to extract them
        if (not refreshed_episodes or len(refreshed_episodes) == 0) and "Episode" in script_content and "Scene" in script_content:
            print("No episodes found in database, attempting to extract from script content")
            
            # Extract episode number
            episode_match = re.search(r'Episode\s*(\d+)', script_content)
            if episode_match:
                episode_number = int(episode_match.group(1))
                print(f"Found Episode {episode_number} in script content")
                
                # Create episode entry
                episode_entry = {
                    "title": f"Episode {episode_number}",
                    "prompt": f"create episode {episode_number}",
                    "child": []
                }
                
                # Find scenes
                scene_matches = re.findall(r'Scene\s*(\d+):\s*([^\n]+)', script_content)
                if scene_matches:
                    print(f"Found {len(scene_matches)} scenes in script content")
                    
                    for scene_num_str, scene_title in scene_matches:
                        try:
                            scene_number = int(scene_num_str)
                        except ValueError:
                            scene_number = len(episode_entry["child"]) + 1
                            
                        scene_entry = {
                            "title": scene_title.strip(),
                            "prompt": f"create scene {scene_number} episode {episode_number}"
                        }
                        episode_entry["child"].append(scene_entry)
                
                # Update the refreshed_episodes list
                refreshed_episodes = [episode_entry]
                print(f"Created episode entry with {len(episode_entry['child'])} scenes from script content")
                
                # Also update the database with this information
                story_data = refreshed_project.get("story_data", {})
                
                # Create structured episodes array
                if "episodes" not in story_data or not story_data["episodes"]:
                    story_data["episodes"] = [{
                        "episode_number": episode_number,
                        "episode_title": f"Episode {episode_number}",
                        "summary": "Auto-extracted from script"
                    }]
                
                # Create structured scenes
                if "structured_scenes" not in story_data:
                    story_data["structured_scenes"] = {}
                
                # Add scenes for this episode
                structured_scenes = []
                for i, scene in enumerate(episode_entry["child"]):
                    scene_number = i + 1
                    scene_title = scene.get("title", f"Scene {scene_number}")
                    
                    # Extract scene description from the script
                    scene_desc_pattern = rf'Scene\s*{scene_number}:[^\n]*\n\n(.*?)(?=Scene\s*\d+:|$)'
                    scene_desc_match = re.search(scene_desc_pattern, script_content, re.DOTALL)
                    
                    if scene_desc_match:
                        scene_desc = scene_desc_match.group(1).strip()
                    else:
                        # Generate a more meaningful description based on scene title and character information
                        character_names = []
                        
                        # Get character names from the project data
                        if "character_profiles" in refreshed_story_data and refreshed_story_data["character_profiles"]:
                            for profile in refreshed_story_data["character_profiles"]:
                                if isinstance(profile, dict) and "name" in profile:
                                    character_names.append(profile["name"])
                        
                        # If no character profiles, try character string descriptions
                        if not character_names and "characters" in refreshed_story_data and refreshed_story_data["characters"]:
                            for char_desc in refreshed_story_data["characters"]:
                                if isinstance(char_desc, str) and "," in char_desc:
                                    char_name = char_desc.split(",")[0].strip()
                                    character_names.append(char_name)
                        
                        # Use default names if none found
                        if not character_names:
                            character_names = ["Aman", "Priya"]
                        
                        # Extract location from scene title if possible
                        location_keywords = ["cafe", "market", "temple", "garden", "park", "restaurant", "museum", 
                                            "fort", "palace", "street", "hotel", "home", "apartment", "river", "mall"]
                        location = ""
                        for keyword in location_keywords:
                            if keyword in scene_title.lower():
                                location = keyword
                                break
                        
                        if not location:
                            location = "Delhi"
                        
                        # Create a detailed scene description
                        scene_desc = f"In this captivating scene at the {location.title()} in Delhi, {' and '.join(character_names)} experience a pivotal moment in their journey. "
                        scene_desc += f"The vibrant atmosphere of Delhi comes alive with rich sensory details—the mingling of spices in the air, colorful textiles adorning nearby stalls, and the melodic sounds of street musicians. "
                        scene_desc += f"As they explore {scene_title.lower()}, their relationship deepens through meaningful conversation and shared experiences. "
                        scene_desc += f"\n\n{character_names[0]} notices how the warm afternoon light catches in {character_names[1]}'s eyes, creating a moment of connection amidst the bustling city. "
                        scene_desc += f"They navigate through the crowds, occasionally stopping to admire the intricate architecture or sample local delicacies from street vendors. "
                        scene_desc += f"Their adventure allows them to discover not just the beauty of Delhi, but new dimensions of their relationship. "
                        scene_desc += f"\n\n\"This is exactly what we needed,\" {character_names[0]} says, taking {character_names[1]}'s hand as they continue their exploration. "
                        scene_desc += f"The scene concludes with a meaningful exchange of glances that speaks volumes about their evolving connection and the memories they're creating together in this enchanting city."
                    
                    structured_scenes.append({
                        "scene_number": scene_number,
                        "title": scene_title,
                        "description": scene_desc
                    })
                
                # Update database
                story_data["structured_scenes"][str(episode_number)] = structured_scenes
                story_projects.update_one(
                    {"session_id": req.session_id},
                    {"$set": {
                        "story_data.episodes": story_data["episodes"],
                        "story_data.structured_scenes": story_data["structured_scenes"],
                        "updated_at": datetime.utcnow()
                    }}
                )
                print("Updated database with extracted episode and scene information")
        
        # Create a UI-ready response with refreshed data
        response = {
            "prompt": req.prompt,
            "left_section": chat_history,
            "tabs": await generate_chat_suggestions(req.session_id),
            "synopsis": project.get("synopsis", "") if project else "",
            "script": script_content,  # Include latest script content
            "character": refreshed_characters if refreshed_characters else [],
            "storyboard": refreshed_project.get("story_data", {}).get("storyboard", []) if refreshed_project else [],
            "episodes": refreshed_episodes if refreshed_episodes else []
        }
        
        # Only include characters if they're properly formatted with image URLs
        if refreshed_characters and isinstance(refreshed_characters, list):
            # Check if any character has an image URL
            has_images = False
            for char in refreshed_characters:
                if isinstance(char, dict) and (char.get("reference_image") or char.get("reference_image")):
                    has_images = True
                    break
            
            # Only use characters if they have images
            if has_images:
                # Filter to only include required fields
                filtered_characters = []
                for char in refreshed_characters:
                    filtered_char = {
                        "name": char.get("name", ""),
                        "token": char.get("token", ""),
                        "description": char.get("description", ""),
                        "reference_image": char.get("reference_image", ""),
                        "gender": char.get("gender", "unknown")
                    }
                    # Ensure only these fields are included
                    filtered_characters.append(filtered_char)
                response["character"] = filtered_characters
            else:
                print("Characters have no image URLs - returning empty character array")
        
        return response
    except Exception as e:
        import traceback
        print(f"Error in refresh_session: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error refreshing session: {str(e)}")

# Add this helper function near the other helper functions
def _ensure_character_profile_fields(profile):
    """
    Ensure a character profile has all the required fields.
    This helps prevent validation errors when saving to database.
    """
    required_fields = {
        "name": "",
        "description": "",
        "gender": "unknown", 
        "reference_image": "",
        "token": ""
    }
    
    # Set default values for any missing fields
    for field, default in required_fields.items():
        if field not in profile or profile[field] is None:
            profile[field] = default
    
    # Ensure token exists - important for frontend
    if not profile.get("token"):
        profile["token"] = f"character-{uuid.uuid4().hex[:8]}"
    
    # Remove image_url field if it exists to prevent duplication
    if "image_url" in profile:
        # If reference_image is empty but image_url has value, copy it over
        if not profile.get("reference_image") and profile["image_url"]:
            profile["reference_image"] = profile["image_url"]
        # Then remove the image_url field
        profile.pop("image_url")
    
    return profile

# Add this new function after other formatting functions (around line 250)
def format_storyboard_as_markdown(storyboard):
    try:
        if not storyboard:
            return "No storyboard available."
            
        # Organize storyboard items by episode and scene
        organized_items = {}
        for item in storyboard:
            ep_num = item.get("episode_number", 0)
            scene_num = item.get("scene_number", 0)
            
            if ep_num not in organized_items:
                organized_items[ep_num] = {}
                
            if scene_num not in organized_items[ep_num]:
                organized_items[ep_num][scene_num] = []
                
            organized_items[ep_num][scene_num].append(item)
            
        # Build markdown
        markdown = "# Scene Storyboard\n\n"
        
        for ep_num in sorted(organized_items.keys()):
            for scene_num in sorted(organized_items[ep_num].keys()):
                shots = organized_items[ep_num][scene_num]
                
                markdown += f"## Episode {ep_num}, Scene {scene_num}\n\n"
                
                for shot in shots:
                    shot_num = shot.get("shot_number", 0)
                    description = shot.get("description", "")
                    image_url = shot.get("image_url", "")
                    
                    markdown += f"### Shot {shot_num}\n\n"
                    markdown += f"{description}\n\n"
                    
                    if image_url:
                        markdown += f"![Shot {shot_num}]({image_url})\n\n"
                    
        return markdown
    except Exception as e:
        print(f"Error formatting storyboard: {e}")
        return "Error formatting storyboard."

# Fix the generate_script_overview function to handle episode lists better
async def generate_script_overview(response_text: str, prompt: str) -> str:
    """Generate a concise overview of script content instead of showing thinking process"""
    try:
        # Check if this is an episode creation request
        if "create episode" in prompt.lower() and "scene" not in prompt.lower():
            episode_match = re.search(r'episode\s+(\d+)', prompt.lower())
            episode_number = episode_match.group(1) if episode_match else ""
            
            # Extract scenes from response text
            scene_matches = re.findall(r'## Scene (\d+):\s*([^\n]+)', response_text)
            
            if scene_matches:
                overview = f"## Episode {episode_number} Overview\n\n"
                overview += f"This episode contains {len(scene_matches)} scenes:\n\n"
                
                for scene_num, scene_title in scene_matches:
                    overview += f"- **Scene {scene_num}**: {scene_title}\n"
                
                return overview
            else:
                return f"Episode {episode_number} has been created successfully."
        
        # Check if this is a scene creation request
        elif "scene" in prompt.lower() and "episode" in prompt.lower():
            scene_match = re.search(r'scene\s+(\d+)', prompt.lower())
            episode_match = re.search(r'episode\s+(\d+)', prompt.lower())
            
            scene_number = scene_match.group(1) if scene_match else ""
            episode_number = episode_match.group(1) if episode_match else ""
            
            # Check if there are shots in the response
            shot_count = len(re.findall(r'## Shot \d+', response_text))
            
            if shot_count > 0:
                return f"## Scene {scene_number} from Episode {episode_number}\n\nThis scene contains {shot_count} cinematic shots with detailed camera directions and dialogue."
            else:
                return f"Scene {scene_number} for Episode {episode_number} has been created successfully."
        
        # Check if this is a "create episodes" (plural) request
        elif prompt.lower() in ["create episodes", "generate episodes"]:
            # Extract episode titles and summaries from the response_text
            episode_sections = re.findall(r'## Episode (\d+): ([^\n]+)\n\n([^#]+)', response_text, re.DOTALL)
            
            if episode_sections:
                episode_count = len(episode_sections)
                overview = f"## Series Episodes Created\n\n"
                overview += f"{episode_count} episodes have been created:\n\n"
                
                # Add the first 5 episodes as a preview
                for i, (ep_num, ep_title, ep_summary) in enumerate(episode_sections):
                    if i < 5:  # Limit to first 5 episodes to keep message reasonable
                        # Format summary to first sentence or 50 characters
                        short_summary = ep_summary.split('.')[0].strip() + "."
                        if len(short_summary) > 70:
                            short_summary = short_summary[:67] + "..."
                            
                        overview += f"- **Episode {ep_num}**: {ep_title} - {short_summary}\n"
                
                # Add a note if there are more episodes
                if episode_count > 5:
                    overview += f"\n... and {episode_count - 5} more episodes.\n"
                    
                overview += "\nYou can now create individual episodes or work with specific scenes."
                return overview
            else:
                # Fallback to checking for episode sections with different pattern
                episode_titles = re.findall(r'## Episode (\d+): ([^\n]+)', response_text)
                if episode_titles:
                    episode_count = len(episode_titles)
                    overview = f"## Series Episodes Created\n\n"
                    overview += f"{episode_count} episodes have been created:\n\n"
                    
                    for i, (ep_num, ep_title) in enumerate(episode_titles):
                        if i < 5:  # Limit to first 5
                            overview += f"- **Episode {ep_num}**: {ep_title}\n"
                    
                    if episode_count > 5:
                        overview += f"\n... and {episode_count - 5} more episodes.\n"
                        
                    return overview
                
                # If all else fails, extract info from the episodes_data
                return "Series episodes have been created successfully. See the episodes tab for details."
                
        # For character creation/finalization
        elif any(term in prompt.lower() for term in ["create character", "finalize character", "finalise character"]):
            # Count how many characters are defined
            character_count = len(re.findall(r'# Character', response_text))
            
            if character_count > 0:
                return f"Characters have been created successfully. {character_count} characters are now available."
            else:
                return "Characters have been finalized successfully."
        
        # Default case - if we can't create a specific overview, return a summary of the script
        if response_text:
            # Try to extract a meaningful title or first heading
            title_match = re.search(r'# ([^\n]+)', response_text)
            if title_match:
                title = title_match.group(1).strip()
                return f"Created: {title}"
            else:
                return "Content has been created successfully."
        else:
            return "Your request has been processed successfully."
            
    except Exception as e:
        print(f"Error generating script overview: {e}")
        import traceback
        print(traceback.format_exc())
        return "Your request has been processed successfully."
