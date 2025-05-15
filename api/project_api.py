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
    """Generate action-oriented suggestions based on Producer agent logic"""
    try:
        # If no recent messages provided, fetch from database
        if not recent_messages:
            session = chat_sessions.find_one({"session_id": session_id})
            if not session or "messages" not in session:
                return ["Generate episodes", "Create characters", "Change story"]
            
            # Get only user and nexora messages (not thinking messages)
            recent_messages = [
                msg for msg in session["messages"] 
                if msg.get("sender") in ["user", "nexora"]
            ]
            
            # Limit to last 6 messages for context
            recent_messages = recent_messages[-6:]
        
        # Get the most recent AI response to use as context
        ai_responses = [msg for msg in recent_messages if msg.get("sender") == "nexora"]
        
        # Check if episodes have been generated yet
        project = story_projects.find_one({"session_id": session_id})
        has_episodes = False
        if project and project.get("story_data"):
            story_data = project.get("story_data")
            if "episodes" in story_data and story_data["episodes"]:
                has_episodes = True
        
        # Return appropriate suggestions based on current state
        if not has_episodes:
            return [
                "Generate episodes",
                "Create characters",
                "Change story"
            ]
        else:
            return [
                "Create character visuals",
                "Generate scene images",
                "Change story"
            ]
            
    except Exception as e:
        print(f"Error generating chat suggestions: {e}")
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
    Re-fetch project data from the database to ensure we have the most up-to-date information.
    This helps address synchronization issues where database updates don't immediately appear in the API response.
    """
    print(f"Refreshing project data for session {session_id} from database")
    
    # Increase delay to ensure database writes have completed
    await asyncio.sleep(1.0)  # Increased from 0.5 to 1.0 seconds
    
    # Re-fetch the project from the database
    refreshed_project = story_projects.find_one({"session_id": session_id})
    
    if not refreshed_project:
        print(f"Warning: Could not find project for session {session_id} during refresh")
        return None, None, None
    
    # Get the refreshed story data
    refreshed_story_data = refreshed_project.get("story_data", {})
    print(f"DEBUG: Refreshed story data keys: {list(refreshed_story_data.keys())}")
    
    # Get refreshed character data
    refreshed_character_data = []
    if refreshed_story_data and "characters" in refreshed_story_data:
        print(f"DEBUG: Found {len(refreshed_story_data['characters'])} characters in refreshed data")
        refreshed_character_data = refreshed_story_data["characters"]
    elif refreshed_story_data and "character_profiles" in refreshed_story_data:
        character_profiles = refreshed_story_data["character_profiles"]
        print(f"DEBUG: Found {len(character_profiles)} character profiles in refreshed data")
        
        # Fix character profiles if they are stored as strings instead of dictionaries
        fixed_profiles = []
        for profile in character_profiles:
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
                
                # Include token information
                if "character_map" in refreshed_story_data:
                    character_map = refreshed_story_data["character_map"] 
                    if isinstance(character_map, dict) and name in character_map:
                        fixed_profile["token"] = character_map[name]
                
                # Ensure all required fields are present
                fixed_profile = _ensure_character_profile_fields(fixed_profile)
                
                # Add to fixed profiles
                fixed_profiles.append(fixed_profile)
                print(f"Fixed string character profile: {name}")
            else:
                # Build the simplified profile with only needed fields
                filtered_profile = {
                    "name": profile.get("name", ""),
                    "reference_image": profile.get("reference_image", ""),
                    "description": profile.get("description", ""),
                    "gender": profile.get("gender", "")
                }
                
                # Include token information
                if "character_map" in refreshed_story_data:
                    character_map = refreshed_story_data["character_map"]
                    char_name = profile.get("name", "")
                    if isinstance(character_map, dict) and char_name in character_map:
                        filtered_profile["token"] = character_map[char_name]
                elif "token" in profile:
                    filtered_profile["token"] = profile["token"]
                
                # Apply field validation
                filtered_profile = _ensure_character_profile_fields(filtered_profile)
                
                fixed_profiles.append(filtered_profile)
            
        # Use the fixed profiles
        refreshed_character_data = fixed_profiles
        print(f"Processed {len(refreshed_character_data)} character profiles after fixing")

    # Build episodes data structure
    refreshed_episodes_data = []
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
            
            # Then check episode_scripts as fallback
            if not has_scenes and "episode_scripts" in refreshed_story_data and episode_str in refreshed_story_data["episode_scripts"]:
                scenes = refreshed_story_data["episode_scripts"][episode_str]
                print(f"DEBUG: Found {len(scenes)} scenes in episode_scripts for episode {episode_number}")
                for i, scene_desc in enumerate(scenes):
                    scene_number = i + 1
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
            
            refreshed_episodes_data.append(episode_entry)
    else:
        print(f"DEBUG: No episodes found in refreshed data. Keys: {list(refreshed_story_data.keys() if refreshed_story_data else [])}")
    
    print(f"Refreshed data: {len(refreshed_character_data)} characters, {len(refreshed_episodes_data)} episodes")
    return refreshed_project, refreshed_character_data, refreshed_episodes_data

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
                
                # Create response with thinking process
                streamed_left_section = chat_history.copy()
                streamed_left_section.append({
                    "receiver": {
                        "time": str(datetime.utcnow()),
                        "message": thinking_process
                    }
                })
                
                # Get suggestions for the UI
                suggestions = await generate_chat_suggestions(req.session_id)
                
                # Create Nexora response message with thinking process only
                nexora_message = {
                    "sender": "nexora",
                    "message": thinking_process,  # Only storing thinking process, not full content
                    "timestamp": datetime.utcnow(),
                    "response_type": "thinking",
                    "message_id": str(uuid.uuid4()),
                    "thinking_process": thinking_process
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
            
            # Get suggestions for the UI
            suggestions = await generate_chat_suggestions(req.session_id)
            
            # Create Nexora response message
            # IMPORTANT: We'll store the full response in the database but only show thinking process in UI
            nexora_message = {
                "sender": "nexora",
                "message": response_text,  # Store full response in DB
                "timestamp": datetime.utcnow(),
                "response_type": determine_response_type(response_text),
                "message_id": str(uuid.uuid4()),
                "thinking_process": thinking_process  # Add thinking process to the message
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
                    
                    story_projects.update_one(
                        {"session_id": req.session_id},
                        {"$set": update_data}
                    )
                    print(f"Updated database with episode {episode_number} and {len(structured_scenes)} scenes")
                    
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
                
                # Refresh from database to ensure we have the most up-to-date episode information
                print("Episode request detected - refreshing data from database")
                refreshed_project, refreshed_character_data, refreshed_episodes_data = await refresh_project_data(req.session_id)
                
                if refreshed_episodes_data:
                    print(f"Using refreshed episodes data: {len(refreshed_episodes_data)} episodes with scenes")
                    episodes_data = refreshed_episodes_data
                
                if refreshed_character_data and not character_data:
                    print(f"Using refreshed character data: {len(refreshed_character_data)} characters")
                    character_data = refreshed_character_data
            
            # Create final response format
            response = {
                "prompt": req.prompt,
                "left_section": chat_history,
                "tabs": suggestions,
                "synopsis": project.get("synopsis", "") if project else "",
                "script": response_text,
                "character": character_data,
                "storyboard": refreshed_project.get("story_data", {}).get("storyboard", []) if refreshed_project else [],
                "episodes": episodes_data
            }
            
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
        
        if session and "messages" in session:
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
                    # Save latest script content for extracting episode data if needed
                    script_content = msg.get("message", "")
        
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
                    scene_desc = scene_desc_match.group(1).strip() if scene_desc_match else "Auto-extracted from script"
                    
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
