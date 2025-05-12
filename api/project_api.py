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

from nexora_engine import run_producer_stream
from agents.producer import producer_agent
from agents.writer import writer_agent
from agents.casting import casting_agent
from agents.ad import ad_agent
from agents.director import director_agent
from agents.video_design import video_design_agent
from agents.editor import editor_agent

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
                return ["Tell me more", "Create Characters", "Show Episodes"]
            
            # Get only user and nexora messages (not thinking messages)
            recent_messages = [
                msg for msg in session["messages"] 
                if msg.get("sender") in ["user", "nexora"]
            ]
            
            # Limit to last 6 messages for context
            recent_messages = recent_messages[-6:]
        
        # Get the most recent AI response to use as context for Producer
        ai_responses = [msg for msg in recent_messages if msg.get("sender") == "nexora"]
        if not ai_responses:
            return ["Create Characters", "Create Scene Images"]
        
        last_ai_response = ai_responses[-1].get("message", "")
        
        # Generate suggestions based on the agent type
        return [
            "Develop the story",
            "Create characters",
            "Generate visuals"
        ]
    except Exception as e:
        print(f"Error generating chat suggestions: {e}")
        return ["Create Characters", "Create Scene Images"]

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

@router.post("/chat/start")
async def start_chat(req: StartChatRequest):
    print(f"Creating new chat session")
    session_id = str(uuid.uuid4())
    user_id = "user123"  # In real system, this would come from auth
    
    # Create initial document in story_projects
    story_projects.insert_one({
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "story_data": {}
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
    
    if prompt:
        print(f"Processing prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Processing prompt: {prompt}")
        
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
            print(f"No prompt provided and not first call, will only return existing data")
    
    # Create state object for processing
    print(f"Creating StoryState object")
    state = StoryState()
    
    # Try to load any existing story data
    print(f"Attempting to load story data from database")
    project = story_projects.find_one({"session_id": req.session_id})
    if project and project.get("story_data"):
        print(f"Found story data in project, loading into state")
        state = StoryState(**project.get("story_data"))
        print(f"Story data loaded successfully")
    else:
        print(f"No story data found for this session or empty data")
    
    # Prepare response structure
    initial_response = {
        "prompt": prompt,
        "left_section": chat_history[:-1] if chat_history and len(chat_history) > 1 else [],
        "tabs": [],
        "synopsis": "",
        "script": "",
        "character": [],
        "storyboard": None
    }
    
    # If there's no prompt, just return the initial response with existing data
    if not prompt:
        return initial_response
    
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
        
        # For the left_section in the UI, we'll only show the thinking process, not the full response
        # This is the key change being made - we're not adding the response_text to the left_section
        
        # Get character data if available
        character_data = []
        if project and project.get("story_data") and "character_profiles" in project.get("story_data", {}):
            character_profiles = project["story_data"]["character_profiles"]
            print(f"Found {len(character_profiles)} character profiles in database")
            character_data = character_profiles
        
        # Create final response
        final_response = {
            "prompt": prompt,
            "left_section": streamed_left_section,  # This now has thinking_process instead of full response
            "tabs": suggestions,
            "synopsis": state.logline if hasattr(state, "logline") else "",
            "script": response_text,  # Full script is still sent in the script field
            "character": character_data,
            "storyboard": None
        }
        
        print(f"Returning complete JSON response")
        return final_response
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
