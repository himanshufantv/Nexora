from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from utils.types import StoryState
from engine.runner import run_agent
from openai import OpenAI

import os
import uuid
import asyncio
import json
import re

from nexora_engine import run_producer_stream

load_dotenv()

router = APIRouter()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["StudioNexora"]
story_projects = db["story_projects"]
chat_sessions = db["chat_sessions"]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class StartChatRequest(BaseModel):
    message: str

class SendChatRequest(BaseModel):
    session_id: str
    message: str

class EditMessageRequest(BaseModel):
    session_id: str
    message_id: str
    new_message: str

# Helper function to extract episodes from text response
def extract_episodes_from_response(response_text: str) -> list:
    """
    Extract episodes from the response text and return a list of episode objects
    """
    episodes = []
    
    try:
        # First, check if the response is valid JSON
        if response_text.strip().startswith('{') or response_text.strip().startswith('['):
            try:
                parsed_json = json.loads(response_text)
                
                # Case 1: Response is a JSON object with an "episodes" field
                if isinstance(parsed_json, dict) and "episodes" in parsed_json:
                    episodes_data = parsed_json["episodes"]
                    series_title = parsed_json.get("series_title", "Untitled Series")
                    
                    for i, episode in enumerate(episodes_data):
                        episode_obj = {
                            "episode_number": i + 1,
                            "series_title": series_title,
                            "title": episode.get("episode_title", f"Episode {i+1}"),
                            "summary": episode.get("summary", "No summary available"),
                            "full_data": episode
                        }
                        episodes.append(episode_obj)
                
                # Case 2: Response is a JSON array of episodes
                elif isinstance(parsed_json, list) and len(parsed_json) > 0:
                    for i, item in enumerate(parsed_json):
                        # Check if item looks like an episode (has title or similar fields)
                        if isinstance(item, dict) and ("episode_title" in item or "title" in item or "summary" in item):
                            title = item.get("episode_title") or item.get("title", f"Episode {i+1}")
                            episode_obj = {
                                "episode_number": i + 1,
                                "series_title": "Untitled Series",
                                "title": title,
                                "summary": item.get("summary", "No summary available"),
                                "full_data": item
                            }
                            episodes.append(episode_obj)
                
            except json.JSONDecodeError:
                # Not valid JSON, try regex parsing
                pass
        
        # If no episodes found via JSON parsing, try regex extraction
        if not episodes:
            # Look for episode patterns like "Episode 1: Title" or similar
            episode_patterns = [
                r'Episode\s+(\d+):\s*([^\n]+)(?:\n+([^#]+))?',  # Episode 1: Title followed by summary
                r'(\d+)\.\s+([^\n]+)(?:\n+([^#]+))?',  # 1. Title followed by summary
                r'"episode_title":\s*"([^"]+)".*?"summary":\s*"([^"]+)"'  # JSON-like format
            ]
            
            for pattern in episode_patterns:
                matches = re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                for i, match in enumerate(matches):
                    if len(match.groups()) >= 2:
                        # First group is episode number or empty, second is title, third (if exists) is summary
                        try:
                            episode_num = int(match.group(1)) if match.group(1).isdigit() else i + 1
                        except (IndexError, AttributeError):
                            episode_num = i + 1
                            
                        try:
                            title = match.group(2).strip()
                        except (IndexError, AttributeError):
                            title = f"Episode {episode_num}"
                            
                        try:
                            summary = match.group(3).strip() if len(match.groups()) > 2 else "No summary available"
                        except (IndexError, AttributeError):
                            summary = "No summary available"
                        
                        episode_obj = {
                            "episode_number": episode_num,
                            "series_title": "Extracted Series",
                            "title": title,
                            "summary": summary,
                            "full_data": {
                                "episode_title": title,
                                "summary": summary
                            }
                        }
                        episodes.append(episode_obj)
                
                # If we found episodes with this pattern, no need to try other patterns
                if episodes:
                    break
    
    except Exception as e:
        print(f"Error extracting episodes: {e}")
    
    return episodes

# Helper function to determine response type
def determine_response_type(content: str) -> str:
    """
    Determine the response type based on the content
    
    Returns one of: "string", "image", "video", "audio", "file", "mixed"
    """
    # Default type
    response_type = "string"
    
    # Check for image URLs or image markdown
    image_patterns = [
        r'https?://\S+\.(?:jpg|jpeg|png|gif|bmp|svg|webp)',  # Image URLs
        r'!\[.*?\]\(.*?\)',  # Markdown image syntax
        r'<img\s+src=[\'"].*?[\'"]'  # HTML image tag
    ]
    
    # Check for video URLs
    video_patterns = [
        r'https?://\S+\.(?:mp4|avi|mov|wmv|flv|mkv|webm)',  # Video URLs
        r'https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+',  # YouTube links
        r'<video\s+src=[\'"].*?[\'"]'  # HTML video tag
    ]
    
    # Check for audio URLs
    audio_patterns = [
        r'https?://\S+\.(?:mp3|wav|ogg|aac|flac)',  # Audio URLs
        r'<audio\s+src=[\'"].*?[\'"]'  # HTML audio tag
    ]
    
    # Check for file URLs
    file_patterns = [
        r'https?://\S+\.(?:pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|txt)'  # File URLs
    ]
    
    # Check for each pattern type
    has_image = any(re.search(pattern, content, re.IGNORECASE) for pattern in image_patterns)
    has_video = any(re.search(pattern, content, re.IGNORECASE) for pattern in video_patterns)
    has_audio = any(re.search(pattern, content, re.IGNORECASE) for pattern in audio_patterns)
    has_file = any(re.search(pattern, content, re.IGNORECASE) for pattern in file_patterns)
    
    # Determine content type
    content_types = []
    if has_image:
        content_types.append("image")
    if has_video:
        content_types.append("video")
    if has_audio:
        content_types.append("audio")
    if has_file:
        content_types.append("file")
    
    # If multiple types, return "mixed"
    if len(content_types) > 1:
        return "mixed"
    # If one type, return that
    elif content_types:
        return content_types[0]
    # Otherwise, it's just text
    else:
        return "string"

# Generate thinking process explanation
async def generate_thinking_process(user_message: str) -> str:
    """Generate a thinking process explanation for how GPT will approach the task"""
    try:
        prompt = f"""
You are an AI assistant that explains its thinking process.
A user has asked: "{user_message}"

Write a brief 2-3 sentence explanation of how you would approach this task, including the steps you would take.
KEEP IT BRIEF AND FOCUSED ON THE PROCESS, NOT THE ACTUAL ANSWER.
Start your explanation with "I will..."
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller, faster model for the thinking part
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        thinking = response.choices[0].message.content.strip()
        return thinking
    except Exception as e:
        print(f"Error generating thinking process: {e}")
        return f"I will help you with: '{user_message}'"

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
        
        # Use the Producer agent to determine which agent should handle the next interaction
        prompt = f"""
You are the Producer Agent in a multi-agent AI filmmaking pipeline.

Based on the current state and user input, choose the next AGENT to run.

Available agents:
Writer, Director, Casting, AD, VideoDesign, Editor

Guidance:
- If the user asks about story, script, plot, episode, or scene → return Writer
- If the user asks about characters or visuals → return Casting
- Return ONLY one valid agent name from the list above. No explanations.

User input: {last_ai_response}
"""
        
        # Get agent recommendation from the Producer
        agent_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20
        )
        
        agent_type = agent_response.choices[0].message.content.strip()
        
        # Generate suggestions based on the selected agent type - simple action phrases
        if agent_type == "Writer":
            return [
                "Develop Plot",
                "Create Episodes",
                "Write Dialogue" 
            ]
        elif agent_type == "Casting":
            return [
                "Create Characters",
                "Define Relationships",
                "Add Character Details"
            ]
        elif agent_type == "Director":
            return [
                "Set Tone",
                "Plan Scenes",
                "Direct Actors"
            ]
        elif agent_type == "VideoDesign":
            return [
                "Create Scene Images",
                "Design Visuals",
                "Set Aesthetics"
            ]
        elif agent_type == "Editor":
            return [
                "Edit Scenes",
                "Refine Plot",
                "Review Story"
            ]
        elif agent_type == "AD":
            return [
                "Schedule Production",
                "Plan Logistics",
                "Manage Resources"
            ]
        else:
            return [
                "Create Characters",
                "Create Scene Images"
            ]
            
    except Exception as e:
        print(f"Error generating chat suggestions: {e}")
        return ["Create Characters", "Create Scene Images"]

@router.post("/chat/start")
async def start_chat(req: StartChatRequest):
    print(f"I AM HERE NOW")
    state = StoryState()
    session_id = str(uuid.uuid4())
    user_id = "user123"  # In real system, this would come from auth

    state.session_memory.append({"user_message": req.message})
    
    # Generate thinking process for the initial message
    thinking_process = await generate_thinking_process(req.message)
    
    # Create initial document
    story_projects.insert_one({
        "user_id": "anonymous",
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "story_data": {}
    })

    # Store initial messages
    initial_messages = [
        {
            "sender": "user",
            "message": req.message,
            "timestamp": datetime.utcnow(),
            "response_type": "string",  # Default to string for user messages
            "message_id": str(uuid.uuid4())  # Add unique message ID
        },
        {
            "sender": "nexora_thinking", 
            "message": req.message,  # Original user message
            "message_thinking": thinking_process,  # The thinking process text
            "timestamp": datetime.utcnow(),
            "response_type": "string",  # Thinking is always string
            "message_id": str(uuid.uuid4())  # Add unique message ID
        }
    ]
    
    chat_sessions.insert_one({
        "session_id": session_id,
        "messages": initial_messages
    })

    async def event_stream():
        # First, send the session ID so the client can capture it
        yield f"data: New session created with ID: {session_id}\n\n"
        
        # To store the complete response for database update
        full_response = []
        
        # Then stream the actual response
        async for chunk in run_producer_stream(state, session_id, req.message):
            yield chunk
            
            # Capture content for database update
            if chunk.startswith("data: ") and "ResponseType" not in chunk and "[DONE]" not in chunk:
                content = chunk[6:].strip() # Remove "data: " prefix
                if content:
                    full_response.append(content)
        
        # Save the complete response to the database
        if full_response:
            complete_text = "".join(full_response)
            
            # Determine response type based on content
            response_type = determine_response_type(complete_text)
            
            # Generate suggestions based on the conversation so far
            suggestions = await generate_chat_suggestions(
                session_id, 
                initial_messages + [{
                    "sender": "nexora",
                    "message": complete_text
                }]
            )
            
            # Extract episodes from the response
            episodes = extract_episodes_from_response(complete_text)
            
            # Create response message with all data embedded directly
            nexora_response = {
                "sender": "nexora",
                "message": complete_text,
                "timestamp": datetime.utcnow(),
                "suggestions": suggestions,
                "response_type": response_type,
                "has_episodes": len(episodes) > 0,
                "message_id": str(uuid.uuid4())  # Add unique message ID
            }
            
            # Add episodes if they exist
            if episodes:
                print(f"✅ Extracted {len(episodes)} episodes")
                nexora_response["episodes"] = episodes
            
            # Add the message to the database
            chat_sessions.update_one(
                {"session_id": session_id},
                {"$push": {"messages": nexora_response}}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/chat/send")
async def send_chat_message(req: SendChatRequest):
    session = chat_sessions.find_one({"session_id": req.session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Generate thinking process for this message
    thinking_process = await generate_thinking_process(req.message)
    
    # First store the user message
    user_message = {
        "sender": "user",
        "message": req.message,
        "timestamp": datetime.utcnow(),
        "response_type": "string",  # Default to string for user messages
        "message_id": str(uuid.uuid4())  # Add unique message ID
    }
    
    chat_sessions.update_one(
        {"session_id": req.session_id},
        {"$push": {"messages": user_message}}
    )
    
    # Then store the thinking process as a separate message
    thinking_message = {
        "sender": "nexora_thinking",
        "message": req.message,  # Original user message
        "message_thinking": thinking_process,  # The thinking process text
        "timestamp": datetime.utcnow(),
        "response_type": "string",  # Thinking is always string
        "message_id": str(uuid.uuid4())  # Add unique message ID
    }
    
    chat_sessions.update_one(
        {"session_id": req.session_id},
        {"$push": {"messages": thinking_message}}
    )
    
    # Create state object for processing
    state = StoryState()
    
    # Try to load any existing story data
    project = story_projects.find_one({"session_id": req.session_id})
    if project and project.get("story_data"):
        state = StoryState(**project.get("story_data"))
    
    async def event_stream():
        # To store the complete response for database update
        full_response = []
        
        async for chunk in run_producer_stream(state, req.session_id, req.message):
            yield chunk
            
            # Capture content for database update
            if chunk.startswith("data: ") and "ResponseType" not in chunk and "[DONE]" not in chunk:
                content = chunk[6:].strip() # Remove "data: " prefix
                if content:
                    full_response.append(content)
        
        # Save the complete response to the database
        if full_response:
            complete_text = "".join(full_response)
            
            # Determine response type based on content
            response_type = determine_response_type(complete_text)
            
            # Generate suggestions based on the conversation
            suggestions = await generate_chat_suggestions(req.session_id)
            
            # Extract episodes from the response
            episodes = extract_episodes_from_response(complete_text)
            
            # Create response message with all data embedded directly
            nexora_response = {
                "sender": "nexora",
                "message": complete_text,
                "timestamp": datetime.utcnow(),
                "suggestions": suggestions,
                "response_type": response_type,
                "has_episodes": len(episodes) > 0,
                "message_id": str(uuid.uuid4())  # Add unique message ID
            }
            
            # Add episodes if they exist
            if episodes:
                print(f"✅ Extracted {len(episodes)} episodes")
                nexora_response["episodes"] = episodes
            
            # Add the message to the database
            chat_sessions.update_one(
                {"session_id": req.session_id},
                {"$push": {"messages": nexora_response}}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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
