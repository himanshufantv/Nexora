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
    import re
    episodes = []
    
    try:
        print(f"\n🔍 PARSING: Extracting episodes from response text ({len(response_text)} chars)")
        
        # First try to find and extract JSON from code blocks (new format)
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                json_text = json_match.group(1).strip()
                print(f"🔍 PARSING: Found JSON block in code fence ({len(json_text)} chars)")
                parsed_json = json.loads(json_text)
                
                if isinstance(parsed_json, dict) and "episodes" in parsed_json:
                    print(f"🔍 PARSING: Extracting episodes from JSON block")
                    episodes_data = parsed_json["episodes"]
                    series_title = parsed_json.get("series_title", "Untitled Series")
                    
                    for ep in episodes_data:
                        episode_obj = {
                            "episode_number": ep.get("episode_number", 0),
                            "series_title": series_title,
                            "title": ep.get("episode_title", f"Episode {ep.get('episode_number', 0)}"),
                            "summary": ep.get("summary", "No summary available"),
                            "full_data": ep
                        }
                        episodes.append(episode_obj)
                    
                    print(f"🔍 PARSING: Successfully extracted {len(episodes)} episodes from JSON")
                    return episodes  # Return early if we found episodes in JSON
            except json.JSONDecodeError as e:
                print(f"🔍 PARSING: Error parsing JSON from code block: {e}")
        
        # Continue with existing methods if no episodes found in JSON blocks
        if not episodes:
            # Try to find JSON-like content in the text
            json_content = None
            json_start_indices = [response_text.find('{'), response_text.find('[')]
            valid_starts = [idx for idx in json_start_indices if idx != -1]
            
            if valid_starts:
                start_idx = min(valid_starts)
                print(f"🔍 PARSING: Found potential JSON start at position {start_idx}")
                json_attempt = response_text[start_idx:]
                
                # Try to balance braces to complete truncated JSON
                open_braces = 0
                open_brackets = 0
                fixed_json = ""
                
                for char in json_attempt:
                    fixed_json += char
                    if char == '{':
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                    elif char == '[':
                        open_brackets += 1
                    elif char == ']':
                        open_brackets -= 1
                
                # Complete unbalanced braces
                while open_braces > 0:
                    fixed_json += '}'
                    open_braces -= 1
                    
                while open_brackets > 0:
                    fixed_json += ']'
                    open_brackets -= 1
                    
                # Try to fix common JSON formatting issues
                fixed_json = fixed_json.replace("'", '"')
                
                # Fix missing quotes around keys
                fixed_json = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', fixed_json)
                
                try:
                    parsed_json = json.loads(fixed_json)
                    json_content = parsed_json
                    print(f"🔍 PARSING: Successfully parsed JSON from raw content")
                    
                    # Case 1: Response is a JSON object with an "episodes" field
                    if isinstance(json_content, dict) and "episodes" in json_content:
                        episodes_data = json_content["episodes"]
                        series_title = json_content.get("series_title", "Untitled Series")
                        print(f"🔍 PARSING: Found episodes array in JSON object, count: {len(episodes_data)}")
                        
                        for i, episode in enumerate(episodes_data):
                            episode_obj = {
                                "episode_number": i + 1,
                                "series_title": series_title,
                                "title": episode.get("episode_title", f"Episode {i+1}"),
                                "summary": episode.get("summary", "No summary available"),
                                "full_data": episode
                            }
                            episodes.append(episode_obj)
                        
                        print(f"🔍 PARSING: Extracted {len(episodes)} episodes from raw JSON")
                        return episodes
                except json.JSONDecodeError:
                    print(f"🔍 PARSING: Failed to parse JSON from raw content")
        
        # Continue with regex extraction if no episodes found
        if not episodes:
            print(f"🔍 PARSING: No episodes found in JSON formats, trying regex patterns")
            # Look for episode patterns like "Episode 1: Title" or similar
            episode_patterns = [
                r'Episode\s*(\d+):\s*"([^"]+)"([^#]*?)(?=###\s*Episode\d|$)',  # Quoted title, stopping at next episode
                r'Episode\s*(\d+):\s*([^#\n]*?)(?=###\s*Episode\d|$)',  # Without quotes, stopping at next episode
                r'###\s*Episode\s*(\d+):\s*([^#\n]*?)(?=###\s*Episode\d|$)',  # Markdown style with ###
                r'Episode\s*(\d+):\s*"?([^"\n]+)"?(?:\n+([^#]+))?',  # Old pattern as fallback
                r'(\d+)\.\s+([^\n]+)(?:\n+([^#]+))?',  # 1. Title followed by summary
                r'"episode_title":\s*"([^"]+)".*?"summary":\s*"([^"]+)"'  # JSON-like format
            ]
            
            for i, pattern in enumerate(episode_patterns):
                print(f"🔍 PARSING: Trying regex pattern {i+1}")
                matches = re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                episode_count = 0
                match_texts = []
                for i, match in enumerate(matches):
                    if len(match.groups()) >= 2:
                        # Save matched text for debugging
                        match_texts.append(match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0))
                        
                        # First group is episode number or empty, second is title, third (if exists) is summary
                        try:
                            episode_num = int(match.group(1)) if match.group(1).isdigit() else i + 1
                        except (IndexError, AttributeError):
                            episode_num = i + 1
                            
                        try:
                            title = match.group(2).strip()
                        except (IndexError, AttributeError):
                            title = f"Episode {episode_num}"
                            
                        # Try to split title and summary more intelligently
                        summary = "No summary available"
                        try:
                            # Common patterns for titles vs summaries:
                            # 1. Title is first word (often capitalized with no spaces)
                            # 2. Title might be before any spaces in the text
                            # 3. Title is often just a few words
                            
                            # Approach: Find the first sentence or ending of camelCase word
                            full_text = title
                            
                            # First try to find the title by looking for CamelCase or TitleCase words
                            title_end_index = 0
                            for i, char in enumerate(full_text):
                                if i > 0 and char.isupper() and full_text[i-1].islower():
                                    title_end_index = i
                                    break
                            
                            # If no CamelCase pattern found, try to find first sentence
                            if title_end_index == 0:
                                for punct in ['.', '!', '?', ':']:
                                    pos = full_text.find(punct)
                                    if pos > 0:
                                        title_end_index = pos + 1
                                        break
                            
                            # If still not found, assume title is first 2-4 words (estimate)
                            if title_end_index == 0:
                                words = full_text.split()
                                if len(words) > 3:  # More than 3 words
                                    # Join first 2-3 words as title
                                    title_end_index = len(' '.join(words[:2]))
                                else:
                                    # Entire content is title
                                    title_end_index = len(full_text)
                            
                            # Extract title and summary
                            if title_end_index > 0 and title_end_index < len(full_text):
                                new_title = full_text[:title_end_index].strip()
                                summary = full_text[title_end_index:].strip()
                                
                                # Only update if we found a non-empty title and summary
                                if new_title and summary:
                                    title = new_title
                                    print(f"🔍 PARSING: Split title: '{title}' | Summary: '{summary[:30]}...'")
                        except Exception as e:
                            print(f"🔍 PARSING: Error splitting title/summary: {e}")
                        
                        print(f"🔍 PARSING: Found episode {episode_num}: {title}")
                        
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
                        episode_count += 1
                
                print(f"🔍 PARSING: Pattern {i+1} found {episode_count} episodes")
                if match_texts:
                    print(f"🔍 PARSING: Sample matches: {match_texts[:2]}")
                # If we found episodes with this pattern, no need to try other patterns
                if episodes:
                    break
        
        # Special handling for markdown formatted episodes if we found some episodes but likely missed others
        if episodes and len(episodes) < 10 and "### Episode" in response_text:
            print(f"🔍 PARSING: Found {len(episodes)} episodes, but there may be more in markdown format")
            # Try a specialized pattern for markdown episodes
            markdown_pattern = r'###\s*Episode\s*(\d+):\s*([^#\n]*?)(?=###\s*Episode\d|$)'
            matches = re.finditer(markdown_pattern, response_text, re.IGNORECASE | re.MULTILINE)
            
            additional_episodes = 0
            for match in matches:
                if len(match.groups()) >= 2:
                    try:
                        episode_num = int(match.group(1)) if match.group(1).isdigit() else 0
                    except (IndexError, AttributeError):
                        continue
                        
                    # Skip if we already have this episode number
                    if any(ep["episode_number"] == episode_num for ep in episodes):
                        continue
                        
                    try:
                        title = match.group(2).strip()
                    except (IndexError, AttributeError):
                        title = f"Episode {episode_num}"
                    
                    # Try to split title and summary more intelligently
                    summary = "No summary available"
                    try:
                        # Common patterns for titles vs summaries:
                        # 1. Title is first word (often capitalized with no spaces)
                        # 2. Title might be before any spaces in the text
                        # 3. Title is often just a few words
                        
                        # Approach: Find the first sentence or ending of camelCase word
                        full_text = title
                        
                        # First try to find the title by looking for CamelCase or TitleCase words
                        title_end_index = 0
                        for i, char in enumerate(full_text):
                            if i > 0 and char.isupper() and full_text[i-1].islower():
                                title_end_index = i
                                break
                        
                        # If no CamelCase pattern found, try to find first sentence
                        if title_end_index == 0:
                            for punct in ['.', '!', '?', ':']:
                                pos = full_text.find(punct)
                                if pos > 0:
                                    title_end_index = pos + 1
                                    break
                        
                        # If still not found, assume title is first 2-4 words (estimate)
                        if title_end_index == 0:
                            words = full_text.split()
                            if len(words) > 3:  # More than 3 words
                                # Join first 2-3 words as title
                                title_end_index = len(' '.join(words[:2]))
                            else:
                                # Entire content is title
                                title_end_index = len(full_text)
                        
                        # Extract title and summary
                        if title_end_index > 0 and title_end_index < len(full_text):
                            new_title = full_text[:title_end_index].strip()
                            summary = full_text[title_end_index:].strip()
                            
                            # Only update if we found a non-empty title and summary
                            if new_title and summary:
                                title = new_title
                                print(f"🔍 PARSING: Split title: '{title}' | Summary: '{summary[:30]}...'")
                    except Exception as e:
                        print(f"🔍 PARSING: Error splitting title/summary: {e}")
                    
                    print(f"🔍 PARSING: Found additional episode {episode_num}: {title}")
                    
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
                    additional_episodes += 1
            
            if additional_episodes > 0:
                print(f"🔍 PARSING: Found {additional_episodes} additional episodes using markdown pattern")
                
        # Sort episodes by episode number
        if episodes:
            episodes.sort(key=lambda x: x["episode_number"])
            print(f"🔍 PARSING: Final episode count: {len(episodes)}")
            
        print(f"🔍 PARSING: Extraction complete, found {len(episodes)} episodes")
    
    except Exception as e:
        print(f"❌ ERROR: Error extracting episodes: {e}")
    
    return episodes

# Helper function to determine response type
def determine_response_type(content: str) -> str:
    """
    Determine the response type based on the content
    
    Returns one of: "string", "image", "video", "audio", "file", "mixed"
    """
    import re
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

Write a clear explanation of how you would approach this task, including the steps you would take.
FORMAT YOUR RESPONSE IN VALID MARKDOWN following these guidelines:
1. Use proper heading structure (# for main headings)
2. Use PROPERLY FORMATTED bullet points with complete sentences
3. Ensure all markdown syntax is valid (no incomplete bullet points, etc.)
4. Properly format any lists with complete bullets
5. Keep your explanation focused and structured

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
        print(f"\n🔵 API RESPONSE (/chat/start): Sending session ID: {session_id}")
        yield f"data: New session created with ID: {session_id}\n\n"
        
        # Stream the thinking process
        print(f"🔵 API RESPONSE (/chat/start): Starting thinking process stream")
        yield f"data: ThinkingProcess: begin\n\n"
        
        # Split the thinking process into chunks and stream them
        # This prevents very large thinking processes from causing issues
        chunk_size = 500  # Characters per chunk
        for i in range(0, len(thinking_process), chunk_size):
            chunk = thinking_process[i:i+chunk_size]
            print(f"🔵 API RESPONSE (/chat/start): Thinking chunk: {chunk[:50]}..." if len(chunk) > 50 else f"🔵 API RESPONSE (/chat/start): Thinking chunk: {chunk}")
            yield f"data: {chunk}\n\n"
            # Small delay to ensure chunks are processed in order
            await asyncio.sleep(0.01)
            
        print(f"🔵 API RESPONSE (/chat/start): Ending thinking process stream")
        yield f"data: ThinkingProcess: end\n\n"
        
        # To store the complete response for database update
        full_response = []
        
        # Then stream the actual response
        async for chunk in run_producer_stream(state, session_id, req.message):
            print(f"🔵 API RESPONSE (/chat/start): Producer stream chunk: {chunk[:50]}..." if len(chunk) > 50 else f"🔵 API RESPONSE (/chat/start): Producer stream chunk: {chunk}")
            yield chunk
            
            # Capture content for database update
            if chunk.startswith("data: ") and "ResponseType" not in chunk and "[DONE]" not in chunk and "ThinkingProcess" not in chunk and "SUGGESTIONS:" not in chunk:
                content = chunk[6:].strip() # Remove "data: " prefix
                if content:
                    full_response.append(content)
        
        # Save the complete response to the database
        if full_response:
            complete_text = "".join(full_response)
            
            print(f"\n📝 COMPLETE RESPONSE (/chat/start): First 200 chars:\n{complete_text[:200]}..." if len(complete_text) > 200 else f"\n📝 COMPLETE RESPONSE (/chat/start):\n{complete_text}")
            
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
            
            # Send suggestions as a special message before [DONE]
            if suggestions:
                suggestions_json = json.dumps(suggestions)
                print(f"🔵 API RESPONSE (/chat/start): Sending suggestions: {suggestions}")
                yield f"data: SUGGESTIONS: {suggestions_json}\n\n"
                await asyncio.sleep(0.05)
            
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
                print(f"🔵 API RESPONSE (/chat/start): Extracted episodes: {episodes}")
            else:
                print(f"🔵 API RESPONSE (/chat/start): No episodes extracted")
            
            # Add the message to the database
            chat_sessions.update_one(
                {"session_id": session_id},
                {"$push": {"messages": nexora_response}}
            )
            
            print(f"🔵 API RESPONSE (/chat/start): Complete response saved to database ({len(complete_text)} chars, type: {response_type})")

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
        # Stream the thinking process
        print(f"\n🔵 API RESPONSE (/chat/send): Starting thinking process stream")
        yield f"data: ThinkingProcess: begin\n\n"
        
        # Split the thinking process into chunks and stream them
        chunk_size = 500  # Characters per chunk
        for i in range(0, len(thinking_process), chunk_size):
            chunk = thinking_process[i:i+chunk_size]
            print(f"🔵 API RESPONSE (/chat/send): Thinking chunk: {chunk[:50]}..." if len(chunk) > 50 else f"🔵 API RESPONSE (/chat/send): Thinking chunk: {chunk}")
            yield f"data: {chunk}\n\n"
            # Small delay to ensure chunks are processed in order
            await asyncio.sleep(0.01)
            
        print(f"🔵 API RESPONSE (/chat/send): Ending thinking process stream")
        yield f"data: ThinkingProcess: end\n\n"
        
        # To store the complete response for database update
        full_response = []
        
        async for chunk in run_producer_stream(state, req.session_id, req.message):
            print(f"🔵 API RESPONSE (/chat/send): Producer stream chunk: {chunk[:50]}..." if len(chunk) > 50 else f"🔵 API RESPONSE (/chat/send): Producer stream chunk: {chunk}")
            yield chunk
            
            # Capture content for database update
            if chunk.startswith("data: ") and "ResponseType" not in chunk and "[DONE]" not in chunk and "ThinkingProcess" not in chunk and "SUGGESTIONS:" not in chunk:
                content = chunk[6:].strip() # Remove "data: " prefix
                if content:
                    full_response.append(content)
        
        # Save the complete response to the database
        if full_response:
            complete_text = "".join(full_response)
            
            print(f"\n📝 COMPLETE RESPONSE (/chat/send): First 200 chars:\n{complete_text[:200]}..." if len(complete_text) > 200 else f"\n📝 COMPLETE RESPONSE (/chat/send):\n{complete_text}")
            
            # Determine response type based on content
            response_type = determine_response_type(complete_text)
            
            # Generate suggestions based on the conversation
            suggestions = await generate_chat_suggestions(req.session_id)
            
            # Send suggestions as a special message before [DONE]
            if suggestions:
                suggestions_json = json.dumps(suggestions)
                print(f"🔵 API RESPONSE (/chat/send): Sending suggestions: {suggestions}")
                yield f"data: SUGGESTIONS: {suggestions_json}\n\n"
                await asyncio.sleep(0.05)
            
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
                print(f"🔵 API RESPONSE (/chat/send): Extracted episodes: {episodes}")
            else:
                print(f"🔵 API RESPONSE (/chat/send): No episodes extracted")
            
            # Add the message to the database
            chat_sessions.update_one(
                {"session_id": req.session_id},
                {"$push": {"messages": nexora_response}}
            )
            
            print(f"🔵 API RESPONSE (/chat/send): Complete response saved to database ({len(complete_text)} chars, type: {response_type})")

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
