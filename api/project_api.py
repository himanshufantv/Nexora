from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from utils.types import StoryState
from engine.runner import run_agent
from openai import OpenAI
from typing import Optional

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
    prompt: str

class SendChatRequest(BaseModel):
    session_id: str
    prompt: Optional[str] = None

class EditMessageRequest(BaseModel):
    session_id: str
    message_id: str
    new_message: str

# Helper function to extract episodes from text response
def extract_episodes_from_response(response_text: str) -> list:
    """
    DEPRECATED: This function is maintained for backward compatibility.
    New code should directly use the structured JSON response instead.
    
    Extract episodes from the response text and return a list of episode objects
    """
    import re
    episodes = []
    
    try:
        print(f"\n🔍 PARSING: Using deprecated episode extraction function ({len(response_text)} chars)")
        
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
        
        # If no JSON found, use GPT to convert the response into structured JSON
        if not episodes:
            print(f"🔍 PARSING: No JSON found, requesting GPT to structure the data")
            
            # Use GPT to convert the unstructured text into structured JSON
            prompt = f"""
You are a data extraction specialist. Please convert the following text into structured JSON.
Extract all episode information (title, number, and summary) into a JSON object.

Input text:
{response_text[:4000]}  # Limit to first 4000 chars to avoid token limits

Return ONLY a valid JSON object in this exact format without any additional text:
```json
{{
  "series_title": "The series title or 'Untitled Series' if not clear",
  "episodes": [
    {{
      "episode_number": 1,
      "episode_title": "The title of episode 1",
      "summary": "The summary of episode 1"
    }},
    ...more episodes...
  ]
}}
```

Do not include any explanations, just the JSON.
"""
            try:
                # Call GPT to structure the data
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                gpt_response = response.choices[0].message.content.strip()
                
                # Extract JSON from the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', gpt_response)
                if json_match:
                    json_text = json_match.group(1).strip()
                else:
                    # If no code block, try to parse the entire response as JSON
                    json_text = gpt_response
                
                # Try to parse the JSON
                parsed_json = json.loads(json_text)
                
                if isinstance(parsed_json, dict) and "episodes" in parsed_json:
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
                    
                    print(f"🔍 PARSING: Successfully extracted {len(episodes)} episodes using GPT")
                    return episodes
                else:
                    print(f"🔍 PARSING: GPT response didn't contain proper episodes structure")
            except Exception as e:
                print(f"❌ ERROR: Failed to structure data with GPT: {e}")
    
    except Exception as e:
        print(f"❌ ERROR: Error extracting episodes: {e}")
    
    # Return whatever episodes we found, or empty list if none
    return episodes

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

# Generate summary of script content
async def generate_synopsis(script_content: str) -> str:
    """Generate a concise summary of the script content"""
    try:
        if not script_content or len(script_content) < 50:
            return "No content available for summary."
        
        # Use a more concise prompt for summary generation
        prompt = f"""
You are a professional script summarizer. Create a brief, engaging synopsis of this script.
Focus on the main concept, themes, and overall arc in 2-3 paragraphs.
Do not include episode breakdowns, just an overall summary. Be concise but compelling.

SCRIPT:
{script_content[:4000]}  # Limit to avoid token limits

Return ONLY the synopsis text without additional explanations, comments, or headers.
"""

        print(f"Generating synopsis from script ({len(script_content)} chars)")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        synopsis = response.choices[0].message.content.strip()
        print(f"Generated synopsis: {len(synopsis)} chars")
        return synopsis
    except Exception as e:
        print(f"Error generating synopsis: {e}")
        return "Synopsis unavailable. Please check back later."

# Extract character names from script content
async def extract_character_names(script_content: str) -> list:
    """Extract character names from the script content"""
    try:
        if not script_content or len(script_content) < 50:
            print(f"Script content too short for character extraction")
            return []
        
        print(f"Extracting characters from script ({len(script_content)} chars)")
        print(f"Script sample: {script_content[:200]}...")
        
        # First try direct pattern matching to identify characters section
        char_section_match = re.search(r'(?:Characters|Character Profiles|Character List)\s*(?:\n|:)(.*?)(?:\n\s*\n\s*[A-Z#]|\n\s*Episode|\n\s*##|\n\s*Series|\n\s*Plot)',
                                       script_content, re.DOTALL | re.IGNORECASE)
        
        if char_section_match:
            char_section = char_section_match.group(1).strip()
            print(f"Found character section in script: {len(char_section)} chars")
            print(f"Character section: {char_section}")
            
            # Extract character names from character section
            char_names = []
            char_lines = char_section.split('\n')
            
            print(f"Processing {len(char_lines)} character lines")
            for line in char_lines:
                if not line.strip():
                    continue
                
                print(f"Processing line: {line}")
        
                # Try to match different character pattern formats
                
                # Format: "Character 1: Jack - A businessman"
                char_pattern1 = re.search(r'(?:Character\s*\d+\s*:?\s*)([A-Z][a-zA-Z\s]+?)(?:\s*[-:]\s*|\s*,\s*|\s*$)', line)
            
                # Format: "Jack - A businessman"
                char_pattern2 = re.search(r'^([A-Z][a-zA-Z\s]+?)(?:\s*[-:]\s*|\s*,\s*)', line)
            
                # Format: "** Jack:**" or "**Jack -**"
                char_pattern3 = re.search(r'\*\*\s*([A-Z][a-zA-Z\s]+?)(?:\s*[-:]\*\*|\s*\*\*)', line)
                
                # Format: "-Henry:An introverted scientist" (dash prefix with no space after colon)
                char_pattern4 = re.search(r'-\s*([A-Z][a-zA-Z\s]+?)[:]\s*', line)
                
                # Format: "-Henry" (just the name with dash prefix)
                char_pattern5 = re.search(r'-\s*([A-Z][a-zA-Z\s]+?)(?:\s*$)', line)
                
                name_match = char_pattern1 or char_pattern2 or char_pattern3 or char_pattern4 or char_pattern5
                
                if name_match:
                    char_name = name_match.group(1).strip()
                    if char_name and len(char_name) > 1:  # Avoid single letters
                        print(f"Found character: {char_name}")
                        char_names.append(char_name)
            else:
                    print(f"No character match found in line: {line}")
            
            print(f"Extracted {len(char_names)} character names using pattern matching: {char_names}")
            if char_names:
                return char_names
            else:
                print(f"No characters found with pattern matching, trying backup method")
                
                # Backup method - look for dash-prefixed names
                dash_names = []
                for line in char_lines:
                    if line.strip().startswith('-'):
                        # Extract the text after dash until a colon or space
                        name_part = line.strip()[1:].strip()
                        if ':' in name_part:
                            name = name_part.split(':', 1)[0].strip()
                        else:
                            name = name_part.split(' ', 1)[0].strip()
                        
                        if name and len(name) > 1 and name[0].isupper():
                            print(f"Found character with backup method: {name}")
                            dash_names.append(name)
                
                if dash_names:
                    print(f"Extracted {len(dash_names)} character names using backup method: {dash_names}")
                    return dash_names
        else:
            print(f"Could not identify character section in script")
        
        # If pattern matching fails, use GPT to extract characters
        print(f"Using GPT to extract character names from script")
        prompt = f"""
You are a character extraction specialist. Based on this script, list ONLY the names of main characters.
Return ONLY the character names, one per line, with no numbering, explanations, or other text.
Look for character introductions, dialogue, and any character lists or descriptions.

SCRIPT:
{script_content[:4000]}  # Limit to avoid token limits

Examples of correct output format:
Jack
Rose
Bob
"""

        print(f"Sending GPT request for character extraction")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        extracted_names = response.choices[0].message.content.strip().split('\n')
        # Clean up names (remove any non-name text)
        char_names = [name.strip() for name in extracted_names if name.strip() and len(name.strip()) > 1]
        print(f"Extracted {len(char_names)} character names using GPT: {char_names}")
        
        return char_names
    
    except Exception as e:
        print(f"Error extracting character names: {e}")
        import traceback
        print(traceback.format_exc())
        return []

# Create placeholder characters if no matches in database
async def create_placeholder_characters(character_names: list) -> list:
    """Create placeholder character objects for script characters not found in database"""
    if not character_names:
        return []
    
    print(f"Creating placeholder characters for {len(character_names)} names")
    placeholder_characters = []
    
    for name in character_names:
        print(f"Creating placeholder for character: {name}")
        placeholder_characters.append({
            "script_name": name,
            "db_name": name,
            "image_url": "",  # No image available
            "description": f"Character from the script."
        })
    
    return placeholder_characters

# Match script characters with database characters
async def match_characters_with_database(script_char_names: list, db_characters: list) -> list:
    """Match character names from script with character profiles in database"""
    try:
        if not script_char_names:
            print(f"No script characters to match")
            return []
        
        if not db_characters:
            print(f"No database characters to match with - creating placeholders")
            return await create_placeholder_characters(script_char_names)
        
        print(f"Matching {len(script_char_names)} script characters with {len(db_characters)} database characters")
        matched_characters = []
        unmatched_names = []
        
        # First, direct name matching (case insensitive)
        for script_char in script_char_names:
            script_char_lower = script_char.lower()
            matched = False
            
            for db_char in db_characters:
                try:
                    db_name = db_char.get("name", "").lower()
                    
                    # Check for direct name match or if script name is contained in db name or vice versa
                    if (script_char_lower == db_name or 
                        script_char_lower in db_name or 
                        db_name in script_char_lower):
                        
                        # Add character, even if it doesn't have an image
                        image_url = db_char.get("reference_image", "")
                        matched_characters.append({
                            "script_name": script_char,
                            "db_name": db_char.get("name", "Unknown Character"),
                            "image_url": image_url,
                            "description": db_char.get("description", "No description available.")
                        })
                        matched = True
                        print(f"Matched character '{script_char}' with '{db_char.get('name')}'")
                        break
                except Exception as e:
                    print(f"Error processing database character: {e}")
                    continue
            
            if not matched:
                print(f"Could not match script character '{script_char}' with any database character")
                unmatched_names.append(script_char)
        
        # Create placeholders for unmatched characters
        if unmatched_names:
            print(f"Creating placeholders for {len(unmatched_names)} unmatched characters")
            placeholder_chars = await create_placeholder_characters(unmatched_names)
            matched_characters.extend(placeholder_chars)

        return matched_characters
    except Exception as e:
        print(f"Error in match_characters_with_database: {e}")
        return []

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
    complete_text = ""  # Initialize for use even if no prompt is provided
    
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
    
    async def event_stream():
        print(f"Starting event stream for response")
        # To store the complete response for database update
        full_response = []
        suggestions = []
        episodes = []
        script_char_names = []  # Initialize this variable
        complete_text = ""  # Initialize complete_text for cases with no prompt
        
        # Start streaming structured response - first send prompt
        initial_response = {
            "prompt": prompt,
            "left_section": chat_history[:-1] if chat_history and len(chat_history) > 1 else [],
            "tabs": [],
            "synopsis": "",
            "script": "",
            "character": [],
            "storyboard": None
        }
        
        # Send initial structure with prompt and existing chat history (minus latest)
        yield f"data: RESPONSE_JSON: {json.dumps(initial_response)}\n\n"
        
        # If there's a prompt, process it
        if prompt:
            print(f"Prompt provided, will generate response using run_producer_stream")
            
            # First generate the thinking process
            print(f"Generating thinking process")
            try:
                thinking_process = await generate_thinking_process(prompt)
                print(f"Generated thinking process: {len(thinking_process)} chars")
                
                # Create streamed left section with thinking process
                streamed_left_section = chat_history.copy()
                if not streamed_left_section:
                    streamed_left_section.append({
                        "receiver": {
                            "time": str(datetime.utcnow()),
                            "message": thinking_process
                        }
                    })
                else:
                    # Add a new receiver message
                    streamed_left_section.append({
                        "receiver": {
                            "time": str(datetime.utcnow()),
                            "message": thinking_process
                        }
                    })
                
                # Create thinking response structure
                thinking_response = {
                    "prompt": prompt,
                    "left_section": streamed_left_section,
                    "tabs": [],
                    "synopsis": "",
                    "script": "",  # Keep script empty initially
                    "character": [],
                    "storyboard": None
                }
                
                # Stream the thinking process
                yield f"data: THINKING_PROCESS: {thinking_process}\n\n"
                yield f"data: RESPONSE_JSON: {json.dumps(thinking_response)}\n\n"
                
                # Small delay for UI clarity before main response
                await asyncio.sleep(0.5)
                
                # Now prepare for main response streaming
                current_receiver_message = thinking_process  # Keep thinking process in message
                current_script = ""  # Separate variable for script content
                
                # Stream the producer response
                try:
                    async for chunk in run_producer_stream(state, req.session_id, prompt):
                        print(f"Producer chunk received: {chunk[:30]}..." if len(chunk) > 30 else f"Producer chunk: {chunk}")
                        
                        if chunk.startswith("data: ") and "ResponseType" not in chunk and "[DONE]" not in chunk and "ThinkingProcess" not in chunk and "SUGGESTIONS:" not in chunk:
                            content = chunk[6:].strip()  # Remove "data: " prefix
                            if content:
                                full_response.append(content)
                                print(f"Added content chunk to full_response ({len(content)} chars)")
                                
                                # Update both script and complete_text
                                current_script += content
                                complete_text += content
                                
                                # Keep the thinking process in message, update script
                                streaming_response = {
                                    "prompt": prompt,
                                    "left_section": streamed_left_section,
                                    "tabs": suggestions,
                                    "synopsis": "",
                                    "script": current_script,
                                    "character": [],
                                    "storyboard": None
                                }
                                
                                yield f"data: RESPONSE_JSON: {json.dumps(streaming_response)}\n\n"
                        
                        elif chunk.startswith("data: SUGGESTIONS:"):
                            suggestions_str = chunk[16:].strip()
                            try:
                                suggestions = json.loads(suggestions_str)
                                print(f"Received suggestions: {suggestions}")
                                
                                streaming_response = {
                                    "prompt": prompt,
                                    "left_section": streamed_left_section,
                                    "tabs": suggestions,
                                    "synopsis": "",
                                    "script": current_script,
                                    "character": [],
                                    "storyboard": None
                                }
                                
                                yield f"data: RESPONSE_JSON: {json.dumps(streaming_response)}\n\n"
                            except json.JSONDecodeError as e:
                                print(f"Error parsing suggestions JSON: {e}")
                except Exception as e:
                    print(f"ERROR during run_producer_stream: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            except Exception as e:
                print(f"Error generating thinking process: {e}")
                thinking_process = f"## My Approach\n\nI will help you with: '{prompt}'"
            
            # Generate suggestions if needed
            if not suggestions:
                print(f"Generating suggestions based on conversation")
                suggestions = await generate_chat_suggestions(req.session_id)
                print(f"Generated {len(suggestions)} suggestions: {suggestions}")
            
            # Determine response type based on complete text
            print(f"Determining response type")
            response_type = determine_response_type(complete_text if complete_text else thinking_process)
            print(f"Response type determined: {response_type}")
            
            # Extract episodes if present in the response
            print(f"Checking for episodes in response")
            try:
                # Check for JSON data in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', complete_text)
                if json_match:
                    print(f"Found JSON data in markdown code block")
                    json_text = json_match.group(1).strip()
                    print(f"Parsing JSON data ({len(json_text)} chars)")
                    json_data = json.loads(json_text)
                    
                    if isinstance(json_data, dict) and "episodes" in json_data:
                        print(f"Found episodes key in JSON data with {len(json_data['episodes'])} episodes")
                        raw_episodes = json_data.get("episodes", [])
                        series_title = json_data.get("series_title", "Untitled Series")
                        
                        print(f"Processing episodes from JSON data")
                        for ep in raw_episodes:
                            episode_obj = {
                                "episode_number": ep.get("episode_number", 0),
                                "series_title": series_title,
                                "title": ep.get("episode_title", f"Episode {ep.get('episode_number', 0)}"),
                                "summary": ep.get("summary", "No summary available"),
                                "full_data": ep
                            }
                            episodes.append(episode_obj)
                        print(f"Processed {len(episodes)} episodes from JSON data")
                else:
                    print(f"No JSON data found in response, checking story state for episodes")
                    # Try to get episodes from the state
                    if project and project.get("story_data") and "episodes" in project.get("story_data", {}):
                        print(f"Found episodes in story data")
                        state_episodes = project["story_data"].get("episodes", [])
                        series_title = project["story_data"].get("title", "Untitled Series")
                        
                        print(f"Processing {len(state_episodes)} episodes from story state")
                        for ep in state_episodes:
                            episode_obj = {
                                "episode_number": ep.get("episode_number", 0),
                                "series_title": series_title,
                                "title": ep.get("episode_title", f"Episode {ep.get('episode_number', 0)}"),
                                "summary": ep.get("summary", "No summary available"),
                                "full_data": ep
                            }
                            episodes.append(episode_obj)
                        print(f"Processed {len(episodes)} episodes from story state")
                    else:
                        print(f"No episodes found in story state")
            except Exception as e:
                print(f"ERROR processing episodes: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
            # Create response message
            print(f"Creating nexora response message for database")
            nexora_response = {
                "sender": "nexora",
                "message": thinking_process,  # Use thinking as message
                "timestamp": datetime.utcnow(),
                "suggestions": suggestions,
                "response_type": response_type,
                "has_episodes": len(episodes) > 0,
                "message_id": str(uuid.uuid4())
            }
            print(f"Created response with message_id: {nexora_response['message_id']}")
            
            # Add episodes if they exist
            if episodes:
                print(f"Adding {len(episodes)} episodes to response")
                nexora_response["episodes"] = episodes
            
            # Add the message to the database
            print(f"Storing nexora response in database")
            chat_sessions.update_one(
                {"session_id": req.session_id},
                {"$push": {"messages": nexora_response}}
            )
            print(f"Nexora response stored successfully")
            
            # Update final chat history with the new response
            print(f"Updating chat history with new response")
            chat_history.append({
                "receiver": {
                    "time": str(nexora_response["timestamp"]),
                    "message": nexora_response["message"]  # Use message field instead of thinking
                }
            })
            print(f"Chat history updated, now has {len(chat_history)} entries")
        else:
            # If no prompt, just get suggestions from the existing conversation
            print(f"No prompt provided, generating suggestions from existing conversation")
            suggestions = await generate_chat_suggestions(req.session_id)
            print(f"Generated {len(suggestions)} suggestions: {suggestions}")
        
        # Get character images if available
        print(f"Fetching character images")
        character_images = []
        
        try:
            # First check if we need to extract characters from the script
            if complete_text and len(complete_text) > 50:
                print(f"Extracting character names from script for matching")
                script_char_names = await extract_character_names(complete_text)
                print(f"Extracted {len(script_char_names)} character names from script")
            # If no new script content but we have existing script characters
            elif project and "story_data" in project and "script_characters" in project["story_data"]:
                script_char_names = project["story_data"].get("script_characters", [])
                print(f"Using {len(script_char_names)} existing script characters from database")
            else:
                script_char_names = []
                print(f"No script content or existing script characters available")
            
            # Always check for character profiles in the database first
            if project and project.get("story_data"):
                # First check if we have character_profiles in the story_data
                story_data = project.get("story_data", {})
                
                if "character_profiles" in story_data:
                    print(f"Found character profiles in story data")
                    characters = story_data["character_profiles"]
                    print(f"Processing {len(characters)} character profiles")
                    
                    # Always include ALL character profiles from database, even without images
                    for char in characters:
                        if "name" in char:
                            name = char.get("name", "Unknown Character")
                            image_url = char.get("reference_image", "")
                            print(f"Adding character: {name}, Has image: {bool(image_url)}")
                            
                            character_images.append({
                                "name": name,
                                "db_name": name,
                                "image_url": image_url,
                                "description": char.get("description", "No description available")
                            })
                    
                    print(f"Added {len(character_images)} character profiles directly from database")
                
                # If still no characters from profiles but we have matched_characters
                if not character_images and "matched_characters" in story_data:
                    print(f"Using previously matched characters from database")
                    db_matched_chars = story_data.get("matched_characters", [])
                    
                    for char in db_matched_chars:
                        character_images.append({
                            "name": char.get("script_name", "Unknown Character"),
                            "db_name": char.get("db_name", "Unknown Character"),
                            "image_url": char.get("image_url", ""),
                            "description": char.get("description", "")
                        })
                    print(f"Added {len(character_images)} previously matched character images from database")
                
                # Fall back to script_char_names if still no characters found
                if not character_images and script_char_names:
                    print(f"Creating placeholder characters for script characters")
                    for char_name in script_char_names:
                        character_images.append({
                            "name": char_name,
                            "db_name": char_name,
                            "image_url": "",
                            "description": "Character from the script."
                        })
                    print(f"Added {len(character_images)} character placeholders")
        except Exception as e:
            print(f"ERROR fetching character images: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        # Final check: directly access character_profiles from the database
        if project and project.get("story_data") and "character_profiles" in project.get("story_data", {}):
            print(f"FINAL CHECK: Directly accessing character_profiles from database")
            db_characters = project["story_data"]["character_profiles"]
            print(f"Found {len(db_characters)} character profiles in database")
            
            # Clear existing character images to avoid duplicates
            character_images = []
            
            for char in db_characters:
                if "name" in char:
                    name = char.get("name", "Unknown Character")
                    image_url = char.get("reference_image", "")
                    print(f"Adding character directly from DB: {name}, Has image: {bool(image_url)}")
                    
                    character_images.append({
                        "name": name,
                        "db_name": name,
                        "image_url": image_url,
                        "description": char.get("description", "No description available")
                    })
            
            print(f"Added {len(character_images)} characters directly from database")
        
        # Final debug check for character images
        print(f"DEBUG: Final character count before response: {len(character_images)}")
        for idx, char in enumerate(character_images):
            print(f"DEBUG: Character {idx+1}: {char.get('name')}, Has image: {bool(char.get('image_url'))}")
        
        # Construct the final response object
        print(f"Constructing final response object")
        
        # For synopsis content
        synopsis_content = ""
        
        # If we have new script content, generate a synopsis
        if complete_text and len(complete_text) > 50:
            print(f"Generating synopsis from script content")
            synopsis_content = await generate_synopsis(complete_text)
            print(f"Synopsis generated: {len(synopsis_content)} chars")
            
            # Save the synopsis and script character names to the database
            if project:
                print(f"Saving synopsis and script characters to database")
                update_data = {
                    "story_data.synopsis": synopsis_content
                }
                
                if script_char_names:
                    update_data["story_data.script_characters"] = script_char_names
                
                story_projects.update_one(
                    {"session_id": req.session_id},
                    {"$set": update_data}
                )
                print(f"Saved synopsis and {len(script_char_names)} script characters to database")
        # If no new content but we're retrieving existing data
        elif not prompt and project and "story_data" in project:
            story_data = project.get("story_data", {})
            if "synopsis" in story_data:
                synopsis_content = story_data.get("synopsis", "")
                print(f"Using existing synopsis from story_data ({len(synopsis_content)} chars)")
            else:
                print(f"No synopsis found in story_data")
                synopsis_content = "Synopsis will be generated after more content is available."
        
        # Final response object with everything
        response_obj = {
            "prompt": prompt,
            "left_section": chat_history,  # Full chat history including the new message
            "tabs": suggestions,
            "synopsis": synopsis_content,  # Use the generated or retrieved synopsis
            "script": complete_text,       # Keep the full script separate
            "character": character_images,
            "storyboard": None
        }
        print(f"Response object constructed with {len(chat_history)} chat history entries, {len(suggestions)} tabs, {len(character_images)} character images")
        
        # Send the final JSON response
        print(f"Serializing response object to JSON")
        response_json = json.dumps(response_obj)
        print(f"Response JSON created ({len(response_json)} chars)")
        
        # Print the full response JSON for debugging
        print("\n====== FULL RESPONSE JSON ======")
        print(response_json)
        print("===============================\n")
        
        print(f"Sending RESPONSE_JSON event")
        yield f"data: RESPONSE_JSON: {response_json}\n\n"
        print(f"Sending [DONE] event")
        yield f"data: [DONE]\n\n"
        print(f"Event stream complete")

    print(f"Returning StreamingResponse")
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
