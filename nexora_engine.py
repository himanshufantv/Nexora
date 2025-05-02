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
from utils.functional_logger import log_flow, log_api_call, log_db_operation, log_error, log_entry_exit

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
    log_flow(f"Starting producer stream for session {session_id}")
    project = projects.find_one({"session_id": session_id})
    if not project:
        log_error(f"Session not found: {session_id}")
        yield "data: Error: session not found.\n\n"
        return        
    log_flow(f"Found project for session {session_id}")
    
    if project.get("story_data"):
        story_data = project.get("story_data")
        log_flow(f"Loaded story data from project")
        state = StoryState(**story_data)
        state.session_id = session_id
       
    try:
        log_flow(f"Determining agent to run based on user message")
        agent_to_run = producer_agent(state, user_message)
        log_flow(f"Producer selected agent: {agent_to_run}")

        if not agent_to_run or agent_to_run not in agent_map:
            log_flow(f"Invalid agent {agent_to_run}, defaulting to Writer")
            agent_to_run = "Writer"

        # First send response type information
        yield f"data: ResponseType: {agent_to_run}\n\n"
        await asyncio.sleep(0.1)
        
        # If user is asking for scene images but was routed to VideoDesign, redirect to AD
        if "scene" in user_message.lower() and "image" in user_message.lower() and agent_to_run == "VideoDesign":
            log_flow("Redirecting from VideoDesign to AD for scene images")
            agent_to_run = "AD"
            yield f"data: ResponseType: {agent_to_run}\n\n"
            await asyncio.sleep(0.1)
            
        # Special handling for Casting agent to get images
        if agent_to_run == "Casting":
            log_flow("Using Casting agent for image generation")
            
            # Import the character utilities
            try:
                from utils.character_utils import save_characters_to_db
                log_flow("Successfully imported character_utils")
            except ImportError:
                log_error("Could not import character_utils")
            
            # Use the actual Casting agent implementation
            log_flow("Calling casting agent")
            updated_state = casting_agent(state)
            
            # Check if we got character profiles with images
            if updated_state.character_profiles:
                log_flow(f"Casting agent returned {len(updated_state.character_profiles)} character profiles with images")
                
                # Start with an introduction
                yield "data: Character visualizations have been generated:\n\n"
                await asyncio.sleep(0.1)
                
                # For each character, send the image URL and description
                for profile in updated_state.character_profiles:
                    char_name = profile.get("name", "Character")
                    image_url = profile.get("reference_image", "")
                    description = profile.get("description", "")
                    
                    log_flow(f"Sending character image for {char_name}")
                    # Send the image URL in a special format the client can handle
                    if image_url:
                        yield f"data: IMAGE_URL:{char_name}:{image_url}\n\n"
                        yield f"data: {char_name} has been visualized with the appearance shown above.\n\n"
                    else:
                        log_flow(f"No image URL for character {char_name}", level="warning")
                        yield f"data: Could not generate an image for {char_name}.\n\n"
                    
                    await asyncio.sleep(0.1)
                
                # Additional: Explicitly save character profiles to database
                try:
                    if 'save_characters_to_db' in locals():
                        log_flow("Saving character profiles to database")
                        log_db_operation("save", "characters", {"session_id": session_id})
                        character_save_result = save_characters_to_db(
                            session_id, 
                            updated_state.character_profiles, 
                            updated_state.character_map
                        )
                        log_flow(f"Character save result: {character_save_result}")
                except Exception as char_save_err:
                    log_error(f"Error saving characters", char_save_err)
                
                # Update the state in the database (original method)
                try:
                    log_db_operation("update", "projects", {"session_id": session_id})
                    projects.update_one(
                        {"session_id": session_id},
                        {"$set": {
                            "story_data": updated_state.dict(),
                            "updated_at": datetime.utcnow()
                        }}
                    )
                    log_flow("Updated state in database with character images and details")
                except Exception as db_err:
                    log_error("Database update error", db_err)
            else:
                # Fallback to description generation if no images are found
                log_flow("No character profiles with images generated, falling back to text descriptions", level="warning")
                prompt = f"""
You are helping visualize characters for a story.

Based on these character descriptions, provide detailed visual descriptions:
{chr(10).join(state.characters)}

For each character, describe their physical appearance, age, clothing style, and distinguishing features.
"""
                log_api_call("OpenAI Chat Completion")
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    stream=True
                )
                
                buffer = ""
                stop_streaming = False
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        buffer += content
                        
                        # If we already hit the JSON section, don't stream
                        if stop_streaming:
                            continue
                            
                        # Check if we're entering the JSON section
                        if "```json" in buffer:
                            # Send whatever is before ```json
                            parts = buffer.split("```json", 1)
                            if parts[0].strip():
                                yield f"data: {parts[0]}\n\n"
                                await asyncio.sleep(0.01)
                            
                            # Clear buffer and set flag to stop streaming
                            buffer = ""
                            stop_streaming = True
                            continue
                        
                        import re
                        if re.search(r'[.!?,;:\s]', buffer) or len(buffer) > 15:
                            if buffer.strip():
                                yield f"data: {buffer}\n\n"
                                await asyncio.sleep(0.01)
                            buffer = ""
                
                if buffer.strip() and not stop_streaming:
                    yield f"data: {buffer}\n\n"
        
        # Special handling for AD agent to generate scene images
        elif agent_to_run == "AD":
            log_flow("Using AD agent for scene image generation")
            
            # Check if we have episode scripts or scene scripts
            if not state.episode_scripts and not state.scene_scripts:
                # If no existing scenes, create quick scene prompts from episodes
                if state.episodes:
                    # Create scene images based on episode summaries
                    log_flow("Generating scene visualizations from episode summaries")
                    yield "data: Generating scene visualizations from episode summaries:\n\n"
                    
                    # Import image generation function
                    from utils.replicate_image_gen import generate_flux_image
                    import replicate
                    log_flow(f"REPLICATE_API_TOKEN is {'SET' if os.getenv('REPLICATE_API_TOKEN') else 'NOT SET'}")
                    
                    # Generate scene prompts based on episode summaries
                    for i, episode in enumerate(state.episodes[:3]):  # Limit to first 3 episodes
                        episode_title = episode.get("episode_title", f"Episode {i+1}")
                        summary = episode.get("summary", "")
                        
                        log_flow(f"Generating scene image for episode: {episode_title}")
                        # Generate image prompt for this episode
                        prompt = f"""
You are an expert visual artist creating scene descriptions for image generation.
Create a detailed visual description for this scene that would be good for generating an image.
Focus on setting, lighting, characters, mood, and visual elements.

Episode: {episode_title}
Summary: {summary}

Keep your description under 100 words and make it visually rich.
"""
                        # Get a scene description
                        response = openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=150
                        )
                        
                        scene_description = response.choices[0].message.content.strip()
                        scene_name = f"Scene from {episode_title}"
                        
                        yield f"data: {scene_name}: {scene_description}\n\n"
                        await asyncio.sleep(0.1)
                        
                        # Now generate a real image using Replicate API
                        try:
                            # Generate the actual image
                            print(f"🔍 Calling generate_flux_image with scene description: {scene_description[:50]}...")
                            image_url, _ = generate_flux_image(scene_description)
                            
                            if not image_url:
                                print("⚠️ Initial image generation failed, trying direct call...")
                                # Direct call as fallback
                                try:
                                    output = replicate.run(
                                        "black-forest-labs/flux-1.1-pro",
                                        input={
                                            "prompt": scene_description,
                                            "aspect_ratio": "9:16",
                                            "output_format": "jpg",
                                            "prompt_strength": 0.8
                                        }
                                    )
                                    
                                    if isinstance(output, list) and len(output) > 0:
                                        image_url = str(output[0])
                                    elif isinstance(output, str):
                                        image_url = output
                                    else:
                                        raise Exception("Unexpected output format from Replicate API")
                                except Exception as e:
                                    print(f"❌ Direct Replicate call failed: {e}")
                                    # Final fallback to placeholder
                                    image_url = f"https://placehold.co/600x400/png?text={episode_title.replace(' ', '+')}"
                            
                            # Send the image URL to the client
                            yield f"data: IMAGE_URL:{scene_name}:{image_url}\n\n"
                            await asyncio.sleep(0.1)
                            
                            # Update state with the new image
                            if not hasattr(state, 'ad_images'):
                                state.ad_images = {}
                            state.ad_images[f"episode{i+1}_main"] = image_url
                            
                        except Exception as e:
                            print(f"❌ Error generating image: {e}")
                            # Fallback to placeholder only if real generation fails
                            placeholder_url = f"https://placehold.co/600x400/png?text={episode_title.replace(' ', '+')}"
                            yield f"data: IMAGE_URL:{scene_name}:{placeholder_url}\n\n"
                            await asyncio.sleep(0.1)
                    
                    # Save the updated state
                    try:
                        projects.update_one(
                            {"session_id": session_id},
                            {"$set": {"story_data": state.dict(), "updated_at": datetime.utcnow()}}
                        )
                        print(f"✅ Updated state in database with generated scene images")
                    except Exception as db_err:
                        print(f"❌ Failed to update database: {str(db_err)}")
                        
                else:
                    # No episodes or scenes to generate images from
                    yield "data: No episodes or scenes found to generate images. Please create a story first.\n\n"
            else:
                # If we have scene scripts, use those for more detailed scene images
                # Run the actual AD agent
                try:
                    updated_state = ad_agent(state)
                    
                    # Check if we got image URLs
                    if updated_state.ad_images and len(updated_state.ad_images) > 0:
                        yield "data: Scene visualizations have been generated:\n\n"
                        await asyncio.sleep(0.1)
                        
                        # For each scene, send the image URL
                        for scene_key, image_url in updated_state.ad_images.items():
                            if image_url:
                                yield f"data: IMAGE_URL:{scene_key}:{image_url}\n\n"
                                yield f"data: Scene {scene_key} has been visualized.\n\n"
                                await asyncio.sleep(0.1)
                        
                        # Update state
                        projects.update_one(
                            {"session_id": session_id},
                            {"$set": {"story_data": updated_state.dict(), "updated_at": datetime.utcnow()}}
                        )
                        print(f"✅ Updated state in database with scene images")
                    else:
                        # No images generated, fallback to generating placeholders
                        yield "data: Generating scene visualizations from episode summaries:\n\n"
                        
                        # Generate placeholder images for episodes
                        if state.episodes:
                            for i, episode in enumerate(state.episodes[:5]):  # Limit to first 5 episodes
                                episode_title = episode.get("episode_title", f"Episode {i+1}")
                                summary = episode.get("summary", "")
                                
                                # Generate placeholder image and description
                                scene_name = f"Scene from {episode_title}"
                                scene_description = f"Visual representation of: {summary[:100]}..."
                                
                                yield f"data: {scene_name}: {scene_description}\n\n"
                                await asyncio.sleep(0.1)
                                
                                # Create a placeholder image URL with episode title
                                placeholder_url = f"https://placehold.co/600x400/png?text={episode_title.replace(' ', '+')}"
                                yield f"data: IMAGE_URL:{scene_name}:{placeholder_url}\n\n"
                                await asyncio.sleep(0.1)
                        else:
                            yield "data: No episodes found to generate scene visualizations.\n\n"
                except Exception as e:
                    print(f"❌ Error running AD agent: {e}")
                    
                    # Fallback to generating actual images even if AD agent fails
                    yield "data: Generating fallback scene visualizations:\n\n"
                    
                    # Import image generation function if not already imported
                    try:
                        from utils.replicate_image_gen import generate_flux_image
                        import replicate
                    except ImportError:
                        pass
                    
                    # Generate real images for episodes
                    if state.episodes:
                        for i, episode in enumerate(state.episodes[:3]):  # Limit to first 3 episodes
                            episode_title = episode.get("episode_title", f"Episode {i+1}")
                            summary = episode.get("summary", "")
                            
                            # Generate a good description for image generation
                            prompt_to_gpt = f"""
You are an expert visual artist creating scene descriptions for image generation.
Create a detailed visual description for this scene that would be good for generating an image.
Focus on setting, lighting, characters, mood, and visual elements.

Episode: {episode_title}
Summary: {summary}

Keep your description under 100 words and make it visually rich.
"""
                            response = openai_client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": prompt_to_gpt}],
                                temperature=0.7,
                                max_tokens=150
                            )
                            
                            scene_description = response.choices[0].message.content.strip()
                            scene_name = f"Scene from {episode_title}"
                            
                            yield f"data: {scene_name}: {scene_description}\n\n"
                            await asyncio.sleep(0.1)
                            
                            # Try to generate a real image
                            try:
                                print(f"🔍 Calling generate_flux_image in fallback for: {scene_description[:50]}...")
                                image_url, _ = generate_flux_image(scene_description)
                                
                                if not image_url:
                                    # Try direct API call as second fallback
                                    try:
                                        output = replicate.run(
                                            "black-forest-labs/flux-1.1-pro",
                                            input={
                                                "prompt": scene_description,
                                                "aspect_ratio": "9:16",
                                                "output_format": "jpg"
                                            }
                                        )
                                        
                                        if isinstance(output, list) and len(output) > 0:
                                            image_url = str(output[0])
                                        elif isinstance(output, str):
                                            image_url = output
                                        else:
                                            raise Exception("Unexpected output format")
                                    except Exception as api_err:
                                        print(f"❌ Direct API call failed: {api_err}")
                                        # Final fallback to placeholder
                                        image_url = f"https://placehold.co/600x400/png?text={episode_title.replace(' ', '+')}"
                            except Exception as img_err:
                                print(f"❌ Image generation failed in fallback: {img_err}")
                                # Use placeholder as last resort
                                image_url = f"https://placehold.co/600x400/png?text={episode_title.replace(' ', '+')}"
                                
                            # Send the image URL
                            yield f"data: IMAGE_URL:{scene_name}:{image_url}\n\n"
                            await asyncio.sleep(0.1)
                            
                            # Save this image to state
                            if not hasattr(state, 'ad_images'):
                                state.ad_images = {}
                            state.ad_images[f"fallback_episode{i+1}"] = image_url
                            
                        # Save the updated state
                        try:
                            projects.update_one(
                                {"session_id": session_id},
                                {"$set": {"story_data": state.dict(), "updated_at": datetime.utcnow()}}
                            )
                            print(f"✅ Updated state in database with fallback scene images")
                        except Exception as db_err:
                            print(f"❌ Failed to update database in fallback: {str(db_err)}")
                    else:
                        yield f"data: Error generating scene images: {str(e)}\n\n"
        else:
            # Extract the profile if it's a Writer
            profile_key = "english_romantic"
            if "::" in agent_to_run:
                agent_parts = agent_to_run.split("::")
                if len(agent_parts) >= 2:
                    agent_to_run = agent_parts[0]
                    profile_key = agent_parts[1]

            print(f"Using agent {agent_to_run} with profile {profile_key}")
            
            # Generate the prompt based on the agent and state
            prompt = generate_agent_prompt(state, agent_to_run, user_message, profile_key)
            print(f"Generated prompt: {prompt[:100]}...")  # Print first 100 chars of prompt for debugging
            
            # Direct OpenAI streaming call
            try:
                import re
                # Use legacy completions format for streaming simplicity
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    stream=True,
                )
                
                # Buffer to accumulate chunks for more natural word grouping
                buffer = ""
                full_response = ""
                stop_streaming = False
                
                # Stream the response chunks
                for chunk in response:
                    # Extract the content from the chunk
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        buffer += content
                        full_response += content
                        
                        # If we already hit the JSON section, just accumulate for DB but don't stream
                        if stop_streaming:
                            continue
                        
                        # Check if we're entering the JSON section (don't stream this to frontend)
                        if "```json" in buffer:
                            # Send whatever is before ```json
                            parts = buffer.split("```json", 1)
                            if parts[0].strip():
                                yield f"data: {parts[0]}\n\n"
                                await asyncio.sleep(0.01)
                            
                            # Clear buffer and set flag to stop streaming to frontend
                            buffer = ""
                            stop_streaming = True
                            continue
                        
                        # Send buffer when we have a complete word/phrase or accumulated enough text
                        if (re.search(r'[.!?,;:\s]', buffer) or 
                            len(buffer) > 15 or 
                            re.search(r'[\n\r]', buffer) or
                            re.search(r'#+\s', buffer)):
                            
                            if buffer.strip():
                                # Add explicit newlines for Markdown headers and list items
                                if re.search(r'^#+\s', buffer.strip()) or re.search(r'^-\s', buffer.strip()):
                                    # Ensure headers start on a new line
                                    yield f"data: \n{buffer}\n\n"
                                else:
                                    yield f"data: {buffer}\n\n"
                                # Small delay to make it feel natural
                                await asyncio.sleep(0.01)
                            buffer = ""
                
                # Send any remaining content in buffer (but not if we're in JSON section)
                if buffer.strip() and not stop_streaming:
                    yield f"data: {buffer}\n\n"
                
                # Update the state with the generated content (simplified for now)
                if full_response:
                    print(f"Full response generated: {len(full_response)} characters")
                    
                    # Look for JSON data that might be enclosed in ```json ``` blocks
                    json_data = None
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', full_response)
                    
                    if json_match:
                        try:
                            json_text = json_match.group(1).strip()
                            print(f"Found JSON block ({len(json_text)} chars)")
                            json_data = json.loads(json_text)
                            print(f"Successfully parsed JSON from code block")
                            
                            # Process JSON data if found
                            if isinstance(json_data, dict):
                                if "series_title" in json_data:
                                    state.title = json_data.get("series_title", state.title)
                                if "logline" in json_data:
                                    state.logline = json_data.get("logline", state.logline)
                                if "characters" in json_data:
                                    # Convert character objects to strings for compatibility
                                    state.characters = [
                                        f"{char.get('name', 'Character')}: {char.get('description', '')}"
                                        for char in json_data.get("characters", [])
                                    ]
                                if "episodes" in json_data:
                                    # Use properly structured episode data
                                    state.episodes = [
                                        {
                                            "episode_title": ep.get("episode_title", f"Episode {ep.get('episode_number', i+1)}"),
                                            "summary": ep.get("summary", "No summary available")
                                        }
                                        for i, ep in enumerate(json_data.get("episodes", []))
                                    ]
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON from code block: {e}")
                            json_data = None
                    
                    # Only try to parse as JSON if it explicitly looks like valid JSON and we didn't already find a JSON block
                    if not json_data and ((full_response.strip().startswith('{') and full_response.strip().endswith('}')) or (full_response.strip().startswith('[') and full_response.strip().endswith(']'))):
                        try:
                            json_data = json.loads(full_response)
                            # Only process JSON if it has the expected structure
                            if isinstance(json_data, dict) and ("series_title" in json_data or "episodes" in json_data):
                                # Looks like a series synopsis
                                state.title = json_data.get("series_title", state.title)
                                state.logline = json_data.get("logline", state.logline)
                                if "characters" in json_data:
                                    state.characters = json_data.get("characters", [])
                                if "episodes" in json_data:
                                    state.episodes = json_data.get("episodes", [])
                        except json.JSONDecodeError:
                            # Not valid JSON, just use as raw text
                            json_data = None
                    
                    # If using Markdown output and we couldn't parse as JSON, try to extract key information from the Markdown
                    if not json_data and agent_to_run == "Writer" and "# " in full_response:
                        try:
                            # Try to extract series title (first h1 heading)
                            title_match = re.search(r'# (.*)', full_response)
                            if title_match:
                                state.title = title_match.group(1).strip()
                            
                            # Try to extract logline (text after title, before next heading)
                            if title_match:
                                title_pos = title_match.start()
                                next_heading = full_response.find('#', title_pos + 1)
                                if next_heading > -1:
                                    logline = full_response[title_pos + len(title_match.group(0)):next_heading].strip()
                                    if logline:
                                        state.logline = logline
                            
                            # Try to extract characters
                            characters_section = re.search(r'## Characters(.*?)(?=^##|\Z)', full_response, re.MULTILINE | re.DOTALL)
                            if characters_section:
                                character_lines = re.findall(r'- (.*)', characters_section.group(1))
                                if character_lines:
                                    state.characters = character_lines
                            
                            # Try to extract episodes
                            episodes = []
                            episode_sections = re.finditer(r'### Episode \d+: (.*?)\n(.*?)(?=^###|\Z)', full_response, re.MULTILINE | re.DOTALL)
                            for match in episode_sections:
                                episode_title = match.group(1).strip()
                                episode_summary = match.group(2).strip()
                                episodes.append({
                                    "episode_title": episode_title,
                                    "summary": episode_summary
                                })
                            
                            if episodes:
                                state.episodes = episodes
                                
                        except Exception as parse_err:
                            print(f"Error parsing Markdown content: {parse_err}")
                    
                    # Generate suggestions based on the conversation
                    try:
                        # Import the generate_chat_suggestions from project_api if it exists there
                        from api.project_api import generate_chat_suggestions as get_suggestions
                        suggestions = await get_suggestions(session_id)
                    except (ImportError, NameError):
                        # Fallback to hard-coded suggestions if function not available
                        print("Couldn't import generate_chat_suggestions, using fallback suggestions")
                        if agent_to_run == "Writer":
                            suggestions = ["Develop the story further", "Create character backgrounds", "Add conflict"]
                        elif agent_to_run == "Casting":
                            suggestions = ["Cast main character", "Design character visuals", "Create character relationships"]
                        elif agent_to_run == "Director":
                            suggestions = ["Design a scene", "Set the visual tone", "Plan camera shots"]
                        else:
                            suggestions = ["Tell me more", "Create a new episode", "Add details"]
                    
                    # Send suggestions as a special message
                    if suggestions:
                        suggestions_json = json.dumps(suggestions)
                        yield f"data: SUGGESTIONS: {suggestions_json}\n\n"
                        await asyncio.sleep(0.05)
                    
                    # Save the updated state to the database
                    try:
                        projects.update_one(
                            {"session_id": session_id},
                            {"$set": {"story_data": state.dict(), "updated_at": datetime.utcnow()}}
                        )
                        print(f"✅ State updated in database for session {session_id}")
                    except Exception as db_err:
                        print(f"❌ Failed to update database: {str(db_err)}")
                
            except Exception as openai_err:
                print(f"❌ OpenAI streaming error: {str(openai_err)}")
                yield f"data: Error with OpenAI streaming: {str(openai_err)}\n\n"

    except Exception as e:
        error_message = str(e)
        print(f"Error in run_producer_stream: {error_message}")
        yield f"data: Error: {error_message}\n\n"

    # Signal completion
    yield "data: [DONE]\n\n"

def generate_agent_prompt(state: StoryState, agent_type: str, user_message: str, profile_key: str = "english_romantic") -> str:
    log_flow(f"Generating prompt for agent type: {agent_type}")
    
    if agent_type == "Writer":
        # Load writer profile properties
        try:
            from agents.writer import load_writer_profile
            profile = load_writer_profile(profile_key)
            system_prompt = profile["system_prompt"]
            language = profile["language"]
            tone = profile["tone"]
            style = profile["style_note"]
            genre = profile["genre"]
        except Exception as e:
            print(f"Error loading writer profile: {e}")
            system_prompt = "You are a creative AI writer."
            language = "English"
            tone = "Dramatic"
            style = "Modern"
            genre = "Drama"
            
        # Check if we need to generate series, episode, or scene
        if "scene" in user_message.lower():
            # Logic for scene generation
            nums = [int(s) for s in user_message.split() if s.isdigit()]
            
            if len(nums) == 1:
                episode_number = 1
                scene_number = nums[0]
            elif len(nums) >= 2:
                episode_number, scene_number = nums[0], nums[1]
            else:
                episode_number, scene_number = 1, 1
                
            scene_list = state.episode_scripts.get(episode_number, [])
            if scene_number <= len(scene_list) and scene_number >= 1:
                scene_text = scene_list[scene_number - 1]
                
                return f"""
{system_prompt}

Break this scene into cinematic shots:
only 1-2 characters per shot
Scene: {scene_text}

FORMAT YOUR RESPONSE IN MARKDOWN, not JSON.
Use headings, bullets, and other formatting to make your response clear and readable.
"""
            else:
                return f"Error: Scene {scene_number} does not exist in Episode {episode_number}."
            
        elif "episode" in user_message.lower():
            # Logic for episode generation
            episode_number = 1
            for s in user_message.split():
                if s.isdigit():
                    episode_number = int(s)
                    break
                    
            if 0 < episode_number <= len(state.episodes):
                episode = state.episodes[episode_number - 1]
                title = episode["episode_title"]
                summary = episode["summary"]

                return f"""
{system_prompt}

Break this episode into 6-8 scenes.
only 1-2 characters per scene
Title: {title}
Summary: {summary}

FORMAT YOUR RESPONSE IN MARKDOWN, not JSON.
Use headings for each scene (e.g., "## Scene 1") and include descriptions of setting, characters present, and key actions.
"""
            else:
                return f"Error: Episode {episode_number} does not exist."
                
        else:
            # Series synopsis generation
            return f"""
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

PROVIDE YOUR RESPONSE IN TWO FORMATS:

FORMAT 1: PROPER, CLEAN MARKDOWN using these guidelines:
1. Use proper line breaks between sections
2. Start with a clear title as a level 1 heading (# Title)
3. Put one blank line between the title and the logline
4. Use level 2 headings (## Heading) for main sections
5. Use bullet points with proper formatting (- Item)
6. Use level 3 headings (### Heading) for episode titles
7. Each section must be on its own line with proper spacing

Structure your markdown response precisely as:

# [Series Title]

[Series Logline]

## Characters

- Character 1: Brief description
- Character 2: Brief description

## Episodes

### Episode 1: [Title]

[Episode summary]

### Episode 2: [Title]

[Episode summary]

FORMAT 2: STRUCTURED JSON:
After your markdown response, include a clearly formatted JSON object with the following structure:

```json
{{
  "series_title": "Title of the series",
  "logline": "One-sentence description of the series",
  "characters": [
    {{"name": "Character 1", "description": "Brief description"}},
    {{"name": "Character 2", "description": "Brief description"}}
  ],
  "episodes": [
    {{
      "episode_number": 1,
      "episode_title": "Title of episode 1",
      "summary": "Summary of episode 1"
    }},
    {{
      "episode_number": 2,
      "episode_title": "Title of episode 2",
      "summary": "Summary of episode 2"
    }}
  ]
}}
```

Ensure your JSON is valid and properly structured. Put the JSON AFTER your markdown response.
"""
                
    # Add prompts for other agent types here
    # For now, let's just create a simple fallback
    return f"""
You are an AI helping create a movie or TV show.

Current title: {state.title}
Current logline: {state.logline}
Number of episodes: {len(state.episodes)}

User request: {user_message}

Please respond with creative content that helps develop this story further.
FORMAT YOUR RESPONSE IN PROPER, CLEAN MARKDOWN, with:
- Clear headings (use # for main heading, ## for subheadings)
- Line breaks between paragraphs
- Proper bullet points where appropriate
- Numbered lists where sequence matters
- Clear formatting for emphasis

Ensure there are proper line breaks between sections and your response is well-structured.
"""

@log_entry_exit
def run_engine(state: StoryState, user_message: str):
    log_flow(f"Starting engine run with message: {user_message[:50]}...")
    
    # Determine which agent to run using the producer
    log_flow("Calling producer agent to determine next agent")
    agent_to_run = producer_agent(state, user_message)
    
    if not agent_to_run or agent_to_run not in agent_map:
        log_flow(f"Invalid agent '{agent_to_run}', defaulting to Writer")
        agent_to_run = "Writer"
    
    log_flow(f"Producer selected agent: {agent_to_run}")
    
    # Run the selected agent
    updated_state = agent_map[agent_to_run](state)
    
    return updated_state, agent_to_run
