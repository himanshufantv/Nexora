# db/models/story_projects.py

from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime
from utils.functional_logger import log_flow, log_db_operation, log_error, log_entry_exit

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["StudioNexora"]
story_projects_collection = db["story_projects"]

@log_entry_exit
def find_project_by_session_id(session_id: str):
    """
    Find a story project by session_id.
    
    Args:
        session_id (str): The session ID to search for
        
    Returns:
        dict or None: The story project if found, None otherwise
    """
    if not session_id:
        log_error("Cannot find project: No session ID provided")
        return None
        
    try:
        log_db_operation("find_one", "story_projects", {"session_id": session_id})
        project = story_projects_collection.find_one({"session_id": session_id})
        if project:
            log_flow(f"Found project for session: {session_id}")
            return project
        else:
            log_flow(f"No project found for session: {session_id}", level="warning")
            return None
    except Exception as e:
        log_error(f"Error finding project", e)
        return None

@log_entry_exit
def save_story_project(user_id: str, session_id: str, story_data: dict):
    # 🔥 FIX 1: Convert episode_scripts keys to string
    if "episode_scripts" in story_data and isinstance(story_data["episode_scripts"], dict):
        log_flow("Converting episode_scripts keys to strings")
        story_data["episode_scripts"] = {str(k): v for k, v in story_data["episode_scripts"].items()}

    # 🔥 FIX 2: Check if a project already exists for this session_id
    log_db_operation("find_one", "story_projects", {"session_id": session_id})
    existing_project = story_projects_collection.find_one({"session_id": session_id})

    if existing_project:
        # 🔥 Update the existing story project
        log_flow(f"Updating existing story project for session: {session_id}")
        log_db_operation("update_one", "story_projects", {"session_id": session_id})
        story_projects_collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "story_data": story_data,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        log_flow(f"Updated story project for session: {session_id}")
    else:
        # 🔥 Insert a new story project
        log_flow(f"Inserting new story project for session: {session_id}")
        log_db_operation("insert_one", "story_projects")
        story_projects_collection.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "story_data": story_data
        })
        log_flow(f"Inserted new story project for session: {session_id}")


@log_entry_exit
def update_story_project_by_session(session_id: str, update_fields: dict):
    """Update fields inside story_data."""
    log_flow(f"Trying to update session_id: {session_id}")
    
    # Prepare the update document - only include fields that are provided
    update_doc = {"updated_at": datetime.utcnow()}
    
    # Handle character_profiles (ensure all fields are preserved)
    if "character_profiles" in update_fields and update_fields["character_profiles"]:
        log_flow(f"Updating character profiles for session: {session_id}")
        # Make sure all necessary fields are present in each profile
        for profile in update_fields["character_profiles"]:
            # Ensure these fields exist, setting defaults if not
            profile.setdefault("name", "Unknown")
            profile.setdefault("token", "")
            profile.setdefault("description", "")
            profile.setdefault("reference_image", "")
            profile.setdefault("hf_lora", "")
            profile.setdefault("combined_lora", "")
            
            # Debug info for each character
            log_flow(f"Character: {profile['name']}")
            log_flow(f"- Image: {profile.get('reference_image', '')[:30]}...")
            log_flow(f"- LoRA: {profile.get('combined_lora', '')}")
        
        update_doc["story_data.character_profiles"] = update_fields["character_profiles"]
    
    # Handle character_map
    if "character_map" in update_fields and update_fields["character_map"]:
        log_flow("Updating character_map")
        update_doc["story_data.character_map"] = update_fields["character_map"]
    
    # Handle any other fields that might be included
    for key, value in update_fields.items():
        if key not in ["character_profiles", "character_map"] and value is not None:
            log_flow(f"Updating field: {key}")
            update_doc[f"story_data.{key}"] = value
    
    # Print update details
    log_flow(f"Update document keys: {list(update_doc.keys())}")
    
    # Perform the update
    log_db_operation("update_one", "story_projects", {"session_id": session_id})
    result = story_projects_collection.update_one(
        {"session_id": session_id},
        {"$set": update_doc}
    )

    if result.matched_count == 0:
        log_flow(f"No story project found for session {session_id}", level="warning")
        return 0
    elif result.modified_count == 0:
        log_flow(f"Story project found but no new data to update for session {session_id}")
        return 0
    else:
        log_flow(f"Successfully updated story project for session {session_id}")
        
        # Verify the update by fetching the updated document
        log_db_operation("find_one", "story_projects", {"session_id": session_id})
        updated = story_projects_collection.find_one({"session_id": session_id})
        if updated and "story_data" in updated:
            char_profiles = updated["story_data"].get("character_profiles", [])
            log_flow(f"Verified: {len(char_profiles)} character profiles saved to database")
            
            # Log summary of saved characters
            for i, profile in enumerate(char_profiles[:3]):  # Show first 3 only
                log_flow(f"[{i+1}] {profile.get('name', 'Unknown')}: {profile.get('reference_image', '')[:30]}...")
        
        return result.modified_count

# Add a helper function to ensure character details are properly saved
@log_entry_exit
def finalize_characters(session_id: str, character_profiles: list, character_map: dict = None):
    """
    Special function to ensure character details are properly finalized and saved.
    
    Args:
        session_id: The session ID
        character_profiles: List of character profile objects
        character_map: Optional mapping of character names to tokens
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        log_flow(f"Finalizing characters for session: {session_id}")
        
        # If character_map not provided, generate it from profiles
        if not character_map:
            character_map = {}
            for idx, profile in enumerate(character_profiles):
                name = profile.get("name", f"Character {idx+1}")
                token = profile.get("token", f"character-{idx+1}")
                character_map[name] = token
                log_flow(f"Generated token '{token}' for character '{name}'")
        
        # Ensure all needed fields exist in each profile
        for idx, profile in enumerate(character_profiles):
            # Set defaults for required fields
            if "token" not in profile:
                profile["token"] = f"character-{idx+1}"
            
            # Debug info
            log_flow(f"Finalizing character: {profile.get('name', 'Unknown')}")
            log_flow(f"- Image: {profile.get('reference_image', '')[:30]}...")
            log_flow(f"- LoRA info: {profile.get('combined_lora', 'None')}")
        
        # Update the database with comprehensive character information
        log_flow("Updating database with finalized character information")
        update_story_project_by_session(
            session_id, 
            {
                "character_profiles": character_profiles,
                "character_map": character_map
            }
        )
        
        log_flow(f"Characters finalized successfully for session {session_id}")
        return True
        
    except Exception as e:
        log_error(f"Error finalizing characters", e)
        return False
