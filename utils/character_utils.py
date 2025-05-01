"""
Character utilities for managing character information in Studio Nexora.
"""
from datetime import datetime
from db.models.story_projects import finalize_characters
from utils.functional_logger import log_flow, log_error, log_entry_exit

@log_entry_exit
def ensure_character_fields(character_profile):
    """
    Ensure all necessary fields exist in a character profile.
    
    Args:
        character_profile (dict): The character profile to check/update
        
    Returns:
        dict: The updated character profile
    """
    # Create a copy to avoid modifying the original
    profile = character_profile.copy()
    
    # Add default values for required fields if missing
    profile.setdefault("name", "Unknown Character")
    profile.setdefault("token", "")
    profile.setdefault("description", "")
    profile.setdefault("reference_image", "")
    profile.setdefault("hf_lora", "")
    profile.setdefault("combined_lora", "")
    profile.setdefault("ethnicity", "")
    profile.setdefault("gender", "")
    profile.setdefault("matched_at", datetime.utcnow().isoformat())
    
    log_flow(f"Ensured character fields for: {profile.get('name')}")
    return profile

@log_entry_exit
def save_characters_to_db(session_id, character_profiles, character_map=None):
    """
    Save character information to the database.
    
    Args:
        session_id (str): The session ID
        character_profiles (list): List of character profile dictionaries 
        character_map (dict, optional): Mapping of character names to tokens
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not session_id:
        log_error("Cannot save characters: No session ID provided")
        return False
    
    log_flow(f"Saving {len(character_profiles)} characters to database for session {session_id}")
    
    # Clean up character profiles
    clean_profiles = [ensure_character_fields(profile) for profile in character_profiles]
    
    # Generate character map if not provided
    if not character_map:
        log_flow("Generating character map from profiles")
        character_map = {}
        for idx, profile in enumerate(clean_profiles):
            name = profile.get("name", f"Character {idx+1}")
            token = profile.get("token", f"character-{idx+1}")
            character_map[name] = token
    
    # Save to database
    result = finalize_characters(session_id, clean_profiles, character_map)
    log_flow(f"Character save result: {result}")
    return result

@log_entry_exit
def get_character_by_name(character_profiles, name):
    """
    Find a character profile by name.
    
    Args:
        character_profiles (list): List of character profiles
        name (str): The character name to search for
        
    Returns:
        dict or None: The character profile if found, None otherwise
    """
    log_flow(f"Searching for character: {name}")
    for profile in character_profiles:
        if profile.get("name", "").lower() == name.lower():
            log_flow(f"Found character: {name}")
            return profile
    
    log_flow(f"Character not found: {name}", level="warning")
    return None
    
@log_entry_exit
def add_metadata_to_characters(character_profiles, metadata_dict):
    """
    Add additional metadata to character profiles.
    
    Args:
        character_profiles (list): List of character profiles
        metadata_dict (dict): Dictionary mapping character names to metadata
        
    Returns:
        list: Updated character profiles
    """
    log_flow(f"Adding metadata to {len(character_profiles)} characters")
    updated_profiles = []
    
    for profile in character_profiles:
        name = profile.get("name", "")
        if name in metadata_dict:
            # Update the profile with metadata
            log_flow(f"Adding metadata to character: {name}")
            updated_profile = profile.copy()
            updated_profile.update(metadata_dict[name])
            updated_profiles.append(updated_profile)
        else:
            updated_profiles.append(profile)
    
    log_flow(f"Updated {len(updated_profiles)} character profiles with metadata")
    return updated_profiles 