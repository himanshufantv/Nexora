from typing import List, Dict, Any
from pydantic import BaseModel

class StoryState(BaseModel):
    # User context
    user_prompt: str = ""
    story_prompt: str = ""
    producer_notes: str = ""
    session_memory: List[Dict[str, Any]] = []
    last_agent_output: Any = None
    ad_prompts: Dict[str, str] = {}
    ad_images: Dict[str, str] = {}
    ad_character_info: Dict[str, Dict[str, Any]] = {}  # Stores character info for each shot
    scene_seeds: Dict[str, int] = {}  # Stores seeds for scene images


    # High-level metadata
    title: str = ""
    logline: str = ""
    genre: str = ""
    style: str = ""
    writer_profile: str = ""  # ✅ NEW


    # Characters
    characters: List[str] = []                     # Raw character lines
    character_map: Dict[str, str] = {}             # "John Harrow" → "character-1"
    character_profiles: List[Dict[str, Any]] = []  # [{name, description, image_url, seed}]
    structured_characters: List[Dict[str, Any]] = []  # Direct JSON character data

    # Series Structure
    script_outline: str = ""
    three_act_structure: Dict[str, str] = {}

    # Episode-level story
    episodes: List[Dict[str, Any]] = []              # [{episode_title, summary}]
    episode_scripts: Dict[str, List[str]] = {}       # episode_num → scene descriptions (OLD FORMAT)
    scene_scripts: Dict[str, List[Dict[str, str]]] = {}  # "ep1_scene2" → [{shot, dialogue}]
    
    # New structured scene format
    structured_scenes: Dict[str, List[Dict[str, Any]]] = {}  # episode_num → [{scene_number, title, description}]

    # Visual Assets
    scenes: List[str] = []                         # fallback if used in director-only flow
    scene_image_prompts: List[str] = []            # Flux image URLs
    video_clips: List[str] = []           
    session_id: str = ""         # Kling video URLs
    
    # Structured Data Storage (from JSON responses)
    structured_data: Dict[str, Any] = {}  # Stores complete JSON data from responses
