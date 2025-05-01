# agents/writer.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.types import StoryState
from utils.parser import safe_parse_json_string
from utils.logger import log_agent_output
from typing import Any
from datetime import datetime
from utils.db import get_writer_profiles_collection

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load writer profile from JSON
def load_writer_profile(profile_key: str) -> dict:
    collection = get_writer_profiles_collection()
    profile = collection.find_one({"profile_key": profile_key})
    if not profile:
        raise ValueError(f"❌ Writer profile '{profile_key}' not found in database.")
    
    # Clean Mongo _id field
    profile.pop("_id", None)
    return profile

# Main Writer Agent
def writer_agent(state: StoryState, user_message: str, profile_key: str = "english_romantic") -> StoryState:
    profile = load_writer_profile(profile_key)
    system_prompt = profile["system_prompt"]
    language = profile["language"]
    tone = profile["tone"]
    style = profile["style_note"]
    genre = profile["genre"]

    if "scene" in user_message.lower():
        return generate_scene_script(state, user_message, system_prompt)
    elif "episode" in user_message.lower():
        return generate_episode_script(state, user_message, system_prompt)
    else:
        return generate_series_synopsis(state, user_message, system_prompt, language, tone, style, genre)

# Level 1: Series Synopsis
def generate_series_synopsis(state, user_message, system_prompt, language, tone, style, genre):
    prompt = f"""
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

Respond in this format:
{{
  "series_title": "...",
  "logline": "...",
  "characters": ["Name, description", "..."],
  "episodes": [
    {{ "episode_title": "...", "summary": "..." }},
    ...
  ]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    parsed = safe_parse_json_string(response.choices[0].message.content)
    new_state = state.copy(update={
        "story_prompt": user_message,
        "title": parsed.get("series_title", ""),
        "logline": parsed.get("logline", ""),
        "characters": parsed.get("characters", []),
        "episodes": parsed.get("episodes", [])
    })
    new_state.last_agent_output = parsed
    log_agent_output("Writer", new_state)
    return new_state
# Level 2: Episode Breakdown
def generate_episode_script(state, user_message, system_prompt):
    episode_number = int("".join(filter(str.isdigit, user_message)))
    episode = state.episodes[episode_number - 1]
    title = episode["episode_title"]
    summary = episode["summary"]

    prompt = f"""
{system_prompt}

Break this episode into 6-8 scenes.
only 1-2 characters per scene
Title: {title}
Summary: {summary}

Return JSON:
["Scene 1...", "Scene 2...", ...]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    scenes = safe_parse_json_string(response.choices[0].message.content)

    new_state = state.copy(update={
        "episode_scripts": {
            **state.episode_scripts,
            episode_number: scenes
        }
    })

    new_state.last_agent_output = scenes
    log_agent_output("Writer", new_state)
    return new_state

# Level 2: Episode Breakdown
def generate_scene_script(state, user_message, system_prompt):
    # Attempt to extract numbers
    nums = [int(s) for s in user_message.split() if s.isdigit()]
    
    if len(nums) == 1:
        episode_number = 1
        scene_number = nums[0]
    elif len(nums) >= 2:
        episode_number, scene_number = nums[0], nums[1]
    else:
        raise ValueError("❌ Please specify the scene number (e.g. 'scene 1' or 'scene 1 of episode 2')")

    scene_list = state.episode_scripts.get(episode_number, [])
    if scene_number > len(scene_list) or scene_number < 1:
        raise IndexError(f"❌ Scene {scene_number} does not exist in Episode {episode_number}.")

    scene_text = scene_list[scene_number - 1]

    prompt = f"""
{system_prompt}

Break this scene into cinematic shots:
only 1-2 characters per shot
Scene: {scene_text}

Return format:
[
  {{ "shot": "...", "dialogue": "..." }},
  ...
]
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85
    )
    shot_list = safe_parse_json_string(response.choices[0].message.content)
    scene_key = f"ep{episode_number}_scene{scene_number}"
    new_state = state.copy(update={
        "scene_scripts": {
            **state.scene_scripts,
            scene_key: shot_list
        }
    })
    new_state.last_agent_output = shot_list
    log_agent_output("Writer", new_state)
    return new_state
