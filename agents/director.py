# agents/director.py
from utils.types import StoryState
from utils.logger import log_agent_output  # ✅ Add this
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def director_agent(state: StoryState) -> StoryState:
    title = state.title
    logline = state.logline
    characters = "\n".join(state.characters)
    acts = state.three_act_structure
    outline = state.script_outline

    prompt = f"""
You are a cinematic director for AI-generated movies.

Given the following story details, break it into 6 visually rich SCENES.
Each scene must have:
- A **scene title**
- A **description of the setting**
- **Key characters involved**
- The **emotional tone**
- A **summary of the scene’s visual action**

Story Title: {title}

Logline:
{logline}

Characters:
{characters}

Story Outline:
{outline}

Three Act Structure:
Act 1: {acts.get('act_1', '')}
Act 2: {acts.get('act_2', '')}
Act 3: {acts.get('act_3', '')}

Respond in the following JSON format:
[
  {{
    "scene_title": "...",
    "setting": "...",
    "characters": ["..."],
    "tone": "...",
    "action_summary": "..."
  }},
  ...
]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.75
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed_scenes = json.loads(content)
        scenes_list = [
            f"{scene['scene_title']}: {scene['action_summary']}" for scene in parsed_scenes
        ]
    except Exception:
        scenes_list = [content]  # fallback to raw text if parse fails

    new_state = state.copy(update={
        "scenes": scenes_list
    })

    log_agent_output("Director", new_state)  # ✅ Save log to disk

    return new_state
