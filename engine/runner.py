# engine/runner.py
from agents.producer import producer_agent
from utils.types import  StoryState

# Agent registry
from agents.writer import writer_agent
from agents.director import director_agent
from agents.casting import casting_agent
from agents.ad import ad_agent
from agents.video_design import video_design_agent
from agents.editor import editor_agent

AGENTS = {
    "Writer": writer_agent,
    "Director": director_agent,
    "Casting": casting_agent,
    "AD": ad_agent,
    "VideoDesign": video_design_agent,
    "Editor": editor_agent
}

def run_agent(state: StoryState, user_message: str) -> tuple[StoryState, str, str]:
    from agents.producer import producer_agent  # local import to avoid recursion

    # Decide what to run
    next_agent = producer_agent(state, user_message)

    if next_agent not in AGENTS:
        response = f"⚠️ No valid next agent found for: '{user_message}'"
        next_agent = "None"
        return state, response, next_agent

    agent_func = AGENTS[next_agent]
    new_state = agent_func(state)

    # Add to session memory
    new_state.session_memory.append({
        "user_message": user_message,
        "agent": next_agent,
        "agent_response_summary": f"{next_agent} completed.",
        "producer_decision": next_agent
    })

    return new_state, f"✅ Ran {next_agent} Agent.", next_agent

