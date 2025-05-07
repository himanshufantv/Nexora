# main.py
import uuid
from agents.producer import producer_agent
from agents.writer import writer_agent
from agents.director import director_agent
from agents.casting import casting_agent
from agents.ad import ad_agent
from agents.video_design import video_design_agent
from agents.editor import editor_agent
from utils.types import StoryState
from utils.chat_logger import start_new_session_log, log_chat_turn
from db.models.chat_sessions import create_chat_session, log_chat_message
from db.models.story_projects import save_story_project
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.project_api import router as project_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(project_router)
AGENTS = {
    "Writer": writer_agent,
    "Director": director_agent,
    "Casting": casting_agent,
    "AD": ad_agent,
    "VideoDesign": video_design_agent,
    "Editor": editor_agent,
}

def run_interactive_chat():
    state = StoryState()
    session_id = str(uuid.uuid4())
    user_id = "user123"  # In real system, this would come from auth
    session_path = start_new_session_log()
    create_chat_session(user_id, session_id)

    print(f"\n🎬 Studio Nexora Chat Started! (Session: {session_id})\n")

    while True:
        user_input = input("🧠 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Update local session memory
        state.session_memory.append({"user_message": user_input})

        # Update chat_sessions collection
        log_chat_message(session_id=session_id, sender="user", message=user_input)

        agent_key = producer_agent(state, user_input)

        if "::" in agent_key:
            agent_name, profile_key = agent_key.split("::")
            agent_fn = AGENTS.get(agent_name)
            state = agent_fn(state, user_input, profile_key)
        else:
            agent_name = agent_key
            agent_fn = AGENTS.get(agent_key)
            state = agent_fn(state)

        # Handle logging
        if agent_name == "Writer":
            if state.scene_scripts:
                agent_response = str(state.scene_scripts)
            elif state.episode_scripts:
                agent_response = str(state.episode_scripts)
            elif state.episodes:
                agent_response = str(state.episodes)
            else:
                agent_response = "Writer agent executed."
        else:
            agent_response = f"{agent_name} agent executed."

        log_chat_turn(session_path, user_input, agent_name, agent_response)
        log_chat_message(session_id=session_id, sender="nexora", message=agent_response)

        print(f"\n✅ Agent Run: {agent_key}")
        print(f"📝 Title: {state.title}")
        print(f"🎞️ Episodes: {len(state.episodes)} | Scenes: {len(state.episode_scripts)} | Shots: {len(state.scene_scripts)}\n")

        # ✅ Save to story_projects when moving past synopsis
        if agent_name == "Writer" and (state.episode_scripts or state.scene_scripts):
            save_story_project(user_id=user_id, session_id=session_id, story_data=state.model_dump())
            print("💾 Project saved to StudioNexora database!")

if __name__ == "__main__":
    run_interactive_chat()
