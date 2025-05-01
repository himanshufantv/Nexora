# graph/story_graph.py
from langgraph.graph import StateGraph
from agents.producer import producer_agent
from agents.writer import writer_agent
from agents.director import director_agent
from agents.casting import casting_agent
from agents.ad import ad_agent
from agents.video_design import video_design_agent
from agents.editor import editor_agent
from utils.types import StoryState  # import the Pydantic schema

def build_story_graph():
    builder = StateGraph(state_schema=StoryState)

    def run_producer(state: StoryState) -> dict:
        last_message = state.session_memory[-1]["user_message"] if state.session_memory else ""
        agent_key = producer_agent(state, last_message)
        return {"next": agent_key}  # LangGraph expects this dict

    def run_writer(state: StoryState) -> StoryState:
        last_message = state.session_memory[-1]["user_message"] if state.session_memory else ""
        agent_key = producer_agent(state, last_message)
        profile_key = agent_key.split("::")[1] if "::" in agent_key else "english_romantic"
        return writer_agent(state, last_message, profile_key)

    builder.add_node("Producer", run_producer)
    builder.add_node("Writer", run_writer)
    builder.add_node("Director", director_agent)
    builder.add_node("Casting", casting_agent)
    builder.add_node("AD", ad_agent)
    builder.add_node("VideoDesign", video_design_agent)
    builder.add_node("Editor", editor_agent)

    builder.set_entry_point("Producer")
    builder.add_edge("Producer", "Writer")
    builder.add_edge("Writer", "Director")
    builder.add_edge("Director", "Casting")
    builder.add_edge("Casting", "AD")
    builder.add_edge("AD", "VideoDesign")
    builder.add_edge("VideoDesign", "Editor")
    builder.set_finish_point("Editor")

    return builder.compile()