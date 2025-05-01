# merge_only.py
from agents.editor import editor_agent
from utils.types import StoryState

def merge_videos():
    dummy_state = StoryState(
        video_clips=[]  # Doesn't matter; Editor reads from videos/ folder
    )
    final_state = editor_agent(dummy_state)
    print("✅ Final merged video:", final_state.final_video)

if __name__ == "__main__":
    merge_videos()
