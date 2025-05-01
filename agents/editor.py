# agents/editor.py (FIXED)
import os
import subprocess
from utils.types import StoryState

def editor_agent(state: StoryState) -> StoryState:
    video_folder = "videos"
    output_file = "final_output.mp4"
    temp_list_path = os.path.join(video_folder, "video_list.txt")

    # Step 1: Get sorted full paths like: scene_1.mp4, scene_2.mp4
    video_files = sorted([
        f for f in os.listdir(video_folder)
        if f.startswith("scene_") and f.endswith(".mp4")
    ])

    # ✅ Step 2: Write RELATIVE paths to video_list.txt
    with open(temp_list_path, "w") as f:
        for filename in video_files:
            f.write(f"file '{filename}'\n")  # ✅ Just the filename, not full path

    # Step 3: Change working dir to videos/ before running FFmpeg
    try:
        subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", "video_list.txt",
            "-c", "copy",
            output_file
        ], cwd=video_folder, check=True)

        final_output_path = os.path.join(video_folder, output_file)
        print(f"🎬 Final video created at: {final_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg merge failed: {e}")
        final_output_path = ""

    return state.copy(update={
        "final_video": final_output_path
    })
