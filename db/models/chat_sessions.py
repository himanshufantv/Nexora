# db/models/chat_sessions.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime
import pprint

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["StudioNexora"]
chat_sessions_collection = db["chat_sessions"]

def create_chat_session(user_id: str, session_id: str):
    document = {
        "user_id": user_id,
        "session_id": session_id,
        "created_at": datetime.utcnow(),
        "messages": []
    }
    print(f"🛠 [DB] Inserting into 'chat_sessions' → {pprint.pformat(document)}")
    try:
        result = chat_sessions_collection.insert_one(document)
        print(f"✅ [DB] Chat session inserted. Inserted ID: {result.inserted_id}")
    except Exception as e:
        print(f"❌ [DB] Failed to insert chat session: {e}")

def log_chat_message(session_id: str, sender: str, message: str):
    log_entry = {
        "sender": sender,
        "message": message,
        "timestamp": datetime.utcnow()
    }
    print(f"🛠 [DB] Logging message into 'chat_sessions' for session_id={session_id} → {pprint.pformat(log_entry)}")
    try:
        result = chat_sessions_collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": log_entry}}
        )
        if result.modified_count == 1:
            print("✅ [DB] Successfully logged chat message.")
        else:
            print("⚠️ [DB] No document found to update.")
    except Exception as e:
        print(f"❌ [DB] Failed to log chat message: {e}")

def get_chat_history(session_id: str):
    print(f"🛠 [DB] Fetching chat history for session_id={session_id}")
    try:
        chat = chat_sessions_collection.find_one({"session_id": session_id})
        if chat:
            print("✅ [DB] Chat history found.")
        else:
            print("⚠️ [DB] No chat history found for this session.")
        return chat
    except Exception as e:
        print(f"❌ [DB] Failed to fetch chat history: {e}")
        return None
