# utils/db.py

import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize global client and database objects
client = None
db = None

def get_db_connection():
    """Get a connection to the MongoDB database"""
    global client, db
    
    if client is None:
        try:
            client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=2000)
            # Test the connection
            client.server_info()
            db = client["StudioNexora"]
            print(f"✅ MongoDB connection established")
            return db
        except Exception as e:
            print(f"⚠️ MongoDB connection error: {e}")
            print("Continuing without database connectivity...")
            return None
    else:
        return db

def get_chat_sessions_collection():
    """Get the chat_sessions collection"""
    db = get_db_connection()
    if db is not None:
        return db["chat_sessions"]
    return None

def get_story_projects_collection():
    """Get the story_projects collection"""
    db = get_db_connection()
    if db is not None:
        return db["story_projects"]
    return None

def get_writer_profiles_collection():
    """Get the writer_profiles collection object"""
    global client, db
    
    try:
        # Initialize connection if not already done
        if client is None:
            db = get_db_connection()
        
        # Check if db is initialized
        if db is not None:
            print(f"Accessing writer_profiles collection")
            return db.get_collection("writer_profiles")
        else:
            print(f"Database connection not established when trying to access writer_profiles")
            return None
    except Exception as e:
        print(f"Error accessing writer_profiles collection: {e}")
    return None

def get_db():
    """Return the database object."""
    return get_db_connection()

def get_collections():
    """Return the collections we use."""
    return get_story_projects_collection(), get_chat_sessions_collection()
