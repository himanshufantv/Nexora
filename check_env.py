import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Check environment variables"""
    print("Checking environment variables...")
    
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")
    mongo_uri = os.getenv("MONGO_URI")
    
    print(f"REPLICATE_API_TOKEN: {'SET' if replicate_token else 'NOT SET'}")
    if replicate_token:
        masked_token = f"{replicate_token[:4]}...{replicate_token[-4:]}"
        print(f"  Token preview: {masked_token}")
    
    print(f"OPENAI_API_KEY: {'SET' if openai_key else 'NOT SET'}")
    if openai_key:
        masked_key = f"{openai_key[:4]}...{openai_key[-4:]}"
        print(f"  Key preview: {masked_key}")
    
    print(f"MONGO_URI: {'SET' if mongo_uri else 'NOT SET'}")
    if mongo_uri:
        # Safely mask MongoDB URI
        if "@" in mongo_uri:
            parts = mongo_uri.split("@")
            credentials = parts[0]
            if ":" in credentials:
                user_pass = credentials.split(":")
                masked_uri = f"{user_pass[0]}:****@{parts[1]}"
                print(f"  URI preview: {masked_uri}")
            else:
                print(f"  URI contains credentials but in unexpected format")
        else:
            print(f"  URI preview: {mongo_uri}")
        
if __name__ == "__main__":
    main() 