import random
import difflib
from datetime import datetime
from utils.types import StoryState
from utils.logger import log_agent_output
from db.models.casting_characters import get_candidate_characters
from db.models.story_projects import update_story_project_by_session, finalize_characters

def extract_character_info(raw: str):
    """Split character name and description cleanly."""
    print(f"⚙️ Processing character: {raw}")
    name = raw.split(",")[0].strip() if "," in raw else raw.split(" - ")[0].strip() if " - " in raw else raw.split(":")[0].strip() if ":" in raw else raw
    description = raw.split(",", 1)[1].strip() if "," in raw else raw.split(" - ", 1)[1].strip() if " - " in raw else raw.split(":", 1)[1].strip() if ":" in raw else "A character in the story."
    print(f"⚙️ Extracted name: '{name}', description: '{description[:50]}...'")
    return name, description

def guess_gender_from_name(name):
    """Try to determine gender based on common name patterns, especially for Indian names."""
    name = name.lower()
    
    # Common Indian male name endings
    male_endings = ['raj', 'deep', 'esh', 'eet', 'ant', 'ay', 'ul', 'un', 'it', 'ik', 'ev', 'am', 'an']
    # Common Indian male names
    male_names = ['aman', 'arjun', 'aarav', 'vikram', 'rahul', 'rohan', 'akash', 'aditya', 'varun', 
                 'vijay', 'karan', 'nikhil', 'suresh', 'ramesh', 'mahesh', 'rajesh', 'anil', 'sunil']
    
    # Common Indian female name endings
    female_endings = ['i', 'a', 'ee', 'ya', 'na', 'ti', 'ni', 'ma', 'ta', 'vi', 'ha', 'ja']
    # Common Indian female names
    female_names = ['kavya', 'naina', 'priya', 'meera', 'neha', 'divya', 'pooja', 'anjali', 'shreya', 
                   'swati', 'rani', 'sunita', 'anita', 'sarita', 'deepti', 'jyoti', 'sakshi']
    
    # Check for exact name matches first
    if name in male_names:
        return "male"
    if name in female_names:
        return "female"
    
    # Then check name endings
    for ending in male_endings:
        if name.endswith(ending) and len(name) > len(ending):
            return "male"
    
    for ending in female_endings:
        if name.endswith(ending) and len(name) > len(ending):
            return "female"
    
    return None

def extract_info(description: str, character_name=None):
    """Extract gender, age, nationality roughly from description and name."""
    desc = description.lower()
    gender = None
    age = None
    nationality = None
    
    # First try to determine gender from character name if provided
    if character_name:
        gender = guess_gender_from_name(character_name)
        if gender:
            print(f"📊 Determined gender '{gender}' from character name '{character_name}'")

    # Common Indian male and female name patterns and suffixes
    indian_male_indicators = ["kumar", "singh", "raj", "aman", "arav", "vikram", "arjun", "rohan", "rahul", "suresh", "nikhil", "varun"]
    indian_female_indicators = ["devi", "kumari", "kaur", "kavya", "naina", "meera", "priya", "neha", "pooja", "anjali", "shreya", "divya"]
    
    # Only continue with description-based gender detection if we don't have a gender yet
    if gender is None:
        # Check common Indian name patterns in description
        for indicator in indian_male_indicators:
            if indicator in desc.split():
                gender = "male"
                break
                
        for indicator in indian_female_indicators:
            if indicator in desc.split():
                gender = "female"
                break
                
        # If no pattern match, use regular keyword detection
        if gender is None:
            if any(word in desc for word in ["female", "woman", "girl", "she", "her", "feminine", "lady", "wife", "daughter", "sister"]):
                gender = "female"
            elif any(word in desc for word in ["male", "man", "boy", "he", "his", "masculine", "guy", "husband", "son", "brother"]):
                gender = "male"
    
    # Improved age detection
    age_keywords = {
        "20s": 20, "twenties": 20, "early twenties": 20, "mid twenties": 25, "late twenties": 28,
        "30s": 30, "thirties": 30, "early thirties": 30, "mid thirties": 35, "late thirties": 38,
        "40s": 40, "forties": 40
    }
    for keyword, value in age_keywords.items():
        if keyword in desc:
            age = value
            break
    
    # Expanded nationality detection
    nationality_keywords = {
        "american": "american", "usa": "american", "states": "american", "america": "american",
        "indian": "indian", "india": "indian", "hindi": "indian", "himalayan": "indian", "himalaya": "indian",
        "british": "british", "uk": "british", "england": "british",
    }
    for keyword, value in nationality_keywords.items():
        if keyword in desc:
            nationality = value
            break
    
    print(f"⚙️ Extracted attributes - Gender: {gender}, Age: {age}, Nationality: {nationality}")
    return gender, age, nationality

def text_similarity(a: str, b: str) -> float:
    """Rough text similarity score using difflib."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def name_similarity(character_name: str, candidate_name: str) -> float:
    """Check if character name matches or contains database entry name."""
    character_name = character_name.lower()
    candidate_name = candidate_name.lower()
    
    # Direct name match (e.g., "Sakshi" == "Sakshi")
    if character_name == candidate_name:
        return 1.0
    
    # Name contained in character name (e.g., "Sakshi Patel" contains "Sakshi")
    if candidate_name in character_name:
        return 0.9
    
    # Partial name match using similarity
    return text_similarity(character_name, candidate_name) 

def filter_candidates(candidates, gender, age, nationality=None, character_name=None):
    """Filter candidates with more flexible matching and prioritize name matches."""
    print(f"⚙️ Filtering {len(candidates)} candidates")
    
    # First try direct name matching if available
    if character_name:
        for candidate in candidates:
            db_name = candidate.get("name", "")
            # If database has a name field and it matches/contains our character name
            if db_name and name_similarity(character_name, db_name) > 0.7:
                print(f"✅ Found direct name match: {db_name} for character {character_name}")
                return [candidate]
    
    filtered = []
    for c in candidates:
        # More flexible gender matching - if we have a gender, it should match
        if gender and c.get("gender", "").lower() != gender and gender != "unknown":
            print(f"❌ Gender mismatch: character={gender}, candidate={c.get('gender')}")
            continue
            
        # More flexible age matching - match within 10 years if age specified
        if age:
            try:
                cand_age = int(c.get("age", "").replace("s", "").strip())
                if abs(cand_age - age) > 10:
                    print(f"❌ Age mismatch: character={age}, candidate={c.get('age')}")
                    continue
            except:
                pass # Don't filter if age parsing fails
        
        # Optional nationality filter 
        if nationality and nationality not in c.get("description", "").lower():
            if c.get("ethnicity", "").lower() != nationality:
                print(f"❌ Nationality mismatch: character={nationality}, candidate={c.get('ethnicity')}")
                continue
        
        print(f"✅ Candidate passed filters: {c.get('gender')}, {c.get('age')}, ID: {c.get('_id')}")
        filtered.append(c)
    
    # If no matches with all filters, try with just gender
    if not filtered and gender:
        print("⚠️ No matches with all filters, trying with just gender")
        return [c for c in candidates if c.get("gender", "").lower() == gender]
    
    # If still no matches, return a random candidate as last resort
    if not filtered and candidates:
        print("⚠️ No matches at all, using random candidate")
        return [random.choice(candidates)]
        
    return filtered

def casting_agent(state: StoryState) -> StoryState:
    print("🎭 Running Casting Agent...")

    character_profiles = []
    character_map = {}
    characters = state.characters
    token_prefix = "character"

    # Set the session_id on the state if not already set
    if not state.session_id:
        print("⚠️ Warning: No session_id set on state")
        
    print(f"🔍 Getting candidates from database...")
    candidates = get_candidate_characters()
    print(f"📊 Retrieved {len(candidates)} candidates from database")
    
    # Debug: Print first 2 candidates to verify database connection
    for i, c in enumerate(candidates[:2]):
        print(f"📝 DB Candidate {i+1}: gender={c.get('gender')}, age={c.get('age')}, image_url={c.get('image_url', '')[:30]}...")

    if not candidates:
        print("⚠️ No candidates found in DB. Fallback to default empty characters.")
        return state

    for idx, raw in enumerate(characters):
        name, description = extract_character_info(raw)
        gender, age, nationality = extract_info(description, name)
        token = f"{token_prefix}-{idx + 1}"
        character_map[name] = token

        # Try with specific name matches first
        print(f"🔍 Finding matches for: {name} ({gender}, {age})")
        eligible_candidates = filter_candidates(candidates, gender, age, nationality, name)

        if not eligible_candidates:
            print(f"⚠️ No eligible candidates found for {name}. Using random candidate.")
            eligible_candidates = [random.choice(candidates)]

        scored_candidates = [
            (c, text_similarity(description, c.get("description", "")))
            for c in eligible_candidates
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        chosen = scored_candidates[0][0]
        similarity_score = scored_candidates[0][1]

        print(f"✅ {name} ({token}) matched → {chosen.get('image_url', '')[:30]}... [similarity: {similarity_score:.2f}]")

        # Store all details from the chosen candidate
        character_profile = {
            # Character-specific details
            "name": name,
            "token": token,
            "description": description,
            
            # Include all fields from the chosen candidate
            "reference_image": chosen.get("image_url"),
            "combined_lora": chosen.get("combined_lora", None),
            "hf_lora": chosen.get("hf_lora", ""),
            "ethnicity": chosen.get("ethnicity", ""),
            "gender": chosen.get("gender", ""),
            "age": chosen.get("age", ""),
            "original_id": str(chosen.get("_id", "")),
            "candidate_name": chosen.get("name", ""),
            "matched_at": datetime.utcnow().isoformat(),
            "candidiate_description": chosen.get("description", ""),
            "height": chosen.get("height", ""),
            "weight": chosen.get("weight", ""),
            "eye_color": chosen.get("eye_color", ""),
            "hair_color": chosen.get("hair_color", ""),
            "body_type": chosen.get("body_type", ""),
            "facial_features": chosen.get("facial_features", ""),
            "skin_tone": chosen.get("skin_tone", ""),
            "clothing_style": chosen.get("clothing_style", ""),
            "personality": chosen.get("personality", ""),
            "distinguishing_features": chosen.get("distinguishing_features", ""),
            "background": chosen.get("background", ""),
            "occupation": chosen.get("occupation", ""),
            "additional_attributes": chosen.get("additional_attributes", {})
        }
        
        character_profiles.append(character_profile)

    # 🛠 Now apply compatible_with LoRA logic if 2 characters
    if len(character_profiles) >= 2:
        char1 = character_profiles[0]
        char2 = character_profiles[1]

        char1_db = next((c for c in candidates if c.get("image_url") == char1["reference_image"]), None)
        char2_db = next((c for c in candidates if c.get("image_url") == char2["reference_image"]), None)

        if char1_db and char2_db:
            compatible = char1_db.get("compatible_with", {})
            if str(char2_db["_id"]) in compatible:
                combined_lora = compatible[str(char2_db["_id"])].get("combined_lora")
                if combined_lora:
                    print(f"🔗 Found Combined LoRA between {char1['name']} and {char2['name']}: {combined_lora}")
                    char1["combined_lora"] = combined_lora
                    char2["combined_lora"] = combined_lora

    # Update state
    state.character_profiles = character_profiles
    state.character_map = character_map

    print("📋 Final Character Profiles:")
    for profile in character_profiles:
        print(f"📋 {profile['name']}: {profile['reference_image'][:50]}...")
    
    # 🛠 Save character_profiles and character_map inside story_data using new function
    if state.session_id:
        try:
            success = finalize_characters(state.session_id, character_profiles, character_map)
            if success:
                print("✅ [DB] Successfully finalized characters and saved to database.")
            else:
                print("⚠️ [DB] Character finalization completed with warnings.")
        except Exception as e:
            print(f"❌ [DB] Failed to finalize characters: {str(e)}")
            
            # Fallback to old method if new one fails
            try:
                update_story_project_by_session(state.session_id, {
                    "character_profiles": character_profiles,
                    "character_map": character_map,
                })
                print("✅ [DB] Successfully updated story project with casting results (fallback method).")
            except Exception as e2:
                print(f"❌ [DB] All database update methods failed: {str(e2)}")
    else:
        print("⚠️ No session_id available, skipping database update")

    # Log cleanly
    try:
        log_agent_output("Casting Agent", {"character_profiles": character_profiles})
    except Exception as e:
        print(f"⚠️ Logging warning: {str(e)}")

    return state
