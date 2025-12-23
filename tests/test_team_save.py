import sys
import os
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import init_db, get_db, User, SavedTeam, SessionLocal
from src.auth import create_access_token, get_password_hash
from jose import jwt
from src.auth import SECRET_KEY, ALGORITHM

def test_team_saving():
    print("Initializing DB...")
    init_db()
    db = SessionLocal()
    
    username = "test_user_saver"
    password = "test_password"
    team_name = "Test Save Team"
    
    # Clean up
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        # Cascade delete should handle teams if configured, but let's be safe
        db.query(SavedTeam).filter(SavedTeam.user_id == existing_user.id).delete()
        db.delete(existing_user)
        db.commit()

    print(f"Creating user: {username}")
    hashed_password = get_password_hash(password)
    user = User(username=username, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token(data={"sub": user.username})
    
    # Mimic the save_team_api fallback logic from streamlit_app.py
    # We want to see if passing dict works (after fix) or if passing string is required (before fix - but that's what's broken for JSON type)
    
    print("Attempting to save team...")
    team_data_dict = {"players": [1, 2], "captain": 1}
    
    # In the BROKEN code, it does json.dumps(team_data_dict)
    # In the FIXED code, it should be team_data_dict (as dict)
    
    # We will try to replicate what the APP does. 
    # Current App Code: team_data=json.dumps(team_data)
    
    try:
        # Simulate logic step-by-step
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        u_name = payload.get("sub")
        u = db.query(User).filter(User.username == u_name).first()
        
        if not u:
             print("User not found via token")
             return

        # Attempting save with just dict (this is the desired state)
        # If we use json.dumps() here, it might be double serialized or cause type issues if SQLAlchemy expects JSON
        
        # NOTE: sqlite + sqlalchemy JSON type usually handles dicts fine. 
        # But if we pass a string, it might treat it as a string instead of JSON object if the column is JSON.
        
        # Let's try the WAY the app does it currently to PROVE failure (or behavior).
        # App currently does: team_data=json.dumps(team_data)
        
        try:
            print("Trying to save with DICT (FIXED BEHAVIOR)...")
            # val_to_save = json.dumps(team_data_dict) 
            # Passing dict directly
            t = SavedTeam(name=team_name, team_data=team_data_dict, user_id=u.id)
            db.add(t)
            db.commit()
            print("Saved with dict directly.")
        except Exception as e:
            print(f"Failed to save with dict: {e}")
            db.rollback()

    except Exception as e:
        print(f"General Error: {e}")
    
    # Check what is in DB
    saved = db.query(SavedTeam).filter(SavedTeam.name == team_name).first()
    if saved:
        print(f"Saved Data Type: {type(saved.team_data)}")
        print(f"Saved Data: {saved.team_data}")
        # If it's a string, it's wrong (double encoded or just string). If it's dict, it's good.
        if isinstance(saved.team_data, str):
             print("DETECTED ISSUE: Data is stored as string (double encoded?)")
        elif isinstance(saved.team_data, dict):
             print("SUCCESS: Data is stored as dict (JSON)")
    else:
        print("No team found.")

    db.close()

if __name__ == "__main__":
    test_team_saving()
