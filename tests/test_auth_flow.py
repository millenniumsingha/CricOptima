from fastapi.testclient import TestClient
from api.main import app
from src.db import Base, engine, SessionLocal
import pytest

# Test Client
client = TestClient(app)

# Setup/Teardown DB
@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    # Base.metadata.drop_all(bind=engine)  # Optional cleanup

def test_auth_and_team_flow(setup_db):
    # 1. Register
    import uuid
    username = f"integration_test_user_{uuid.uuid4().hex[:8]}"
    password = "securepassword"
    
    # Try login first (should fail)
    response = client.post("/auth/token", data={"username": username, "password": password})
    assert response.status_code == 401
    
    # Register
    response = client.post("/auth/register", json={"username": username, "password": password})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    token = data["access_token"]
    
    # 2. Login
    response = client.post("/auth/token", data={"username": username, "password": password})
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Save Team
    team_payload = {
        "name": "Winning XI",
        "team_data": {"players": [1, 2, 3], "score": 99}
    }
    response = client.post("/teams/", json=team_payload, headers=headers)
    assert response.status_code == 200
    team_data = response.json()
    assert team_data["name"] == "Winning XI"
    team_id = team_data["id"]
    
    # 4. List Teams
    response = client.get("/teams/", headers=headers)
    assert response.status_code == 200
    teams = response.json()
    assert len(teams) >= 1
    assert teams[0]["name"] == "Winning XI"
    
    # 5. Delete Team
    response = client.delete(f"/teams/{team_id}", headers=headers)
    assert response.status_code == 204
    
    # Verify deletion
    response = client.get("/teams/", headers=headers)
    assert len(response.json()) == 0
