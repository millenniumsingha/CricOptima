from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_simulation_api():
    team_a = {
        "name": "Team A",
        "predicted_points": 1000.0
    }
    team_b = {
        "name": "Team B",
        "predicted_points": 900.0
    }
    
    response = client.post(
        "/teams/simulate/",
        json={"team_a": team_a, "team_b": team_b, "iterations": 100}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "team_a" in data
    assert "team_b" in data
    assert "simulation_data" in data
    
    # Check stats
    stats_a = data["team_a"]["stats"]
    assert "mean" in stats_a
    assert "std_dev" in stats_a
    
    # Check iterations
    assert len(data["simulation_data"]["scores_a"]) == 100
    assert data["simulation_data"]["iterations"] == 100
    
    # Check win prob (A should verify win most times given 100pt diff)
    assert data["team_a"]["win_probability"] > 0.6
