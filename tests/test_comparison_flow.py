from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_comparison_api():
    team_a = {
        "name": "Team A",
        "predicted_points": 1000.0
    }
    team_b = {
        "name": "Team B",
        "predicted_points": 900.0
    }
    
    response = client.post(
        "/teams/compare/",
        json={"team_a": team_a, "team_b": team_b}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["team_a"]["points"] == 1000.0
    assert data["team_b"]["points"] == 900.0
    assert data["point_diff"] == 100.0
    # 1 / (1 + exp(-100/50)) = 1 / (1 + exp(-2)) = 1 / (1 + 0.135) = 0.88
    assert data["win_probability_a"] > 0.8
    assert data["win_probability_b"] < 0.2
