"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health endpoint."""
    
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_has_required_fields(self, client):
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "players_available" in data


class TestPlayersEndpoint:
    """Test players endpoint."""
    
    def test_get_players(self, client):
        response = client.get("/players")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_filter_by_role(self, client):
        response = client.get("/players?role=BAT")
        assert response.status_code == 200
        
        players = response.json()
        for p in players:
            assert p["role"] == "BAT"
    
    def test_get_single_player(self, client):
        # First get a valid player ID
        response = client.get("/players")
        players = response.json()
        
        if players:
            player_id = players[0]["id"]
            response = client.get(f"/players/{player_id}")
            assert response.status_code == 200
    
    def test_player_not_found(self, client):
        response = client.get("/players/nonexistent_id")
        assert response.status_code == 404


class TestOptimizeEndpoint:
    """Test optimization endpoint."""
    
    def test_optimize_returns_team(self, client):
        response = client.post("/optimize", json={
            "budget": 1000,
            "team_name": "Test XI"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "team" in data
        assert "optimization_score" in data
        assert len(data["team"]["players"]) == 11
    
    def test_optimize_respects_budget(self, client):
        response = client.post("/optimize", json={
            "budget": 800,
            "team_name": "Budget XI"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["team"]["total_cost"] <= 800


class TestPredictionsEndpoint:
    """Test predictions endpoint."""
    
    def test_get_predictions(self, client):
        response = client.get("/predictions")
        assert response.status_code == 200
        
        predictions = response.json()
        assert isinstance(predictions, list)
        assert len(predictions) <= 10  # Default top_n
    
    def test_predictions_limit(self, client):
        response = client.get("/predictions?top_n=5")
        assert response.status_code == 200
        
        predictions = response.json()
        assert len(predictions) <= 5
