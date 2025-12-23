"""Shared test fixtures for CricOptima."""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.models.player import Player, PlayerPool, PlayerRole, PlayerStats
from src.models.team import TeamConstraints

@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_player_pool() -> PlayerPool:
    """Create sample player pool for testing."""
    players = []
    
    # Create diverse player pool
    for i in range(5):
        players.append(Player(
            id=f"bat_{i}",
            name=f"Batsman {i}",
            team="Team A" if i < 3 else "Team B",
            role=PlayerRole.BATSMAN,
            cost=85 + i * 2,
            stats=PlayerStats(batting_average=30 + i * 5),
            predicted_points=25 + i * 5
        ))
    
    for i in range(5):
        players.append(Player(
            id=f"bowl_{i}",
            name=f"Bowler {i}",
            team="Team A" if i < 2 else "Team B",
            role=PlayerRole.BOWLER,
            cost=80 + i * 2,
            stats=PlayerStats(),
            predicted_points=20 + i * 4
        ))
    
    for i in range(3):
        players.append(Player(
            id=f"ar_{i}",
            name=f"All-Rounder {i}",
            team="Team B",
            role=PlayerRole.ALL_ROUNDER,
            cost=90 + i * 2,
            stats=PlayerStats(batting_average=25),
            predicted_points=30 + i * 5
        ))
    
    for i in range(2):
        players.append(Player(
            id=f"wk_{i}",
            name=f"Wicket-Keeper {i}",
            team="Team A" if i == 0 else "Team B",
            role=PlayerRole.WICKET_KEEPER,
            cost=85 + i * 2,
            stats=PlayerStats(batting_average=28),
            predicted_points=22 + i * 8
        ))
    
    return PlayerPool(players=players)
