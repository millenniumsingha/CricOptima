"""Sample data for demo and testing."""

import random
from typing import List, Tuple
import numpy as np
from src.models.player import Player, PlayerRole, PlayerStats


# Real IPL team names and sample players
IPL_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings", 
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Punjab Kings",
    "Sunrisers Hyderabad"
]

SAMPLE_PLAYERS_DATA = [
    # Format: (name, team_idx, role, cost, avg_runs, avg_wickets, sr, economy)
    # Batsmen
    ("Virat Kohli", 2, "BAT", 180, 45.5, 0, 135.2, 0),
    ("Rohit Sharma", 0, "BAT", 175, 42.3, 0, 132.5, 0),
    ("KL Rahul", 6, "BAT", 170, 47.8, 0, 138.4, 0),
    ("Shubman Gill", 3, "BAT", 155, 38.2, 0, 128.6, 0),
    ("Faf du Plessis", 2, "BAT", 145, 35.4, 0, 130.2, 0),
    ("David Warner", 4, "BAT", 160, 41.2, 0, 142.3, 0),
    ("Quinton de Kock", 0, "BAT", 150, 36.8, 0, 140.1, 0),
    ("Ruturaj Gaikwad", 1, "BAT", 140, 34.5, 0, 125.8, 0),
    ("Yashasvi Jaiswal", 5, "BAT", 135, 32.1, 0, 145.6, 0),
    ("Shikhar Dhawan", 6, "BAT", 130, 33.4, 0, 126.3, 0),
    
    # Bowlers
    ("Jasprit Bumrah", 0, "BOWL", 175, 5, 1.8, 0, 6.2),
    ("Mohammed Shami", 3, "BOWL", 155, 3, 1.5, 0, 7.8),
    ("Yuzvendra Chahal", 5, "BOWL", 145, 2, 1.6, 0, 7.2),
    ("Rashid Khan", 3, "BOWL", 170, 8, 1.7, 0, 6.5),
    ("Kuldeep Yadav", 4, "BOWL", 140, 4, 1.4, 0, 7.5),
    ("Arshdeep Singh", 6, "BOWL", 130, 2, 1.3, 0, 8.2),
    ("Mohammed Siraj", 2, "BOWL", 145, 3, 1.4, 0, 8.0),
    ("Trent Boult", 5, "BOWL", 150, 4, 1.5, 0, 7.6),
    ("Kagiso Rabada", 6, "BOWL", 160, 3, 1.6, 0, 7.4),
    ("Bhuvneshwar Kumar", 7, "BOWL", 135, 5, 1.2, 0, 7.0),
    
    # All-rounders
    ("Hardik Pandya", 0, "AR", 180, 28.5, 1.2, 145.2, 8.5),
    ("Ravindra Jadeja", 1, "AR", 175, 22.3, 1.5, 128.4, 6.8),
    ("Andre Russell", 3, "AR", 185, 32.1, 1.0, 175.3, 9.2),
    ("Glenn Maxwell", 2, "AR", 165, 25.6, 0.8, 155.6, 8.0),
    ("Marcus Stoinis", 4, "AR", 145, 24.2, 0.7, 138.2, 8.8),
    ("Axar Patel", 4, "AR", 140, 18.5, 1.3, 125.4, 7.2),
    ("Shardul Thakur", 1, "AR", 125, 15.2, 1.1, 142.1, 9.0),
    ("Washington Sundar", 7, "AR", 120, 16.8, 0.9, 118.5, 7.0),
    
    # Wicket-keepers
    ("MS Dhoni", 1, "WK", 165, 28.4, 0, 135.6, 0),
    ("Rishabh Pant", 4, "WK", 170, 35.2, 0, 148.3, 0),
    ("Sanju Samson", 5, "WK", 155, 32.6, 0, 142.8, 0),
    ("Ishan Kishan", 0, "WK", 145, 28.8, 0, 138.5, 0),
    ("Jos Buttler", 5, "WK", 175, 42.3, 0, 152.4, 0),
    ("Dinesh Karthik", 2, "WK", 120, 22.5, 0, 145.2, 0),
]


def get_sample_players() -> List[Player]:
    """Get sample player pool for demo."""
    players = []
    
    for i, (name, team_idx, role, cost, avg_runs, avg_wkts, sr, eco) in enumerate(SAMPLE_PLAYERS_DATA):
        # Generate realistic recent form
        if role == "BAT" or role == "WK":
            base_runs = avg_runs
            recent_runs = [max(0, int(base_runs + random.gauss(0, 15))) for _ in range(5)]
            recent_wickets = [0] * 5
        elif role == "BOWL":
            recent_runs = [random.randint(0, 15) for _ in range(5)]
            recent_wickets = [max(0, int(avg_wkts + random.gauss(0, 0.8))) for _ in range(5)]
        else:  # AR
            recent_runs = [max(0, int(avg_runs + random.gauss(0, 12))) for _ in range(5)]
            recent_wickets = [max(0, int(avg_wkts + random.gauss(0, 0.6))) for _ in range(5)]
        
        stats = PlayerStats(
            matches_played=random.randint(30, 150),
            total_runs=int(avg_runs * random.randint(50, 120)),
            total_wickets=int(avg_wkts * random.randint(30, 80)) if avg_wkts > 0 else 0,
            total_catches=random.randint(10, 50),
            highest_score=int(avg_runs * random.uniform(2, 3.5)),
            batting_average=avg_runs,
            bowling_average=25.0 / avg_wkts if avg_wkts > 0 else 0,
            strike_rate=sr if sr > 0 else random.uniform(120, 145),
            economy_rate=eco if eco > 0 else 0,
            recent_runs=recent_runs,
            recent_wickets=recent_wickets
        )
        
        player = Player(
            id=f"player_{i:03d}",
            name=name,
            team=IPL_TEAMS[team_idx],
            role=PlayerRole(role),
            cost=cost,
            stats=stats
        )
        players.append(player)
    
    return players


def generate_training_data(n_players: int = 200) -> Tuple[List[Player], List[float]]:
    """
    Generate synthetic training data for ML model.
    
    Creates realistic player profiles and corresponding fantasy points.
    """
    players = []
    actual_points = []
    
    roles = list(PlayerRole)
    
    for i in range(n_players):
        role = random.choice(roles)
        
        # Generate stats based on role
        if role == PlayerRole.BATSMAN:
            avg_runs = random.gauss(32, 10)
            avg_wickets = 0
            sr = random.gauss(130, 15)
            eco = 0
        elif role == PlayerRole.BOWLER:
            avg_runs = random.gauss(8, 5)
            avg_wickets = random.gauss(1.5, 0.5)
            sr = random.gauss(100, 20)
            eco = random.gauss(7.5, 1.2)
        elif role == PlayerRole.ALL_ROUNDER:
            avg_runs = random.gauss(22, 8)
            avg_wickets = random.gauss(1.0, 0.4)
            sr = random.gauss(135, 15)
            eco = random.gauss(8.2, 1.0)
        else:  # WK
            avg_runs = random.gauss(28, 8)
            avg_wickets = 0
            sr = random.gauss(132, 12)
            eco = 0
        
        recent_runs = [max(0, int(avg_runs + random.gauss(0, 12))) for _ in range(5)]
        recent_wickets = [max(0, int(avg_wickets + random.gauss(0, 0.5))) for _ in range(5)]
        
        stats = PlayerStats(
            matches_played=random.randint(20, 120),
            batting_average=max(0, avg_runs),
            bowling_average=25.0 / max(avg_wickets, 0.1) if avg_wickets > 0 else 0,
            strike_rate=max(80, sr),
            economy_rate=max(4, eco) if eco > 0 else 0,
            recent_runs=recent_runs,
            recent_wickets=recent_wickets
        )
        
        player = Player(
            id=f"train_{i:04d}",
            name=f"Player {i}",
            team=random.choice(IPL_TEAMS),
            role=role,
            cost=random.randint(80, 180),
            stats=stats
        )
        players.append(player)
        
        # Calculate "actual" fantasy points with some noise
        base_batting = avg_runs * 0.5 + (sr - 100) * 0.1
        base_bowling = avg_wickets * 15 + (8 - eco) * 2 if eco > 0 else 0
        base_fielding = random.gauss(5, 3)
        
        # Form bonus
        form_bonus = (sum(recent_runs) / 5 - avg_runs) * 0.3
        
        points = max(0, base_batting + base_bowling + base_fielding + form_bonus + random.gauss(0, 8))
        actual_points.append(points)
    
    return players, actual_points
