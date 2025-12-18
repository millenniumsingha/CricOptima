# CricOptima - Fantasy Cricket Optimizer
## Complete Upgrade Guide for AI Agent Implementation

---

## PROJECT OVERVIEW

**Original Repository:** https://github.com/millenniumsingha/My_Python_FantasyCricket

**New Name:** CricOptima

**Objective:** Transform a basic PyQt fantasy cricket game into a production-ready web application featuring:
1. ML-based player performance predictions
2. Optimal team selection algorithm (budget-constrained optimization)
3. Live/historical cricket data integration
4. FastAPI REST backend
5. Streamlit interactive dashboard
6. Docker containerization

---

## STEP 1: CREATE NEW PROJECT STRUCTURE

```
CricOptima/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .env.example
│
├── src/                          # Core application code
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   │
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   ├── player.py             # Player data models
│   │   ├── team.py               # Team models
│   │   └── match.py              # Match models
│   │
│   ├── scoring/                  # Fantasy scoring logic
│   │   ├── __init__.py
│   │   └── calculator.py         # Points calculation (from original)
│   │
│   ├── optimizer/                # Team optimization
│   │   ├── __init__.py
│   │   └── team_builder.py       # Optimal team selection algorithm
│   │
│   ├── ml/                       # Machine learning
│   │   ├── __init__.py
│   │   ├── features.py           # Feature engineering
│   │   ├── predictor.py          # Performance prediction model
│   │   └── train.py              # Model training script
│   │
│   ├── data/                     # Data layer
│   │   ├── __init__.py
│   │   ├── database.py           # Database connection
│   │   ├── cricket_api.py        # External cricket data API
│   │   └── sample_data.py        # Sample/mock data for demo
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── helpers.py
│
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── players.py            # Player endpoints
│   │   ├── teams.py              # Team endpoints
│   │   └── predictions.py        # ML prediction endpoints
│   └── schemas.py                # Pydantic schemas
│
├── app/                          # Streamlit frontend
│   └── streamlit_app.py          # Main dashboard
│
├── data/                         # Data storage
│   ├── players.json              # Player database
│   ├── historical_stats.csv      # Historical performance data
│   └── sample_matches.json       # Sample match data
│
├── ml_models/                    # Trained ML models
│   └── .gitkeep
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_scoring.py
│   ├── test_optimizer.py
│   ├── test_api.py
│   └── test_predictor.py
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
└── legacy/                       # Original project files
    ├── my_fantasy_cricket.py
    ├── match_evaluation.py
    └── cricket_match.db
```

---

## STEP 2: FILE CONTENTS

### FILE: requirements.txt

```
# Core
python-dotenv>=1.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Data & ML
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0

# API
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
httpx>=0.26.0

# Frontend
streamlit>=1.31.0
plotly>=5.18.0
altair>=5.2.0

# Database
sqlalchemy>=2.0.0
aiosqlite>=0.19.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0

# Optimization
scipy>=1.11.0
ortools>=9.8.0

# Utilities
requests>=2.31.0
python-multipart>=0.0.6
```

---

### FILE: src/__init__.py

```python
"""CricOptima - Fantasy Cricket Optimizer."""
__version__ = "2.0.0"
__author__ = "Millennium Singha"
```

---

### FILE: src/config.py

```python
"""Configuration settings for CricOptima."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ML_MODELS_DIR: Path = PROJECT_ROOT / "ml_models"
    
    # Database
    DATABASE_URL: str = "sqlite:///./cricoptima.db"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Cricket API (optional - for live data)
    CRICKET_API_KEY: Optional[str] = None
    CRICKET_API_URL: str = "https://api.cricapi.com/v1"
    
    # Fantasy Game Settings
    BUDGET_LIMIT: int = 1000
    TEAM_SIZE: int = 11
    MAX_BATSMEN: int = 5
    MAX_BOWLERS: int = 5
    MAX_ALL_ROUNDERS: int = 3
    MAX_WICKET_KEEPERS: int = 1
    MIN_BATSMEN: int = 3
    MIN_BOWLERS: int = 3
    MIN_ALL_ROUNDERS: int = 1
    MIN_WICKET_KEEPERS: int = 1
    
    # ML Settings
    MODEL_PATH: Path = ML_MODELS_DIR / "player_predictor.joblib"
    PREDICTION_FEATURES: list = [
        "recent_avg_runs", "recent_avg_wickets", "strike_rate",
        "economy_rate", "matches_played", "venue_avg", "opposition_avg"
    ]
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(exist_ok=True)
settings.ML_MODELS_DIR.mkdir(exist_ok=True)
```

---

### FILE: src/models/__init__.py

```python
"""Data models for CricOptima."""
from src.models.player import Player, PlayerRole, PlayerStats
from src.models.team import Team, TeamConstraints
from src.models.match import Match, MatchResult

__all__ = [
    "Player", "PlayerRole", "PlayerStats",
    "Team", "TeamConstraints", 
    "Match", "MatchResult"
]
```

---

### FILE: src/models/player.py

```python
"""Player data models."""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class PlayerRole(str, Enum):
    """Player role categories."""
    BATSMAN = "BAT"
    BOWLER = "BOWL"
    ALL_ROUNDER = "AR"
    WICKET_KEEPER = "WK"


class PlayerStats(BaseModel):
    """Historical player statistics."""
    matches_played: int = 0
    total_runs: int = 0
    total_wickets: int = 0
    total_catches: int = 0
    highest_score: int = 0
    best_bowling: str = "0/0"
    batting_average: float = 0.0
    bowling_average: float = 0.0
    strike_rate: float = 0.0
    economy_rate: float = 0.0
    
    # Recent form (last 5 matches)
    recent_runs: List[int] = Field(default_factory=list)
    recent_wickets: List[int] = Field(default_factory=list)
    
    @property
    def recent_avg_runs(self) -> float:
        """Average runs in recent matches."""
        if not self.recent_runs:
            return self.batting_average
        return sum(self.recent_runs) / len(self.recent_runs)
    
    @property
    def recent_avg_wickets(self) -> float:
        """Average wickets in recent matches."""
        if not self.recent_wickets:
            return 0.0
        return sum(self.recent_wickets) / len(self.recent_wickets)


class Player(BaseModel):
    """Cricket player model."""
    id: str
    name: str
    team: str
    role: PlayerRole
    cost: int = Field(ge=0, le=200, description="Cost in fantasy points")
    
    # Statistics
    stats: PlayerStats = Field(default_factory=PlayerStats)
    
    # Predictions (filled by ML model)
    predicted_points: Optional[float] = None
    prediction_confidence: Optional[float] = None
    
    @property
    def value_score(self) -> float:
        """Points per cost ratio (value for money)."""
        if self.cost == 0:
            return 0.0
        predicted = self.predicted_points or self.stats.batting_average
        return predicted / self.cost
    
    class Config:
        use_enum_values = True


class PlayerPool(BaseModel):
    """Collection of available players."""
    players: List[Player]
    last_updated: Optional[str] = None
    
    def get_by_role(self, role: PlayerRole) -> List[Player]:
        """Get players by role."""
        return [p for p in self.players if p.role == role]
    
    def get_by_team(self, team: str) -> List[Player]:
        """Get players by team."""
        return [p for p in self.players if p.team == team]
```

---

### FILE: src/models/team.py

```python
"""Team data models."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, model_validator
from src.models.player import Player, PlayerRole
from src.config import settings


class TeamConstraints(BaseModel):
    """Constraints for team building."""
    budget: int = Field(default=settings.BUDGET_LIMIT)
    team_size: int = Field(default=settings.TEAM_SIZE)
    max_batsmen: int = Field(default=settings.MAX_BATSMEN)
    max_bowlers: int = Field(default=settings.MAX_BOWLERS)
    max_all_rounders: int = Field(default=settings.MAX_ALL_ROUNDERS)
    max_wicket_keepers: int = Field(default=settings.MAX_WICKET_KEEPERS)
    min_batsmen: int = Field(default=settings.MIN_BATSMEN)
    min_bowlers: int = Field(default=settings.MIN_BOWLERS)
    min_all_rounders: int = Field(default=settings.MIN_ALL_ROUNDERS)
    min_wicket_keepers: int = Field(default=settings.MIN_WICKET_KEEPERS)
    max_per_team: int = Field(default=7, description="Max players from one team")


class Team(BaseModel):
    """Fantasy cricket team."""
    id: Optional[str] = None
    name: str
    players: List[Player] = Field(default_factory=list)
    captain_id: Optional[str] = None
    vice_captain_id: Optional[str] = None
    
    @property
    def total_cost(self) -> int:
        """Total cost of the team."""
        return sum(p.cost for p in self.players)
    
    @property
    def predicted_points(self) -> float:
        """Total predicted points for the team."""
        total = 0.0
        for player in self.players:
            points = player.predicted_points or 0
            if player.id == self.captain_id:
                points *= 2  # Captain gets 2x points
            elif player.id == self.vice_captain_id:
                points *= 1.5  # Vice-captain gets 1.5x points
            total += points
        return total
    
    @property
    def role_counts(self) -> Dict[str, int]:
        """Count of players by role."""
        counts = {role.value: 0 for role in PlayerRole}
        for player in self.players:
            counts[player.role] += 1
        return counts
    
    @property
    def team_counts(self) -> Dict[str, int]:
        """Count of players by cricket team."""
        counts: Dict[str, int] = {}
        for player in self.players:
            counts[player.team] = counts.get(player.team, 0) + 1
        return counts
    
    def validate_constraints(self, constraints: TeamConstraints) -> List[str]:
        """Validate team against constraints. Returns list of violations."""
        violations = []
        
        # Team size
        if len(self.players) != constraints.team_size:
            violations.append(
                f"Team must have exactly {constraints.team_size} players, "
                f"has {len(self.players)}"
            )
        
        # Budget
        if self.total_cost > constraints.budget:
            violations.append(
                f"Team cost {self.total_cost} exceeds budget {constraints.budget}"
            )
        
        # Role constraints
        roles = self.role_counts
        if roles.get("BAT", 0) < constraints.min_batsmen:
            violations.append(f"Need at least {constraints.min_batsmen} batsmen")
        if roles.get("BAT", 0) > constraints.max_batsmen:
            violations.append(f"Maximum {constraints.max_batsmen} batsmen allowed")
        if roles.get("BOWL", 0) < constraints.min_bowlers:
            violations.append(f"Need at least {constraints.min_bowlers} bowlers")
        if roles.get("BOWL", 0) > constraints.max_bowlers:
            violations.append(f"Maximum {constraints.max_bowlers} bowlers allowed")
        if roles.get("AR", 0) < constraints.min_all_rounders:
            violations.append(f"Need at least {constraints.min_all_rounders} all-rounder")
        if roles.get("WK", 0) < constraints.min_wicket_keepers:
            violations.append(f"Need at least {constraints.min_wicket_keepers} wicket-keeper")
        
        # Max per team
        for team, count in self.team_counts.items():
            if count > constraints.max_per_team:
                violations.append(
                    f"Maximum {constraints.max_per_team} players from {team}, has {count}"
                )
        
        return violations
    
    def is_valid(self, constraints: TeamConstraints) -> bool:
        """Check if team is valid."""
        return len(self.validate_constraints(constraints)) == 0
```

---

### FILE: src/models/match.py

```python
"""Match data models."""

from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class MatchFormat(str, Enum):
    """Cricket match formats."""
    T20 = "T20"
    ODI = "ODI"
    TEST = "TEST"


class PlayerPerformance(BaseModel):
    """Individual player performance in a match."""
    player_id: str
    player_name: str
    
    # Batting
    runs: int = 0
    balls_faced: int = 0
    fours: int = 0
    sixes: int = 0
    not_out: bool = False
    
    # Bowling
    overs_bowled: float = 0.0
    runs_conceded: int = 0
    wickets: int = 0
    maidens: int = 0
    
    # Fielding
    catches: int = 0
    stumpings: int = 0
    run_outs: int = 0
    
    @property
    def strike_rate(self) -> float:
        """Batting strike rate."""
        if self.balls_faced == 0:
            return 0.0
        return (self.runs / self.balls_faced) * 100
    
    @property
    def economy_rate(self) -> float:
        """Bowling economy rate."""
        if self.overs_bowled == 0:
            return 0.0
        return self.runs_conceded / self.overs_bowled


class MatchResult(BaseModel):
    """Match result and performances."""
    match_id: str
    team1: str
    team2: str
    winner: Optional[str] = None
    format: MatchFormat = MatchFormat.T20
    venue: str
    date: datetime
    performances: List[PlayerPerformance] = Field(default_factory=list)
    
    def get_player_performance(self, player_id: str) -> Optional[PlayerPerformance]:
        """Get performance for a specific player."""
        for perf in self.performances:
            if perf.player_id == player_id:
                return perf
        return None


class Match(BaseModel):
    """Upcoming match for fantasy selection."""
    id: str
    team1: str
    team2: str
    format: MatchFormat = MatchFormat.T20
    venue: str
    date: datetime
    team1_players: List[str] = Field(default_factory=list)
    team2_players: List[str] = Field(default_factory=list)
```

---

### FILE: src/scoring/__init__.py

```python
"""Fantasy scoring module."""
from src.scoring.calculator import FantasyScorer

__all__ = ["FantasyScorer"]
```

---

### FILE: src/scoring/calculator.py

```python
"""
Fantasy Cricket Points Calculator.
Based on original scoring logic from my_fantasy_cricket.py, enhanced and modularized.
"""

from typing import Dict, Optional
from pydantic import BaseModel
from src.models.match import PlayerPerformance, MatchFormat


class ScoringRules(BaseModel):
    """Configurable scoring rules."""
    # Batting
    run_point: float = 0.5
    four_bonus: float = 1.0
    six_bonus: float = 2.0
    fifty_bonus: float = 10.0
    century_bonus: float = 20.0
    duck_penalty: float = -5.0  # For batsmen/all-rounders only
    
    # Strike rate bonuses (T20/ODI)
    sr_above_150_bonus: float = 4.0
    sr_120_to_150_bonus: float = 2.0
    sr_below_70_penalty: float = -2.0
    
    # Bowling
    wicket_point: float = 10.0
    three_wicket_bonus: float = 5.0
    five_wicket_bonus: float = 10.0
    maiden_bonus: float = 2.0
    
    # Economy rate bonuses (T20)
    economy_below_5_bonus: float = 4.0
    economy_5_to_6_bonus: float = 2.0
    economy_above_10_penalty: float = -2.0
    
    # Fielding
    catch_point: float = 10.0
    stumping_point: float = 10.0
    run_out_point: float = 10.0


class FantasyScorer:
    """Calculate fantasy points for player performances."""
    
    def __init__(self, rules: Optional[ScoringRules] = None):
        """Initialize with scoring rules."""
        self.rules = rules or ScoringRules()
    
    def calculate_batting_points(
        self, 
        performance: PlayerPerformance,
        is_batsman: bool = True
    ) -> Dict[str, float]:
        """Calculate batting points breakdown."""
        points = {}
        r = self.rules
        
        # Base runs
        points["runs"] = performance.runs * r.run_point
        
        # Boundary bonuses
        points["fours"] = performance.fours * r.four_bonus
        points["sixes"] = performance.sixes * r.six_bonus
        
        # Milestone bonuses
        if performance.runs >= 100:
            points["century"] = r.century_bonus
        elif performance.runs >= 50:
            points["fifty"] = r.fifty_bonus
        
        # Duck penalty (only for batsmen/all-rounders who faced balls)
        if performance.runs == 0 and performance.balls_faced > 0 and is_batsman:
            if not performance.not_out:
                points["duck"] = r.duck_penalty
        
        # Strike rate bonus/penalty (min 10 balls faced)
        if performance.balls_faced >= 10:
            sr = performance.strike_rate
            if sr > 150:
                points["strike_rate"] = r.sr_above_150_bonus
            elif sr >= 120:
                points["strike_rate"] = r.sr_120_to_150_bonus
            elif sr < 70:
                points["strike_rate"] = r.sr_below_70_penalty
        
        return points
    
    def calculate_bowling_points(
        self, 
        performance: PlayerPerformance
    ) -> Dict[str, float]:
        """Calculate bowling points breakdown."""
        points = {}
        r = self.rules
        
        # Wickets
        points["wickets"] = performance.wickets * r.wicket_point
        
        # Wicket haul bonuses
        if performance.wickets >= 5:
            points["five_wicket_haul"] = r.five_wicket_bonus
        elif performance.wickets >= 3:
            points["three_wicket_haul"] = r.three_wicket_bonus
        
        # Maidens
        points["maidens"] = performance.maidens * r.maiden_bonus
        
        # Economy rate bonus/penalty (min 2 overs bowled)
        if performance.overs_bowled >= 2:
            economy = performance.economy_rate
            if economy < 5:
                points["economy"] = r.economy_below_5_bonus
            elif economy <= 6:
                points["economy"] = r.economy_5_to_6_bonus
            elif economy > 10:
                points["economy"] = r.economy_above_10_penalty
        
        return points
    
    def calculate_fielding_points(
        self, 
        performance: PlayerPerformance
    ) -> Dict[str, float]:
        """Calculate fielding points breakdown."""
        r = self.rules
        return {
            "catches": performance.catches * r.catch_point,
            "stumpings": performance.stumpings * r.stumping_point,
            "run_outs": performance.run_outs * r.run_out_point
        }
    
    def calculate_total_points(
        self, 
        performance: PlayerPerformance,
        player_role: str = "BAT"
    ) -> Dict[str, any]:
        """
        Calculate total fantasy points for a player performance.
        
        Returns breakdown and total.
        """
        is_batsman = player_role in ["BAT", "AR", "WK"]
        
        batting = self.calculate_batting_points(performance, is_batsman)
        bowling = self.calculate_bowling_points(performance)
        fielding = self.calculate_fielding_points(performance)
        
        total = sum(batting.values()) + sum(bowling.values()) + sum(fielding.values())
        
        return {
            "batting": batting,
            "bowling": bowling,
            "fielding": fielding,
            "batting_total": sum(batting.values()),
            "bowling_total": sum(bowling.values()),
            "fielding_total": sum(fielding.values()),
            "total": total
        }


# Convenience function
def calculate_fantasy_points(
    performance: PlayerPerformance, 
    player_role: str = "BAT"
) -> float:
    """Quick function to get total fantasy points."""
    scorer = FantasyScorer()
    result = scorer.calculate_total_points(performance, player_role)
    return result["total"]
```

---

### FILE: src/optimizer/__init__.py

```python
"""Team optimization module."""
from src.optimizer.team_builder import TeamOptimizer, OptimizationResult

__all__ = ["TeamOptimizer", "OptimizationResult"]
```

---

### FILE: src/optimizer/team_builder.py

```python
"""
Optimal Team Builder using constraint optimization.
Solves the fantasy cricket team selection as a variant of the knapsack problem.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel
import numpy as np
from src.models.player import Player, PlayerRole, PlayerPool
from src.models.team import Team, TeamConstraints
from src.config import settings


class OptimizationResult(BaseModel):
    """Result of team optimization."""
    team: Team
    total_predicted_points: float
    total_cost: int
    budget_remaining: int
    optimization_score: float
    suggested_captain: Optional[str] = None
    suggested_vice_captain: Optional[str] = None


class TeamOptimizer:
    """
    Optimizes fantasy cricket team selection.
    Uses a greedy algorithm with constraint satisfaction.
    """
    
    def __init__(self, constraints: Optional[TeamConstraints] = None):
        """Initialize optimizer with constraints."""
        self.constraints = constraints or TeamConstraints()
    
    def optimize(
        self, 
        player_pool: PlayerPool,
        team_name: str = "Optimized XI"
    ) -> OptimizationResult:
        """
        Build optimal team from player pool.
        
        Uses a greedy value-based selection with constraint satisfaction.
        """
        players = player_pool.players
        
        # Sort by value score (predicted points / cost)
        sorted_players = sorted(
            players, 
            key=lambda p: p.value_score, 
            reverse=True
        )
        
        selected: List[Player] = []
        role_counts = {role: 0 for role in PlayerRole}
        team_counts: dict = {}
        total_cost = 0
        
        # Phase 1: Fill minimum requirements
        selected, role_counts, team_counts, total_cost = self._fill_minimums(
            sorted_players, selected, role_counts, team_counts, total_cost
        )
        
        # Phase 2: Fill remaining slots with best value players
        remaining_slots = self.constraints.team_size - len(selected)
        
        for player in sorted_players:
            if remaining_slots <= 0:
                break
                
            if player in selected:
                continue
            
            if self._can_add_player(player, role_counts, team_counts, total_cost):
                selected.append(player)
                role_counts[PlayerRole(player.role)] += 1
                team_counts[player.team] = team_counts.get(player.team, 0) + 1
                total_cost += player.cost
                remaining_slots -= 1
        
        # Build team
        team = Team(name=team_name, players=selected)
        
        # Suggest captain/vice-captain (highest predicted points)
        sorted_by_points = sorted(
            selected,
            key=lambda p: p.predicted_points or 0,
            reverse=True
        )
        
        captain_id = sorted_by_points[0].id if sorted_by_points else None
        vc_id = sorted_by_points[1].id if len(sorted_by_points) > 1 else None
        
        team.captain_id = captain_id
        team.vice_captain_id = vc_id
        
        return OptimizationResult(
            team=team,
            total_predicted_points=team.predicted_points,
            total_cost=total_cost,
            budget_remaining=self.constraints.budget - total_cost,
            optimization_score=team.predicted_points / max(total_cost, 1),
            suggested_captain=captain_id,
            suggested_vice_captain=vc_id
        )
    
    def _fill_minimums(
        self,
        sorted_players: List[Player],
        selected: List[Player],
        role_counts: dict,
        team_counts: dict,
        total_cost: int
    ) -> Tuple[List[Player], dict, dict, int]:
        """Fill minimum role requirements."""
        minimums = {
            PlayerRole.BATSMAN: self.constraints.min_batsmen,
            PlayerRole.BOWLER: self.constraints.min_bowlers,
            PlayerRole.ALL_ROUNDER: self.constraints.min_all_rounders,
            PlayerRole.WICKET_KEEPER: self.constraints.min_wicket_keepers,
        }
        
        for role, min_count in minimums.items():
            role_players = [p for p in sorted_players if p.role == role.value]
            
            for player in role_players:
                if role_counts[role] >= min_count:
                    break
                    
                if player not in selected:
                    # Check team constraint
                    current_team_count = team_counts.get(player.team, 0)
                    if current_team_count >= self.constraints.max_per_team:
                        continue
                    
                    # Check budget
                    if total_cost + player.cost > self.constraints.budget:
                        continue
                    
                    selected.append(player)
                    role_counts[role] += 1
                    team_counts[player.team] = current_team_count + 1
                    total_cost += player.cost
        
        return selected, role_counts, team_counts, total_cost
    
    def _can_add_player(
        self,
        player: Player,
        role_counts: dict,
        team_counts: dict,
        current_cost: int
    ) -> bool:
        """Check if player can be added given constraints."""
        role = PlayerRole(player.role)
        
        # Check role maximum
        max_for_role = {
            PlayerRole.BATSMAN: self.constraints.max_batsmen,
            PlayerRole.BOWLER: self.constraints.max_bowlers,
            PlayerRole.ALL_ROUNDER: self.constraints.max_all_rounders,
            PlayerRole.WICKET_KEEPER: self.constraints.max_wicket_keepers,
        }
        
        if role_counts[role] >= max_for_role[role]:
            return False
        
        # Check team maximum
        if team_counts.get(player.team, 0) >= self.constraints.max_per_team:
            return False
        
        # Check budget
        if current_cost + player.cost > self.constraints.budget:
            return False
        
        return True
    
    def evaluate_team(self, team: Team) -> dict:
        """Evaluate an existing team's optimization metrics."""
        violations = team.validate_constraints(self.constraints)
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "total_cost": team.total_cost,
            "budget_remaining": self.constraints.budget - team.total_cost,
            "predicted_points": team.predicted_points,
            "role_distribution": team.role_counts,
            "team_distribution": team.team_counts,
            "value_score": team.predicted_points / max(team.total_cost, 1)
        }
```

---

### FILE: src/ml/__init__.py

```python
"""Machine learning module for player predictions."""
from src.ml.predictor import PlayerPredictor
from src.ml.features import FeatureEngineer

__all__ = ["PlayerPredictor", "FeatureEngineer"]
```

---

### FILE: src/ml/features.py

```python
"""Feature engineering for ML predictions."""

from typing import List, Dict, Optional
import numpy as np
from src.models.player import Player, PlayerStats


class FeatureEngineer:
    """Extract and engineer features for ML models."""
    
    FEATURE_NAMES = [
        "recent_avg_runs",
        "recent_avg_wickets", 
        "career_batting_avg",
        "career_bowling_avg",
        "strike_rate",
        "economy_rate",
        "matches_played",
        "is_batsman",
        "is_bowler",
        "is_all_rounder",
        "is_wicket_keeper",
        "consistency_score",
        "form_trend"
    ]
    
    def extract_features(self, player: Player) -> np.ndarray:
        """Extract feature vector from player data."""
        stats = player.stats
        
        # Role one-hot encoding
        is_bat = 1.0 if player.role == "BAT" else 0.0
        is_bowl = 1.0 if player.role == "BOWL" else 0.0
        is_ar = 1.0 if player.role == "AR" else 0.0
        is_wk = 1.0 if player.role == "WK" else 0.0
        
        # Consistency score (lower variance in recent scores = more consistent)
        consistency = self._calculate_consistency(stats.recent_runs)
        
        # Form trend (improving, stable, declining)
        form_trend = self._calculate_form_trend(stats.recent_runs)
        
        features = np.array([
            stats.recent_avg_runs,
            stats.recent_avg_wickets,
            stats.batting_average,
            stats.bowling_average if stats.bowling_average > 0 else 50.0,  # Default
            stats.strike_rate,
            stats.economy_rate if stats.economy_rate > 0 else 8.0,  # Default
            float(stats.matches_played),
            is_bat,
            is_bowl,
            is_ar,
            is_wk,
            consistency,
            form_trend
        ])
        
        return features
    
    def extract_features_batch(self, players: List[Player]) -> np.ndarray:
        """Extract features for multiple players."""
        return np.array([self.extract_features(p) for p in players])
    
    def _calculate_consistency(self, recent_scores: List[int]) -> float:
        """Calculate consistency score (0-1, higher is more consistent)."""
        if len(recent_scores) < 2:
            return 0.5
        
        std = np.std(recent_scores)
        mean = np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1
        cv = std / mean  # Coefficient of variation
        
        # Convert to 0-1 scale (lower CV = more consistent)
        return max(0, 1 - min(cv, 1))
    
    def _calculate_form_trend(self, recent_scores: List[int]) -> float:
        """
        Calculate form trend (-1 to 1).
        Positive = improving, Negative = declining
        """
        if len(recent_scores) < 3:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        # Normalize to -1 to 1
        avg = np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1
        normalized_slope = slope / avg
        
        return max(-1, min(1, normalized_slope))
```

---

### FILE: src/ml/predictor.py

```python
"""Player performance prediction model."""

from typing import List, Optional, Tuple
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.models.player import Player
from src.ml.features import FeatureEngineer
from src.config import settings


class PlayerPredictor:
    """
    Predict fantasy points for players using ML.
    Uses Gradient Boosting for robust predictions.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize predictor."""
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
    
    def train(
        self, 
        players: List[Player], 
        actual_points: List[float],
        save_model: bool = True
    ) -> dict:
        """
        Train the prediction model.
        
        Args:
            players: List of players with historical stats
            actual_points: Corresponding actual fantasy points scored
            save_model: Whether to save the trained model
            
        Returns:
            Training metrics
        """
        # Extract features
        X = self.feature_engineer.extract_features_batch(players)
        y = np.array(actual_points)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        # Fit on full data
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Feature importance
        importance = dict(zip(
            FeatureEngineer.FEATURE_NAMES,
            self.model.feature_importances_
        ))
        
        metrics = {
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "feature_importance": importance,
            "n_samples": len(players)
        }
        
        if save_model:
            self.save()
        
        return metrics
    
    def predict(self, player: Player) -> Tuple[float, float]:
        """
        Predict fantasy points for a player.
        
        Returns:
            Tuple of (predicted_points, confidence)
        """
        if not self.is_fitted:
            self.load()
        
        features = self.feature_engineer.extract_features(player)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Estimate confidence based on player's data completeness
        confidence = self._estimate_confidence(player)
        
        return max(0, prediction), confidence
    
    def predict_batch(self, players: List[Player]) -> List[Tuple[float, float]]:
        """Predict for multiple players."""
        return [self.predict(p) for p in players]
    
    def enrich_players_with_predictions(
        self, 
        players: List[Player]
    ) -> List[Player]:
        """Add predictions to player objects."""
        for player in players:
            points, confidence = self.predict(player)
            player.predicted_points = points
            player.prediction_confidence = confidence
        return players
    
    def _estimate_confidence(self, player: Player) -> float:
        """Estimate prediction confidence based on data quality."""
        stats = player.stats
        
        confidence = 0.5  # Base confidence
        
        # More matches = higher confidence
        if stats.matches_played >= 50:
            confidence += 0.2
        elif stats.matches_played >= 20:
            confidence += 0.1
        
        # Recent form data available
        if len(stats.recent_runs) >= 5:
            confidence += 0.15
        elif len(stats.recent_runs) >= 3:
            confidence += 0.1
        
        # Consistency bonus
        if stats.recent_runs:
            cv = np.std(stats.recent_runs) / (np.mean(stats.recent_runs) + 1)
            if cv < 0.5:
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def save(self) -> None:
        """Save model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": FeatureEngineer.FEATURE_NAMES
        }, self.model_path)
    
    def load(self) -> None:
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first."
            )
        
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True


# Singleton instance
_predictor: Optional[PlayerPredictor] = None

def get_predictor() -> PlayerPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PlayerPredictor()
    return _predictor
```

---

### FILE: src/ml/train.py

```python
"""Training script for ML model."""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.ml.predictor import PlayerPredictor
from src.data.sample_data import generate_training_data
from src.config import settings


def train_model(use_sample_data: bool = True) -> dict:
    """
    Train the player prediction model.
    
    Args:
        use_sample_data: If True, use generated sample data for training
        
    Returns:
        Training metrics
    """
    print("=" * 50)
    print("CricOptima - ML Model Training")
    print("=" * 50)
    
    # Get training data
    if use_sample_data:
        print("\nGenerating sample training data...")
        players, actual_points = generate_training_data(n_players=200)
        print(f"Generated {len(players)} training samples")
    else:
        # Load from historical data file
        raise NotImplementedError("Historical data loading not yet implemented")
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    predictor = PlayerPredictor()
    metrics = predictor.train(players, actual_points, save_model=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nCross-Validation R² Score: {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})")
    print(f"Training samples: {metrics['n_samples']}")
    print(f"\nModel saved to: {settings.MODEL_PATH}")
    
    print("\nTop 5 Important Features:")
    importance = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feature, score in importance:
        print(f"  - {feature}: {score:.3f}")
    
    # Save metrics
    metrics_path = settings.MODEL_PATH.with_suffix('.metrics.json')
    metrics['timestamp'] = datetime.now().isoformat()
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CricOptima ML model")
    parser.add_argument(
        "--sample-data", 
        action="store_true", 
        default=True,
        help="Use generated sample data for training"
    )
    
    args = parser.parse_args()
    train_model(use_sample_data=args.sample_data)
```

---

### FILE: src/data/__init__.py

```python
"""Data layer for CricOptima."""
from src.data.sample_data import get_sample_players, generate_training_data

__all__ = ["get_sample_players", "generate_training_data"]
```

---

### FILE: src/data/sample_data.py

```python
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
```

---

### FILE: api/__init__.py

```python
"""FastAPI backend for CricOptima."""
```

---

### FILE: api/schemas.py

```python
"""Pydantic schemas for API."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class PlayerResponse(BaseModel):
    """Player API response."""
    id: str
    name: str
    team: str
    role: str
    cost: int
    predicted_points: Optional[float] = None
    prediction_confidence: Optional[float] = None
    value_score: Optional[float] = None
    
    # Stats summary
    batting_average: float = 0
    bowling_average: float = 0
    strike_rate: float = 0
    economy_rate: float = 0
    recent_form: List[int] = Field(default_factory=list)


class TeamRequest(BaseModel):
    """Request to build a team."""
    player_ids: List[str]
    team_name: str = "My Team"
    captain_id: Optional[str] = None
    vice_captain_id: Optional[str] = None


class TeamResponse(BaseModel):
    """Team API response."""
    name: str
    players: List[PlayerResponse]
    total_cost: int
    budget_remaining: int
    predicted_points: float
    is_valid: bool
    violations: List[str] = Field(default_factory=list)
    captain_id: Optional[str] = None
    vice_captain_id: Optional[str] = None


class OptimizationRequest(BaseModel):
    """Request for team optimization."""
    budget: int = 1000
    team_name: str = "Optimized XI"
    exclude_players: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)


class OptimizationResponse(BaseModel):
    """Optimization result response."""
    team: TeamResponse
    optimization_score: float
    suggested_captain: Optional[str] = None
    suggested_vice_captain: Optional[str] = None


class PredictionResponse(BaseModel):
    """ML prediction response."""
    player_id: str
    player_name: str
    predicted_points: float
    confidence: float
    feature_contributions: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    model_loaded: bool
    players_available: int
```

---

### FILE: api/main.py

```python
"""FastAPI application for CricOptima."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from src import __version__
from src.config import settings
from src.models.player import PlayerPool
from src.models.team import Team, TeamConstraints
from src.optimizer.team_builder import TeamOptimizer
from src.ml.predictor import get_predictor, PlayerPredictor
from src.data.sample_data import get_sample_players
from api.schemas import (
    PlayerResponse, TeamRequest, TeamResponse,
    OptimizationRequest, OptimizationResponse,
    PredictionResponse, HealthResponse
)


# Initialize FastAPI
app = FastAPI(
    title="CricOptima API",
    description="""
    🏏 **Fantasy Cricket Optimizer API**
    
    Build optimal fantasy cricket teams using ML-powered predictions 
    and constraint optimization.
    
    ## Features
    - Player performance predictions using ML
    - Optimal team selection within budget constraints
    - Fantasy points calculation
    - Team validation
    
    ## Quick Start
    1. Get available players: `GET /players`
    2. Get predictions: `GET /predictions`
    3. Build optimal team: `POST /optimize`
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_player_pool: Optional[PlayerPool] = None
_predictor: Optional[PlayerPredictor] = None


def get_player_pool() -> PlayerPool:
    """Get or initialize player pool."""
    global _player_pool
    if _player_pool is None:
        players = get_sample_players()
        
        # Add predictions
        try:
            predictor = get_predictor()
            players = predictor.enrich_players_with_predictions(players)
        except FileNotFoundError:
            pass  # Model not trained yet
        
        _player_pool = PlayerPool(players=players)
    return _player_pool


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    global _predictor
    try:
        _predictor = get_predictor()
        _predictor.load()
        print("✅ ML model loaded successfully")
    except FileNotFoundError:
        print("⚠️ ML model not found. Run training first: python -m src.ml.train")
    
    # Initialize player pool
    get_player_pool()
    print(f"✅ Loaded {len(get_player_pool().players)} players")


@app.get("/", response_model=dict)
async def root():
    """API root with info."""
    return {
        "name": "CricOptima API",
        "version": __version__,
        "description": "Fantasy Cricket Optimizer",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    pool = get_player_pool()
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=_predictor is not None and _predictor.is_fitted,
        players_available=len(pool.players)
    )


@app.get("/players", response_model=List[PlayerResponse])
async def get_players(
    role: Optional[str] = Query(None, description="Filter by role: BAT, BOWL, AR, WK"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    min_cost: int = Query(0, ge=0),
    max_cost: int = Query(200, le=200),
    sort_by: str = Query("predicted_points", description="Sort by: predicted_points, cost, value_score, name")
):
    """Get available players with optional filters."""
    pool = get_player_pool()
    players = pool.players
    
    # Apply filters
    if role:
        players = [p for p in players if p.role == role]
    if team:
        players = [p for p in players if team.lower() in p.team.lower()]
    
    players = [p for p in players if min_cost <= p.cost <= max_cost]
    
    # Sort
    if sort_by == "predicted_points":
        players.sort(key=lambda p: p.predicted_points or 0, reverse=True)
    elif sort_by == "cost":
        players.sort(key=lambda p: p.cost, reverse=True)
    elif sort_by == "value_score":
        players.sort(key=lambda p: p.value_score, reverse=True)
    elif sort_by == "name":
        players.sort(key=lambda p: p.name)
    
    return [
        PlayerResponse(
            id=p.id,
            name=p.name,
            team=p.team,
            role=p.role,
            cost=p.cost,
            predicted_points=p.predicted_points,
            prediction_confidence=p.prediction_confidence,
            value_score=p.value_score,
            batting_average=p.stats.batting_average,
            bowling_average=p.stats.bowling_average,
            strike_rate=p.stats.strike_rate,
            economy_rate=p.stats.economy_rate,
            recent_form=p.stats.recent_runs
        )
        for p in players
    ]


@app.get("/players/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: str):
    """Get single player by ID."""
    pool = get_player_pool()
    for p in pool.players:
        if p.id == player_id:
            return PlayerResponse(
                id=p.id,
                name=p.name,
                team=p.team,
                role=p.role,
                cost=p.cost,
                predicted_points=p.predicted_points,
                prediction_confidence=p.prediction_confidence,
                value_score=p.value_score,
                batting_average=p.stats.batting_average,
                bowling_average=p.stats.bowling_average,
                strike_rate=p.stats.strike_rate,
                economy_rate=p.stats.economy_rate,
                recent_form=p.stats.recent_runs
            )
    raise HTTPException(status_code=404, detail="Player not found")


@app.post("/teams/validate", response_model=TeamResponse)
async def validate_team(request: TeamRequest):
    """Validate a team selection."""
    pool = get_player_pool()
    
    # Find selected players
    selected = []
    for pid in request.player_ids:
        found = next((p for p in pool.players if p.id == pid), None)
        if not found:
            raise HTTPException(status_code=400, detail=f"Player {pid} not found")
        selected.append(found)
    
    team = Team(
        name=request.team_name,
        players=selected,
        captain_id=request.captain_id,
        vice_captain_id=request.vice_captain_id
    )
    
    constraints = TeamConstraints()
    violations = team.validate_constraints(constraints)
    
    return TeamResponse(
        name=team.name,
        players=[
            PlayerResponse(
                id=p.id, name=p.name, team=p.team, role=p.role,
                cost=p.cost, predicted_points=p.predicted_points,
                prediction_confidence=p.prediction_confidence,
                value_score=p.value_score,
                batting_average=p.stats.batting_average,
                bowling_average=p.stats.bowling_average,
                strike_rate=p.stats.strike_rate,
                economy_rate=p.stats.economy_rate,
                recent_form=p.stats.recent_runs
            )
            for p in team.players
        ],
        total_cost=team.total_cost,
        budget_remaining=constraints.budget - team.total_cost,
        predicted_points=team.predicted_points,
        is_valid=len(violations) == 0,
        violations=violations,
        captain_id=team.captain_id,
        vice_captain_id=team.vice_captain_id
    )


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_team(request: OptimizationRequest):
    """Build optimal team using ML predictions and optimization."""
    pool = get_player_pool()
    
    # Filter players
    available = [p for p in pool.players if p.id not in request.exclude_players]
    
    # Apply custom constraints
    constraints = TeamConstraints(budget=request.budget)
    
    # Optimize
    optimizer = TeamOptimizer(constraints)
    result = optimizer.optimize(
        PlayerPool(players=available),
        team_name=request.team_name
    )
    
    team = result.team
    violations = team.validate_constraints(constraints)
    
    return OptimizationResponse(
        team=TeamResponse(
            name=team.name,
            players=[
                PlayerResponse(
                    id=p.id, name=p.name, team=p.team, role=p.role,
                    cost=p.cost, predicted_points=p.predicted_points,
                    prediction_confidence=p.prediction_confidence,
                    value_score=p.value_score,
                    batting_average=p.stats.batting_average,
                    bowling_average=p.stats.bowling_average,
                    strike_rate=p.stats.strike_rate,
                    economy_rate=p.stats.economy_rate,
                    recent_form=p.stats.recent_runs
                )
                for p in team.players
            ],
            total_cost=result.total_cost,
            budget_remaining=result.budget_remaining,
            predicted_points=result.total_predicted_points,
            is_valid=len(violations) == 0,
            violations=violations,
            captain_id=result.suggested_captain,
            vice_captain_id=result.suggested_vice_captain
        ),
        optimization_score=result.optimization_score,
        suggested_captain=result.suggested_captain,
        suggested_vice_captain=result.suggested_vice_captain
    )


@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(
    top_n: int = Query(10, ge=1, le=50, description="Number of top predictions to return")
):
    """Get ML predictions for all players, sorted by predicted points."""
    pool = get_player_pool()
    
    players = sorted(
        pool.players,
        key=lambda p: p.predicted_points or 0,
        reverse=True
    )[:top_n]
    
    return [
        PredictionResponse(
            player_id=p.id,
            player_name=p.name,
            predicted_points=p.predicted_points or 0,
            confidence=p.prediction_confidence or 0.5
        )
        for p in players
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
```

---

### FILE: app/streamlit_app.py

```python
"""Streamlit dashboard for CricOptima."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.player import PlayerPool, PlayerRole
from src.models.team import Team, TeamConstraints
from src.optimizer.team_builder import TeamOptimizer
from src.ml.predictor import get_predictor
from src.data.sample_data import get_sample_players
from src.config import settings

# Page config
st.set_page_config(
    page_title="CricOptima - Fantasy Cricket Optimizer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a5f 0%, #2e7d32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .player-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .captain-badge {
        background-color: #ffd700;
        color: black;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .vc-badge {
        background-color: #c0c0c0;
        color: black;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """Load player data and ML model."""
    players = get_sample_players()
    
    # Try to load predictions
    try:
        predictor = get_predictor()
        predictor.load()
        players = predictor.enrich_players_with_predictions(players)
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        # Set default predictions
        for p in players:
            p.predicted_points = p.stats.batting_average + p.stats.recent_avg_wickets * 10
            p.prediction_confidence = 0.5
    
    return PlayerPool(players=players), model_loaded


def main():
    # Header
    st.markdown('<p class="main-header">🏏 CricOptima</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Fantasy Cricket Team Optimizer</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    player_pool, model_loaded = load_data()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Budget
        budget = st.slider("Budget", 500, 1500, settings.BUDGET_LIMIT, step=50)
        
        st.markdown("---")
        st.header("📊 Constraints")
        
        col1, col2 = st.columns(2)
        with col1:
            min_bat = st.number_input("Min Batsmen", 1, 5, 3)
            min_bowl = st.number_input("Min Bowlers", 1, 5, 3)
        with col2:
            max_bat = st.number_input("Max Batsmen", 3, 6, 5)
            max_bowl = st.number_input("Max Bowlers", 3, 6, 5)
        
        st.markdown("---")
        
        # Model status
        if model_loaded:
            st.success("✅ ML Model Loaded")
        else:
            st.warning("⚠️ ML Model not trained")
            st.caption("Run: `python -m src.ml.train`")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Auto-Optimize", 
        "🏃 Player Pool", 
        "📈 Analytics",
        "ℹ️ About"
    ])
    
    # Tab 1: Auto-Optimize
    with tab1:
        st.header("Build Optimal Team")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            team_name = st.text_input("Team Name", "My Dream XI")
            
            if st.button("🚀 Generate Optimal Team", type="primary", use_container_width=True):
                constraints = TeamConstraints(
                    budget=budget,
                    min_batsmen=min_bat,
                    max_batsmen=max_bat,
                    min_bowlers=min_bowl,
                    max_bowlers=max_bowl
                )
                
                optimizer = TeamOptimizer(constraints)
                
                with st.spinner("Optimizing team selection..."):
                    result = optimizer.optimize(player_pool, team_name)
                
                st.session_state['optimized_team'] = result
                st.success("✅ Team optimized successfully!")
        
        # Display optimized team
        if 'optimized_team' in st.session_state:
            result = st.session_state['optimized_team']
            team = result.team
            
            st.markdown("---")
            st.subheader(f"🏆 {team.name}")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Points", f"{result.total_predicted_points:.1f}")
            m2.metric("Total Cost", f"{result.total_cost}")
            m3.metric("Budget Left", f"{result.budget_remaining}")
            m4.metric("Value Score", f"{result.optimization_score:.3f}")
            
            # Player table
            st.markdown("### Selected Players")
            
            player_data = []
            for p in team.players:
                badges = ""
                if p.id == team.captain_id:
                    badges = "👑 C"
                elif p.id == team.vice_captain_id:
                    badges = "⭐ VC"
                
                player_data.append({
                    "": badges,
                    "Player": p.name,
                    "Team": p.team,
                    "Role": p.role,
                    "Cost": p.cost,
                    "Pred. Pts": f"{p.predicted_points:.1f}" if p.predicted_points else "N/A",
                    "Confidence": f"{p.prediction_confidence:.0%}" if p.prediction_confidence else "N/A"
                })
            
            df = pd.DataFrame(player_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Role distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                role_counts = team.role_counts
                fig_roles = px.pie(
                    values=list(role_counts.values()),
                    names=list(role_counts.keys()),
                    title="Role Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_roles, use_container_width=True)
            
            with col2:
                team_counts = team.team_counts
                fig_teams = px.bar(
                    x=list(team_counts.keys()),
                    y=list(team_counts.values()),
                    title="Team Distribution",
                    labels={"x": "Team", "y": "Players"}
                )
                st.plotly_chart(fig_teams, use_container_width=True)
    
    # Tab 2: Player Pool
    with tab2:
        st.header("Available Players")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            role_filter = st.selectbox(
                "Filter by Role",
                ["All"] + [r.value for r in PlayerRole]
            )
        
        with col2:
            teams = sorted(set(p.team for p in player_pool.players))
            team_filter = st.selectbox("Filter by Team", ["All"] + teams)
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Predicted Points", "Cost", "Value Score", "Name"]
            )
        
        # Apply filters
        players = player_pool.players
        
        if role_filter != "All":
            players = [p for p in players if p.role == role_filter]
        if team_filter != "All":
            players = [p for p in players if p.team == team_filter]
        
        # Sort
        if sort_by == "Predicted Points":
            players = sorted(players, key=lambda p: p.predicted_points or 0, reverse=True)
        elif sort_by == "Cost":
            players = sorted(players, key=lambda p: p.cost, reverse=True)
        elif sort_by == "Value Score":
            players = sorted(players, key=lambda p: p.value_score, reverse=True)
        else:
            players = sorted(players, key=lambda p: p.name)
        
        # Display
        player_data = []
        for p in players:
            player_data.append({
                "Name": p.name,
                "Team": p.team,
                "Role": p.role,
                "Cost": p.cost,
                "Batting Avg": f"{p.stats.batting_average:.1f}",
                "Strike Rate": f"{p.stats.strike_rate:.1f}",
                "Pred. Points": f"{p.predicted_points:.1f}" if p.predicted_points else "N/A",
                "Value": f"{p.value_score:.3f}",
                "Recent Form": str(p.stats.recent_runs)
            })
        
        df = pd.DataFrame(player_data)
        st.dataframe(df, hide_index=True, use_container_width=True, height=500)
    
    # Tab 3: Analytics
    with tab3:
        st.header("Player Analytics")
        
        # Scatter plot: Cost vs Predicted Points
        fig_scatter = px.scatter(
            x=[p.cost for p in player_pool.players],
            y=[p.predicted_points or 0 for p in player_pool.players],
            color=[p.role for p in player_pool.players],
            hover_name=[p.name for p in player_pool.players],
            title="Cost vs Predicted Points by Role",
            labels={"x": "Cost", "y": "Predicted Points", "color": "Role"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Top 10 by Predicted Points")
            top_points = sorted(
                player_pool.players,
                key=lambda p: p.predicted_points or 0,
                reverse=True
            )[:10]
            
            for i, p in enumerate(top_points, 1):
                st.write(f"{i}. **{p.name}** ({p.role}) - {p.predicted_points:.1f} pts")
        
        with col2:
            st.subheader("💰 Top 10 by Value Score")
            top_value = sorted(
                player_pool.players,
                key=lambda p: p.value_score,
                reverse=True
            )[:10]
            
            for i, p in enumerate(top_value, 1):
                st.write(f"{i}. **{p.name}** ({p.role}) - {p.value_score:.3f}")
    
    # Tab 4: About
    with tab4:
        st.header("About CricOptima")
        
        st.markdown("""
        ### 🏏 What is CricOptima?
        
        CricOptima is an AI-powered fantasy cricket team optimizer that helps you 
        build the best possible team within your budget constraints.
        
        ### 🧠 How It Works
        
        1. **ML Predictions**: Uses machine learning to predict player fantasy points
           based on historical performance, recent form, and other features.
        
        2. **Optimization**: Applies constraint optimization algorithms to select
           the best team within budget and role requirements.
        
        3. **Smart Recommendations**: Suggests captain and vice-captain based on
           predicted performance.
        
        ### 📊 Features
        
        - **Auto Team Builder**: Generate optimal team with one click
        - **Player Analytics**: Explore player stats and predictions
        - **Custom Constraints**: Adjust budget and role requirements
        - **Value Analysis**: Find undervalued players with high potential
        
        ### 🛠️ Technology Stack
        
        - **ML Model**: Gradient Boosting for performance prediction
        - **Optimization**: Greedy constraint satisfaction algorithm
        - **Backend**: FastAPI REST API
        - **Frontend**: Streamlit dashboard
        - **Data**: IPL player statistics
        
        ---
        
        *Built with ❤️ for cricket fans*
        """)


if __name__ == "__main__":
    main()
```

---

### FILE: tests/__init__.py

```python
"""Test suite for CricOptima."""
```

---

### FILE: tests/test_scoring.py

```python
"""Tests for fantasy scoring calculator."""

import pytest
from src.scoring.calculator import FantasyScorer, ScoringRules
from src.models.match import PlayerPerformance


class TestFantasyScorer:
    """Test fantasy points calculation."""
    
    @pytest.fixture
    def scorer(self):
        return FantasyScorer()
    
    @pytest.fixture
    def batting_performance(self):
        return PlayerPerformance(
            player_id="test_1",
            player_name="Test Batsman",
            runs=75,
            balls_faced=50,
            fours=8,
            sixes=3,
            not_out=False
        )
    
    @pytest.fixture
    def bowling_performance(self):
        return PlayerPerformance(
            player_id="test_2",
            player_name="Test Bowler",
            overs_bowled=4.0,
            runs_conceded=24,
            wickets=3,
            maidens=1
        )
    
    def test_batting_points_calculation(self, scorer, batting_performance):
        """Test batting points are calculated correctly."""
        points = scorer.calculate_batting_points(batting_performance)
        
        # 75 runs * 0.5 = 37.5
        assert points["runs"] == 37.5
        # 8 fours * 1 = 8
        assert points["fours"] == 8.0
        # 3 sixes * 2 = 6
        assert points["sixes"] == 6.0
        # 50+ bonus
        assert points["fifty"] == 10.0
    
    def test_bowling_points_calculation(self, scorer, bowling_performance):
        """Test bowling points are calculated correctly."""
        points = scorer.calculate_bowling_points(bowling_performance)
        
        # 3 wickets * 10 = 30
        assert points["wickets"] == 30.0
        # 3-wicket haul bonus
        assert points["three_wicket_haul"] == 5.0
        # 1 maiden * 2 = 2
        assert points["maidens"] == 2.0
    
    def test_fielding_points_calculation(self, scorer):
        """Test fielding points are calculated correctly."""
        performance = PlayerPerformance(
            player_id="test_3",
            player_name="Test Fielder",
            catches=2,
            run_outs=1
        )
        
        points = scorer.calculate_fielding_points(performance)
        
        assert points["catches"] == 20.0
        assert points["run_outs"] == 10.0
    
    def test_total_points(self, scorer, batting_performance):
        """Test total points calculation."""
        result = scorer.calculate_total_points(batting_performance, "BAT")
        
        assert "batting" in result
        assert "bowling" in result
        assert "fielding" in result
        assert "total" in result
        assert result["total"] > 0
    
    def test_duck_penalty(self, scorer):
        """Test duck penalty for batsman."""
        performance = PlayerPerformance(
            player_id="test_4",
            player_name="Duck Batsman",
            runs=0,
            balls_faced=5,
            not_out=False
        )
        
        points = scorer.calculate_batting_points(performance, is_batsman=True)
        assert points.get("duck", 0) == -5.0
    
    def test_no_duck_penalty_if_not_out(self, scorer):
        """Test no duck penalty if not out."""
        performance = PlayerPerformance(
            player_id="test_5",
            player_name="Survivor",
            runs=0,
            balls_faced=5,
            not_out=True
        )
        
        points = scorer.calculate_batting_points(performance, is_batsman=True)
        assert points.get("duck", 0) == 0
```

---

### FILE: tests/test_optimizer.py

```python
"""Tests for team optimizer."""

import pytest
from src.optimizer.team_builder import TeamOptimizer, OptimizationResult
from src.models.player import Player, PlayerPool, PlayerRole, PlayerStats
from src.models.team import TeamConstraints


class TestTeamOptimizer:
    """Test team optimization."""
    
    @pytest.fixture
    def sample_players(self) -> PlayerPool:
        """Create sample player pool."""
        players = []
        
        # Create diverse player pool
        for i in range(5):
            players.append(Player(
                id=f"bat_{i}",
                name=f"Batsman {i}",
                team="Team A" if i < 3 else "Team B",
                role=PlayerRole.BATSMAN,
                cost=100 + i * 10,
                stats=PlayerStats(batting_average=30 + i * 5),
                predicted_points=25 + i * 5
            ))
        
        for i in range(5):
            players.append(Player(
                id=f"bowl_{i}",
                name=f"Bowler {i}",
                team="Team A" if i < 2 else "Team B",
                role=PlayerRole.BOWLER,
                cost=90 + i * 10,
                stats=PlayerStats(),
                predicted_points=20 + i * 4
            ))
        
        for i in range(3):
            players.append(Player(
                id=f"ar_{i}",
                name=f"All-Rounder {i}",
                team="Team B",
                role=PlayerRole.ALL_ROUNDER,
                cost=110 + i * 15,
                stats=PlayerStats(batting_average=25),
                predicted_points=30 + i * 5
            ))
        
        for i in range(2):
            players.append(Player(
                id=f"wk_{i}",
                name=f"Wicket-Keeper {i}",
                team="Team A" if i == 0 else "Team B",
                role=PlayerRole.WICKET_KEEPER,
                cost=95 + i * 20,
                stats=PlayerStats(batting_average=28),
                predicted_points=22 + i * 8
            ))
        
        return PlayerPool(players=players)
    
    def test_optimizer_returns_valid_team(self, sample_players):
        """Test optimizer returns a valid team."""
        optimizer = TeamOptimizer()
        result = optimizer.optimize(sample_players)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.team.players) == 11
        assert result.team.is_valid(TeamConstraints())
    
    def test_optimizer_respects_budget(self, sample_players):
        """Test optimizer stays within budget."""
        constraints = TeamConstraints(budget=1000)
        optimizer = TeamOptimizer(constraints)
        result = optimizer.optimize(sample_players)
        
        assert result.total_cost <= 1000
    
    def test_optimizer_meets_role_minimums(self, sample_players):
        """Test optimizer meets minimum role requirements."""
        optimizer = TeamOptimizer()
        result = optimizer.optimize(sample_players)
        
        roles = result.team.role_counts
        constraints = TeamConstraints()
        
        assert roles.get("BAT", 0) >= constraints.min_batsmen
        assert roles.get("BOWL", 0) >= constraints.min_bowlers
        assert roles.get("AR", 0) >= constraints.min_all_rounders
        assert roles.get("WK", 0) >= constraints.min_wicket_keepers
    
    def test_optimizer_suggests_captain(self, sample_players):
        """Test optimizer suggests captain and vice-captain."""
        optimizer = TeamOptimizer()
        result = optimizer.optimize(sample_players)
        
        assert result.suggested_captain is not None
        assert result.suggested_vice_captain is not None
        assert result.suggested_captain != result.suggested_vice_captain
    
    def test_optimizer_with_tight_budget(self, sample_players):
        """Test optimizer with very tight budget."""
        constraints = TeamConstraints(budget=800)
        optimizer = TeamOptimizer(constraints)
        result = optimizer.optimize(sample_players)
        
        # Should still produce valid team or handle gracefully
        assert result.total_cost <= 800
```

---

### FILE: tests/test_api.py

```python
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
```

---

### FILE: Dockerfile

```dockerfile
# CricOptima - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p ml_models data

# Expose ports
EXPOSE 8000 8501

# Default: API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### FILE: docker-compose.yml

```yaml
version: '3.8'

services:
  # FastAPI Backend
  api:
    build: .
    container_name: cricoptima-api
    ports:
      - "8000:8000"
    volumes:
      - ./ml_models:/app/ml_models
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Streamlit Dashboard
  streamlit:
    build: .
    container_name: cricoptima-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./ml_models:/app/ml_models
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
    command: streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      - api

  # Training service (run once)
  train:
    build: .
    container_name: cricoptima-train
    volumes:
      - ./ml_models:/app/ml_models
    environment:
      - PYTHONPATH=/app
    command: python -m src.ml.train
    profiles:
      - training
```

---

### FILE: .gitignore

```gitignore
# Byte-compiled
__pycache__/
*.py[cod]
*$py.class

# Distribution
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
.env

# IDE
.idea/
.vscode/
*.swp

# Jupyter
.ipynb_checkpoints/

# ML Models (large files)
ml_models/*.joblib
ml_models/*.pkl
!ml_models/.gitkeep

# Database
*.db
*.sqlite3

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temp
*.tmp
*.temp
```

---

### FILE: .env.example

```env
# CricOptima Configuration

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Database
DATABASE_URL=sqlite:///./cricoptima.db

# Cricket API (optional - for live data)
CRICKET_API_KEY=your_api_key_here
CRICKET_API_URL=https://api.cricapi.com/v1

# Fantasy Settings
BUDGET_LIMIT=1000
TEAM_SIZE=11
```

---

### FILE: README.md

```markdown
# 🏏 CricOptima

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

**AI-Powered Fantasy Cricket Team Optimizer**

Build optimal fantasy cricket teams using machine learning predictions and constraint optimization algorithms.

![CricOptima Dashboard](images/fantasy_cricket1.jpg)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Predictions** | Gradient Boosting model predicts player fantasy points |
| 🎯 **Smart Optimization** | Builds best team within budget and role constraints |
| 📊 **Analytics Dashboard** | Explore player stats, value scores, and predictions |
| 🔌 **REST API** | Full-featured FastAPI backend with Swagger docs |
| 🐳 **Docker Ready** | One-command deployment with docker-compose |

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/millenniumsingha/CricOptima.git
cd CricOptima

# Train ML model and start services
docker-compose --profile training up train
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API Docs:  http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train ML model
python -m src.ml.train

# Start API (terminal 1)
uvicorn api.main:app --reload

# Start Dashboard (terminal 2)
streamlit run app/streamlit_app.py
```

## 📊 How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Player Data   │────▶│   ML Predictor  │────▶│   Predictions   │
│   (Stats/Form)  │     │ (Gradient Boost)│     │  (Points/Conf)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │    Optimizer    │◀─────────────┘
                        │  (Constraints)  │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   Optimal XI    │
                        │ (Best Team)     │
                        └─────────────────┘
```

### ML Features Used

- Recent batting/bowling averages
- Strike rate & economy rate
- Form trend (improving/declining)
- Consistency score
- Matches played (experience)

### Optimization Constraints

- Budget limit (default: 1000 points)
- Team size: 11 players
- Min 3 batsmen, 3 bowlers, 1 all-rounder, 1 wicket-keeper
- Max 7 players from same team

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/players` | List all players with filters |
| GET | `/players/{id}` | Get single player details |
| POST | `/optimize` | Build optimal team |
| GET | `/predictions` | Get ML predictions |
| POST | `/teams/validate` | Validate team selection |
| GET | `/health` | Health check |

### Example: Get Optimal Team

```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{"budget": 1000, "team_name": "My Dream XI"}'
```

## 📁 Project Structure

```
CricOptima/
├── src/
│   ├── models/          # Data models (Player, Team, Match)
│   ├── scoring/         # Fantasy points calculator
│   ├── optimizer/       # Team optimization algorithm
│   ├── ml/              # ML prediction model
│   └── data/            # Data layer & sample data
├── api/                 # FastAPI backend
├── app/                 # Streamlit dashboard
├── tests/               # Test suite
├── ml_models/           # Trained models
└── legacy/              # Original project files
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=api
```

## 🎯 Fantasy Scoring Rules

| Category | Points |
|----------|--------|
| Run | 0.5 |
| Four | +1 bonus |
| Six | +2 bonus |
| 50 runs | +10 bonus |
| 100 runs | +20 bonus |
| Wicket | 10 |
| 3-wicket haul | +5 bonus |
| 5-wicket haul | +10 bonus |
| Catch/Stumping/Run-out | 10 |

## 🚧 Roadmap

- [ ] Live cricket data integration (CricAPI)
- [ ] User authentication & team saving
- [ ] Head-to-head matchup predictions
- [ ] Mobile-responsive design
- [ ] Historical match simulation

## 📜 License

MIT License

## 🙏 Acknowledgments

- Original project from Internshala Python Training
- IPL teams and player data
- scikit-learn, FastAPI, and Streamlit communities

---

*Built with ❤️ for cricket fans and data enthusiasts*
```

---

## STEP 3: IMPLEMENTATION INSTRUCTIONS

### PHASE 1: Setup
```bash
# Rename repository on GitHub: My_Python_FantasyCricket → CricOptima

# Clone and setup
git clone https://github.com/millenniumsingha/CricOptima.git
cd CricOptima

# Create directory structure
mkdir -p src/models src/scoring src/optimizer src/ml src/data src/utils
mkdir -p api/routes app tests notebooks data ml_models legacy

# Move original files to legacy
mv my_fantasy_cricket.py legacy/
mv match_evaluation.py legacy/
mv cricket_match.db legacy/
```

### PHASE 2: Create All Files
Create each file from Step 2 with exact contents.

### PHASE 3: Train & Test
```bash
# Install dependencies
pip install -r requirements.txt

# Train ML model
python -m src.ml.train

# Run tests
pytest tests/ -v

# Start API
uvicorn api.main:app --reload --port 8000

# Start Streamlit (new terminal)
streamlit run app/streamlit_app.py
```

### PHASE 4: Verify
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- All tests pass
- ML model achieves reasonable R² score

### PHASE 5: Commit
```bash
git add .
git commit -m "Transform to CricOptima - AI-powered fantasy cricket optimizer

Major changes:
- Added ML player performance prediction (Gradient Boosting)
- Added constraint optimization for team building
- Added FastAPI REST API with full documentation
- Added Streamlit interactive dashboard
- Modernized project structure
- Added comprehensive test suite
- Added Docker containerization
- Preserved original scoring logic

Features:
- Predict fantasy points using player stats and form
- Auto-build optimal team within budget constraints
- Player analytics and value scoring
- Captain/vice-captain recommendations"

git push origin master
```

---

## VERIFICATION CHECKLIST

- [ ] All directories created
- [ ] All Python files created with correct content
- [ ] requirements.txt has all dependencies
- [ ] ML model trains successfully
- [ ] All tests pass
- [ ] API starts at :8000/docs
- [ ] Streamlit loads at :8501
- [ ] Docker build succeeds
- [ ] README renders correctly
- [ ] Original files preserved in legacy/
