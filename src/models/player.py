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
