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
