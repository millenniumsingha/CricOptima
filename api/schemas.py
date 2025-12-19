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
