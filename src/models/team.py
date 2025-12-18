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
