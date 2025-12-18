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
