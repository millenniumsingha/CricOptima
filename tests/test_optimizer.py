"""Tests for team optimizer."""

import pytest
from src.optimizer.team_builder import TeamOptimizer, OptimizationResult
from src.models.player import Player, PlayerPool, PlayerRole, PlayerStats
from src.models.team import TeamConstraints


class TestTeamOptimizer:
    """Test team optimization."""
    

    
    def test_optimizer_returns_valid_team(self, sample_player_pool):
        """Test optimizer returns a valid team."""
        optimizer = TeamOptimizer()
        result = optimizer.optimize(sample_player_pool)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.team.players) == 11
        assert result.team.is_valid(TeamConstraints())
    
    def test_optimizer_respects_budget(self, sample_player_pool):
        """Test optimizer stays within budget."""
        constraints = TeamConstraints(budget=1000)
        optimizer = TeamOptimizer(constraints)
        result = optimizer.optimize(sample_player_pool)
        
        assert result.total_cost <= 1000
    
    def test_optimizer_meets_role_minimums(self, sample_player_pool):
        """Test optimizer meets minimum role requirements."""
        optimizer = TeamOptimizer()
        result = optimizer.optimize(sample_player_pool)
        
        roles = result.team.role_counts
        constraints = TeamConstraints()
        
        assert roles.get("BAT", 0) >= constraints.min_batsmen
        assert roles.get("BOWL", 0) >= constraints.min_bowlers
        assert roles.get("AR", 0) >= constraints.min_all_rounders
        assert roles.get("WK", 0) >= constraints.min_wicket_keepers
    
    def test_optimizer_suggests_captain(self, sample_player_pool):
        """Test optimizer suggests captain and vice-captain."""
        optimizer = TeamOptimizer()
        result = optimizer.optimize(sample_player_pool)
        
        assert result.suggested_captain is not None
        assert result.suggested_vice_captain is not None
        assert result.suggested_captain != result.suggested_vice_captain
    
    def test_optimizer_with_tight_budget(self, sample_player_pool):
        """Test optimizer with very tight budget."""
        constraints = TeamConstraints(budget=800)
        optimizer = TeamOptimizer(constraints)
        result = optimizer.optimize(sample_player_pool)
        
        # Should still produce valid team or handle gracefully
        assert result.total_cost <= 800
