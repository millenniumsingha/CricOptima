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
                cost=85 + i * 2,  # Reduced cost
                stats=PlayerStats(batting_average=30 + i * 5),
                predicted_points=25 + i * 5
            ))
        
        for i in range(5):
            players.append(Player(
                id=f"bowl_{i}",
                name=f"Bowler {i}",
                team="Team A" if i < 2 else "Team B",
                role=PlayerRole.BOWLER,
                cost=80 + i * 2,  # Reduced cost
                stats=PlayerStats(),
                predicted_points=20 + i * 4
            ))
        
        for i in range(3):
            players.append(Player(
                id=f"ar_{i}",
                name=f"All-Rounder {i}",
                team="Team B",
                role=PlayerRole.ALL_ROUNDER,
                cost=90 + i * 2,  # Reduced cost
                stats=PlayerStats(batting_average=25),
                predicted_points=30 + i * 5
            ))
        
        for i in range(2):
            players.append(Player(
                id=f"wk_{i}",
                name=f"Wicket-Keeper {i}",
                team="Team A" if i == 0 else "Team B",
                role=PlayerRole.WICKET_KEEPER,
                cost=85 + i * 2,  # Reduced cost
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
