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
