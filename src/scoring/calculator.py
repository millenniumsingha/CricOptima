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
