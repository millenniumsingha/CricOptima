import math
from typing import Dict, Any

class TeamComparator:
    def __init__(self, scaling_factor: float = 50.0):
        """
        Initialize comparator.
        :param scaling_factor: Controls how much point difference affects win prob.
                               50 points diff ~= 73% win probability.
        """
        self.scaling_factor = scaling_factor

    def calculate_win_probability(self, points_a: float, points_b: float) -> float:
        """Calculate win probability for Team A vs Team B."""
        diff = points_a - points_b
        try:
            return 1 / (1 + math.exp(-diff / self.scaling_factor))
        except OverflowError:
            return 1.0 if diff > 0 else 0.0

    def compare_teams(self, team_a_data: Dict[str, Any], team_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two teams based on their data (dictionaries)."""
        points_a = team_a_data.get("predicted_points", 0)
        points_b = team_b_data.get("predicted_points", 0)
        
        win_prob = self.calculate_win_probability(points_a, points_b)
        
        return {
            "team_a": {
                "name": team_a_data.get("name", "Team A"),
                "points": points_a
            },
            "team_b": {
                "name": team_b_data.get("name", "Team B"),
                "points": points_b
            },
            "win_probability_a": win_prob,
            "win_probability_b": 1 - win_prob,
            "point_diff": points_a - points_b
        }
