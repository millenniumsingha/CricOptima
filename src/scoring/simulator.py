import numpy as np
from typing import Dict, Any, List

class MatchSimulator:
    def __init__(self, iterations: int = 1000, default_std_dev_pct: float = 0.2):
        """
        Initialize simulator.
        :param iterations: Number of Monte Carlo runs.
        :param default_std_dev_pct: Assumed variance (e.g., 0.2 = 20% of predicted points).
        """
        self.iterations = iterations
        self.default_std_dev_pct = default_std_dev_pct

    def simulate_match(self, team_a_data: Dict[str, Any], team_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate match between two teams.
        """
        # Extract predicted points
        points_a_base = team_a_data.get("predicted_points", 0)
        points_b_base = team_b_data.get("predicted_points", 0)
        
        name_a = team_a_data.get("name", "Team A")
        name_b = team_b_data.get("name", "Team B")

        # Simulate distributions (Vectorized for speed)
        # Score = Normal(mean=points, std=points*0.2)
        std_a = max(points_a_base * self.default_std_dev_pct, 5.0) # Min std dev 5 points
        std_b = max(points_b_base * self.default_std_dev_pct, 5.0)
        
        # Sample 1000 outcomes
        scores_a = np.random.normal(points_a_base, std_a, self.iterations)
        scores_b = np.random.normal(points_b_base, std_b, self.iterations)
        
        # Calculate wins
        a_wins = np.sum(scores_a > scores_b)
        b_wins = self.iterations - a_wins  # Ignoring exact ties for simplicity float comparison
        
        win_prob_a = float(a_wins) / self.iterations
        win_prob_b = 1.0 - win_prob_a
        
        return {
            "team_a": {
                "name": name_a,
                "base_points": points_a_base,
                "win_probability": win_prob_a,
                "stats": {
                    "mean": float(np.mean(scores_a)),
                    "min": float(np.min(scores_a)),
                    "max": float(np.max(scores_a)),
                    "std_dev": float(np.std(scores_a))
                }
            },
            "team_b": {
                "name": name_b,
                "base_points": points_b_base,
                "win_probability": win_prob_b,
                "stats": {
                    "mean": float(np.mean(scores_b)),
                    "min": float(np.min(scores_b)),
                    "max": float(np.max(scores_b)),
                    "std_dev": float(np.std(scores_b))
                }
            },
            "simulation_data": {
                "scores_a": scores_a.tolist(), # Send full data for frontend charts
                "scores_b": scores_b.tolist(),
                "iterations": self.iterations
            }
        }
