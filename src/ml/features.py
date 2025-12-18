"""Feature engineering for ML predictions."""

from typing import List, Dict, Optional
import numpy as np
from src.models.player import Player, PlayerStats


class FeatureEngineer:
    """Extract and engineer features for ML models."""
    
    FEATURE_NAMES = [
        "recent_avg_runs",
        "recent_avg_wickets", 
        "career_batting_avg",
        "career_bowling_avg",
        "strike_rate",
        "economy_rate",
        "matches_played",
        "is_batsman",
        "is_bowler",
        "is_all_rounder",
        "is_wicket_keeper",
        "consistency_score",
        "form_trend"
    ]
    
    def extract_features(self, player: Player) -> np.ndarray:
        """Extract feature vector from player data."""
        stats = player.stats
        
        # Role one-hot encoding
        is_bat = 1.0 if player.role == "BAT" else 0.0
        is_bowl = 1.0 if player.role == "BOWL" else 0.0
        is_ar = 1.0 if player.role == "AR" else 0.0
        is_wk = 1.0 if player.role == "WK" else 0.0
        
        # Consistency score (lower variance in recent scores = more consistent)
        consistency = self._calculate_consistency(stats.recent_runs)
        
        # Form trend (improving, stable, declining)
        form_trend = self._calculate_form_trend(stats.recent_runs)
        
        features = np.array([
            stats.recent_avg_runs,
            stats.recent_avg_wickets,
            stats.batting_average,
            stats.bowling_average if stats.bowling_average > 0 else 50.0,  # Default
            stats.strike_rate,
            stats.economy_rate if stats.economy_rate > 0 else 8.0,  # Default
            float(stats.matches_played),
            is_bat,
            is_bowl,
            is_ar,
            is_wk,
            consistency,
            form_trend
        ])
        
        return features
    
    def extract_features_batch(self, players: List[Player]) -> np.ndarray:
        """Extract features for multiple players."""
        return np.array([self.extract_features(p) for p in players])
    
    def _calculate_consistency(self, recent_scores: List[int]) -> float:
        """Calculate consistency score (0-1, higher is more consistent)."""
        if len(recent_scores) < 2:
            return 0.5
        
        std = np.std(recent_scores)
        mean = np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1
        cv = std / mean  # Coefficient of variation
        
        # Convert to 0-1 scale (lower CV = more consistent)
        return max(0, 1 - min(cv, 1))
    
    def _calculate_form_trend(self, recent_scores: List[int]) -> float:
        """
        Calculate form trend (-1 to 1).
        Positive = improving, Negative = declining
        """
        if len(recent_scores) < 3:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        
        # Normalize to -1 to 1
        avg = np.mean(recent_scores) if np.mean(recent_scores) > 0 else 1
        normalized_slope = slope / avg
        
        return max(-1, min(1, normalized_slope))
