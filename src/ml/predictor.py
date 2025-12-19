"""Player performance prediction model."""

from typing import List, Optional, Tuple
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.models.player import Player
from src.ml.features import FeatureEngineer
from src.config import settings


class PlayerPredictor:
    """
    Predict fantasy points for players using ML.
    Uses Gradient Boosting for robust predictions.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize predictor."""
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
    
    def train(
        self, 
        players: List[Player], 
        actual_points: List[float],
        save_model: bool = True
    ) -> dict:
        """
        Train the prediction model.
        
        Args:
            players: List of players with historical stats
            actual_points: Corresponding actual fantasy points scored
            save_model: Whether to save the trained model
            
        Returns:
            Training metrics
        """
        # Extract features
        X = self.feature_engineer.extract_features_batch(players)
        y = np.array(actual_points)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        # Fit on full data
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Feature importance
        importance = dict(zip(
            FeatureEngineer.FEATURE_NAMES,
            self.model.feature_importances_
        ))
        
        metrics = {
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "feature_importance": importance,
            "n_samples": len(players)
        }
        
        if save_model:
            self.save()
        
        return metrics
    
    def predict(self, player: Player) -> Tuple[float, float]:
        """
        Predict fantasy points for a player.
        
        Returns:
            Tuple of (predicted_points, confidence)
        """
        if not self.is_fitted:
            self.load()
        
        features = self.feature_engineer.extract_features(player)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Estimate confidence based on player's data completeness
        confidence = self._estimate_confidence(player)
        
        return max(0, prediction), confidence
    
    def predict_batch(self, players: List[Player]) -> List[Tuple[float, float]]:
        """Predict for multiple players."""
        return [self.predict(p) for p in players]
    
    def enrich_players_with_predictions(
        self, 
        players: List[Player]
    ) -> List[Player]:
        """Add predictions to player objects."""
        for player in players:
            points, confidence = self.predict(player)
            player.predicted_points = points
            player.prediction_confidence = confidence
        return players
    
    def _estimate_confidence(self, player: Player) -> float:
        """Estimate prediction confidence based on data quality."""
        stats = player.stats
        
        confidence = 0.5  # Base confidence
        
        # More matches = higher confidence
        if stats.matches_played >= 50:
            confidence += 0.2
        elif stats.matches_played >= 20:
            confidence += 0.1
        
        # Recent form data available
        if len(stats.recent_runs) >= 5:
            confidence += 0.15
        elif len(stats.recent_runs) >= 3:
            confidence += 0.1
        
        # Consistency bonus
        if stats.recent_runs:
            cv = np.std(stats.recent_runs) / (np.mean(stats.recent_runs) + 1)
            if cv < 0.5:
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def save(self) -> None:
        """Save model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": FeatureEngineer.FEATURE_NAMES
        }, self.model_path)
    
    def load(self) -> None:
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first."
            )
        
        data = joblib.load(self.model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True


def get_predictor() -> PlayerPredictor:
    """Get predictor instance."""
    return PlayerPredictor()
