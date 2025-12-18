"""Configuration settings for CricOptima."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ML_MODELS_DIR: Path = PROJECT_ROOT / "ml_models"
    
    # Database
    DATABASE_URL: str = "sqlite:///./cricoptima.db"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Cricket API (optional - for live data)
    CRICKET_API_KEY: Optional[str] = None
    CRICKET_API_URL: str = "https://api.cricapi.com/v1"
    
    # Fantasy Game Settings
    BUDGET_LIMIT: int = 1000
    TEAM_SIZE: int = 11
    MAX_BATSMEN: int = 5
    MAX_BOWLERS: int = 5
    MAX_ALL_ROUNDERS: int = 3
    MAX_WICKET_KEEPERS: int = 1
    MIN_BATSMEN: int = 3
    MIN_BOWLERS: int = 3
    MIN_ALL_ROUNDERS: int = 1
    MIN_WICKET_KEEPERS: int = 1
    
    # ML Settings
    MODEL_PATH: Path = ML_MODELS_DIR / "player_predictor.joblib"
    PREDICTION_FEATURES: list = [
        "recent_avg_runs", "recent_avg_wickets", "strike_rate",
        "economy_rate", "matches_played", "venue_avg", "opposition_avg"
    ]
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(exist_ok=True)
settings.ML_MODELS_DIR.mkdir(exist_ok=True)
