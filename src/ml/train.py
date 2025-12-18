"""Training script for ML model."""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.ml.predictor import PlayerPredictor
from src.data.sample_data import generate_training_data
from src.config import settings


def train_model(use_sample_data: bool = True) -> dict:
    """
    Train the player prediction model.
    
    Args:
        use_sample_data: If True, use generated sample data for training
        
    Returns:
        Training metrics
    """
    print("=" * 50)
    print("CricOptima - ML Model Training")
    print("=" * 50)
    
    # Get training data
    if use_sample_data:
        print("\nGenerating sample training data...")
        players, actual_points = generate_training_data(n_players=200)
        print(f"Generated {len(players)} training samples")
    else:
        # Load from historical data file
        raise NotImplementedError("Historical data loading not yet implemented")
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    predictor = PlayerPredictor()
    metrics = predictor.train(players, actual_points, save_model=True)
    
    # Print results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"\nCross-Validation R² Score: {metrics['cv_r2_mean']:.3f} (+/- {metrics['cv_r2_std']:.3f})")
    print(f"Training samples: {metrics['n_samples']}")
    print(f"\nModel saved to: {settings.MODEL_PATH}")
    
    print("\nTop 5 Important Features:")
    importance = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feature, score in importance:
        print(f"  - {feature}: {score:.3f}")
    
    # Save metrics
    metrics_path = settings.MODEL_PATH.with_suffix('.metrics.json')
    metrics['timestamp'] = datetime.now().isoformat()
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CricOptima ML model")
    parser.add_argument(
        "--sample-data", 
        action="store_true", 
        default=True,
        help="Use generated sample data for training"
    )
    
    args = parser.parse_args()
    train_model(use_sample_data=args.sample_data)
