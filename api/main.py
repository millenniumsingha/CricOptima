"""FastAPI application for CricOptima."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from src import __version__
from src.config import settings
from src.models.player import PlayerPool, Player
from src.models.team import Team, TeamConstraints
from src.optimizer.team_builder import TeamOptimizer
from src.ml.predictor import PlayerPredictor, get_predictor
from src.data.mock_provider import MockDataProvider
from src.data.live_provider import LiveDataProvider
from api.schemas import (
    PlayerResponse, TeamRequest, TeamResponse,
    OptimizationRequest, OptimizationResponse,
    PredictionResponse, HealthResponse
)
from api.routers import auth
from src.db import init_db


# Initialize FastAPI
app = FastAPI(
    title="CricOptima API",
    description="""
    🏏 **CricOptima API**
    
    A comprehensive REST API for fantasy cricket team optimization.
    
    ## Core Capabilities
    - **Performance Prediction**: Gradient Boosting regressor for player point forecasting.
    - **Constraint Optimization**: Knapsack-style algorithm for budget and role validation.
    - **Team Management**: Endpoints for team generation and validation.
    - **User Authentication**: Secure registration and login.
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize Database
init_db()

# Include Routers
app.include_router(auth.router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_player_pool: Optional[PlayerPool] = None
_predictor: Optional[PlayerPredictor] = None


def get_player_pool() -> PlayerPool:
    """Get or initialize player pool."""
    global _player_pool
    if _player_pool is None:
        if settings.DATA_SOURCE == "live":
            # TODO: Add endpoint to set match_id context
            provider = LiveDataProvider()
            # For now, return empty or default match's players
            players = provider.get_players() 
        else:
            provider = MockDataProvider()
            players = provider.get_players()
        
        # Add predictions
        try:
            predictor = get_predictor()
            players = predictor.enrich_players_with_predictions(players)
        except FileNotFoundError:
            pass  # Model not trained yet
        
        _player_pool = PlayerPool(players=players)
    return _player_pool


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    global _predictor
    try:
        _predictor = get_predictor()
        _predictor.load()
        print("✅ ML model loaded successfully")
    except FileNotFoundError:
        print("⚠️ ML model not found. Run training first: python -m src.ml.train")
    
    # Initialize player pool
    get_player_pool()
    print(f"✅ Loaded {len(get_player_pool().players)} players")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    pool = get_player_pool()
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": _predictor._is_fitted if _predictor else False,
        "players_available": len(pool.players) if pool else 0
    }

@app.get("/", response_model=dict)
async def root():
    """API root with info."""
    return {
        "name": "CricOptima API",
        "version": __version__,
        "description": "Fantasy Cricket Optimizer",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    pool = get_player_pool()
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=_predictor is not None and _predictor.is_fitted,
        players_available=len(pool.players)
    )


@app.get("/players", response_model=List[PlayerResponse])
async def get_players(
    role: Optional[str] = Query(None, description="Filter by role: BAT, BOWL, AR, WK"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    min_cost: int = Query(0, ge=0),
    max_cost: int = Query(200, le=200),
    sort_by: str = Query("predicted_points", description="Sort by: predicted_points, cost, value_score, name")
):
    """Get available players with optional filters."""
    pool = get_player_pool()
    players = pool.players
    
    # Apply filters
    if role:
        players = [p for p in players if p.role == role]
    if team:
        players = [p for p in players if team.lower() in p.team.lower()]
    
    players = [p for p in players if min_cost <= p.cost <= max_cost]
    
    # Sort
    if sort_by == "predicted_points":
        players.sort(key=lambda p: p.predicted_points or 0, reverse=True)
    elif sort_by == "cost":
        players.sort(key=lambda p: p.cost, reverse=True)
    elif sort_by == "value_score":
        players.sort(key=lambda p: p.value_score, reverse=True)
    elif sort_by == "name":
        players.sort(key=lambda p: p.name)
    
    return [
        PlayerResponse(
            id=p.id,
            name=p.name,
            team=p.team,
            role=p.role,
            cost=p.cost,
            predicted_points=p.predicted_points,
            prediction_confidence=p.prediction_confidence,
            value_score=p.value_score,
            batting_average=p.stats.batting_average,
            bowling_average=p.stats.bowling_average,
            strike_rate=p.stats.strike_rate,
            economy_rate=p.stats.economy_rate,
            recent_form=p.stats.recent_runs
        )
        for p in players
    ]


@app.get("/players/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: str):
    """Get single player by ID."""
    pool = get_player_pool()
    for p in pool.players:
        if p.id == player_id:
            return PlayerResponse(
                id=p.id,
                name=p.name,
                team=p.team,
                role=p.role,
                cost=p.cost,
                predicted_points=p.predicted_points,
                prediction_confidence=p.prediction_confidence,
                value_score=p.value_score,
                batting_average=p.stats.batting_average,
                bowling_average=p.stats.bowling_average,
                strike_rate=p.stats.strike_rate,
                economy_rate=p.stats.economy_rate,
                recent_form=p.stats.recent_runs
            )
    raise HTTPException(status_code=404, detail="Player not found")


@app.post("/teams/validate", response_model=TeamResponse)
async def validate_team(request: TeamRequest):
    """Validate a team selection."""
    pool = get_player_pool()
    
    # Find selected players
    selected = []
    for pid in request.player_ids:
        found = next((p for p in pool.players if p.id == pid), None)
        if not found:
            raise HTTPException(status_code=400, detail=f"Player {pid} not found")
        selected.append(found)
    
    team = Team(
        name=request.team_name,
        players=selected,
        captain_id=request.captain_id,
        vice_captain_id=request.vice_captain_id
    )
    
    constraints = TeamConstraints()
    violations = team.validate_constraints(constraints)
    
    return TeamResponse(
        name=team.name,
        players=[
            PlayerResponse(
                id=p.id, name=p.name, team=p.team, role=p.role,
                cost=p.cost, predicted_points=p.predicted_points,
                prediction_confidence=p.prediction_confidence,
                value_score=p.value_score,
                batting_average=p.stats.batting_average,
                bowling_average=p.stats.bowling_average,
                strike_rate=p.stats.strike_rate,
                economy_rate=p.stats.economy_rate,
                recent_form=p.stats.recent_runs
            )
            for p in team.players
        ],
        total_cost=team.total_cost,
        budget_remaining=constraints.budget - team.total_cost,
        predicted_points=team.predicted_points,
        is_valid=len(violations) == 0,
        violations=violations,
        captain_id=team.captain_id,
        vice_captain_id=team.vice_captain_id
    )


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_team(request: OptimizationRequest):
    """Build optimal team using ML predictions and optimization."""
    pool = get_player_pool()
    
    # Filter players
    available = [p for p in pool.players if p.id not in request.exclude_players]
    
    # Apply custom constraints
    constraints = TeamConstraints(budget=request.budget)
    
    # Optimize
    optimizer = TeamOptimizer(constraints)
    result = optimizer.optimize(
        PlayerPool(players=available),
        team_name=request.team_name
    )
    
    team = result.team
    violations = team.validate_constraints(constraints)
    
    return OptimizationResponse(
        team=TeamResponse(
            name=team.name,
            players=[
                PlayerResponse(
                    id=p.id, name=p.name, team=p.team, role=p.role,
                    cost=p.cost, predicted_points=p.predicted_points,
                    prediction_confidence=p.prediction_confidence,
                    value_score=p.value_score,
                    batting_average=p.stats.batting_average,
                    bowling_average=p.stats.bowling_average,
                    strike_rate=p.stats.strike_rate,
                    economy_rate=p.stats.economy_rate,
                    recent_form=p.stats.recent_runs
                )
                for p in team.players
            ],
            total_cost=result.total_cost,
            budget_remaining=result.budget_remaining,
            predicted_points=result.total_predicted_points,
            is_valid=len(violations) == 0,
            violations=violations,
            captain_id=result.suggested_captain,
            vice_captain_id=result.suggested_vice_captain
        ),
        optimization_score=result.optimization_score,
        suggested_captain=result.suggested_captain,
        suggested_vice_captain=result.suggested_vice_captain
    )


@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(
    top_n: int = Query(10, ge=1, le=50, description="Number of top predictions to return")
):
    """Get ML predictions for all players, sorted by predicted points."""
    pool = get_player_pool()
    
    players = sorted(
        pool.players,
        key=lambda p: p.predicted_points or 0,
        reverse=True
    )[:top_n]
    
    return [
        PredictionResponse(
            player_id=p.id,
            player_name=p.name,
            predicted_points=p.predicted_points or 0,
            confidence=p.prediction_confidence or 0.5
        )
        for p in players
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
