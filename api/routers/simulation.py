from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from src.scoring.simulator import MatchSimulator

router = APIRouter(
    prefix="/teams/simulate",
    tags=["Simulation"]
)

class SimulationRequest(BaseModel):
    team_a: Dict[str, Any]
    team_b: Dict[str, Any]
    iterations: int = 1000

@router.post("/")
def simulate_match(request: SimulationRequest):
    """Run Monte Carlo simulation for two teams."""
    simulator = MatchSimulator(iterations=request.iterations)
    return simulator.simulate_match(request.team_a, request.team_b)
