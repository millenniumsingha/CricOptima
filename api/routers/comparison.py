from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from src.scoring.comparator import TeamComparator

router = APIRouter(
    prefix="/teams/compare",
    tags=["Comparison"]
)

class ComparisonRequest(BaseModel):
    team_a: Dict[str, Any]
    team_b: Dict[str, Any]

@router.post("/")
def compare_teams(request: ComparisonRequest):
    """Compare two teams and predict winner."""
    comparator = TeamComparator()
    return comparator.compare_teams(request.team_a, request.team_b)
