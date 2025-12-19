from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from src.db import get_db, SavedTeam, User
from src.auth import get_current_user
from api.schemas import SavedTeamCreate, SavedTeamResponse

router = APIRouter(
    prefix="/teams",
    tags=["Teams"]
)

@router.get("/", response_model=List[SavedTeamResponse])
def get_my_teams(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all saved teams for the current user."""
    return current_user.teams

@router.post("/", response_model=SavedTeamResponse)
def save_team(
    team: SavedTeamCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save a new team."""
    db_team = SavedTeam(
        name=team.name,
        team_data=team.team_data,
        user_id=current_user.id
    )
    db.add(db_team)
    db.commit()
    db.refresh(db_team)
    return db_team

@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_team(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a saved team."""
    team = db.query(SavedTeam).filter(SavedTeam.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    if team.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this team")
    
    db.delete(team)
    db.commit()
