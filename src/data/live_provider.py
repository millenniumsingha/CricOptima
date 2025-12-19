import os
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.data.base import BaseDataProvider
from src.models.player import Player, PlayerRole, PlayerStats
from src.models.match import Match, MatchStatus, MatchFormat
from src.config import settings

class LiveDataProvider(BaseDataProvider):
    """
    Fetches real-time cricket data from CricAPI.
    Requires CRIC_API_KEY in environment variables.
    """
    
    BASE_URL = "https://api.cricapi.com/v1"
    
    def __init__(self):
        self.api_key = os.getenv("CRIC_API_KEY")
        if not self.api_key:
            print("⚠️ CRIC_API_KEY not found in env. Live data may fail.")
            
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Helper to make API requests."""
        if params is None:
            params = {}
        
        params["apikey"] = self.api_key
        
        try:
            response = requests.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from {endpoint}: {e}")
            return {"status": "failure", "data": []}

    def get_players(self, match_id: Optional[str] = None) -> List[Player]:
        """
        Fetch players for a live match.
        
        Args:
            match_id: The specific match ID to fetch squads for.
        """
        if not match_id:
            print("⚠️ No match_id provided for live data. Returning empty list.")
            return []
            
        data = self._make_request("match_squad", {"id": match_id})
        
        if data.get("status") != "success":
            return []
            
        players = []
        
        # Parse squads
        # Note: API response structure depends on the specific provider.
        # This is a generic implementation for CricAPI's 'squad' endpoint.
        
        # For demonstration, we'll iterate through the teams provided in squads
        squads = data.get("data", [])
        for squad in squads:
            team_name = squad.get("teamName", "Unknown Team")
            team_players = squad.get("players", [])
            
            for p_data in team_players:
                # API doesn't always provide full stats, so we might need to fetch individual player info
                # or infer/mock some stats if they are missing.
                
                # Determining role safely
                role_str = p_data.get("role", "batting").lower()
                if "wike" in role_str or "keep" in role_str:
                    role = PlayerRole.WICKET_KEEPER
                elif "all" in role_str:
                    role = PlayerRole.ALL_ROUNDER
                elif "bowl" in role_str:
                    role = PlayerRole.BOWLER
                else:
                    role = PlayerRole.BATSMAN
                    
                # We need to fetch detailed stats if not present
                # For MVP, we might initialize with minimal stats
                
                stats = PlayerStats(
                    matches_played=0,
                    batting_average=0.0,
                    # We can fetch detailed stats in a separate call if needed
                    # or leave them 0 to be filled by the ML predictor features
                )
                
                player = Player(
                    id=p_data.get("id"),
                    name=p_data.get("name"),
                    team=team_name,
                    role=role,
                    cost=100, # Placeholder cost, would need a pricing algorithm
                    stats=stats
                )
                players.append(player)
                
        return players

    def get_match_info(self, match_id: str) -> Match:
        """Fetch live match info."""
        data = self._make_request("match_info", {"id": match_id})
        
        if data.get("status") != "success":
            # Return dummy if failed
            return Match(
                id=match_id,
                team1="Unknown", 
                team2="Unknown",
                date=datetime.now(),
                venue="Unknown",
                status=MatchStatus.UPCOMING
            )
            
        info = data.get("data", {})
        
        status_str = info.get("status", "").upper()
        if "LIVE" in status_str:
            status = MatchStatus.LIVE
        elif "ENDED" in status_str or "COMPLETED" in status_str:
            status = MatchStatus.COMPLETED
        elif "ABANDONED" in status_str:
            status = MatchStatus.ABANDONED
        else:
            status = MatchStatus.UPCOMING

        return Match(
            id=info.get("id", match_id),
            team1=info.get("team1", "Team A"),
            team2=info.get("team2", "Team B"),
            date=datetime.now(), # API returns string date, needs parsing
            venue=info.get("venue", "Unknown Venue"),
            status=status,
            format=MatchFormat.T20 # Default or parse from info
        )
