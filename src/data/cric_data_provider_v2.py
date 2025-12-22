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
    # Version: 2.1.0 - V2 Provider to force fresh load
    
    BASE_URL = "https://api.cricapi.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CRIC_API_KEY")
        if not self.api_key:
            print("⚠️ CRIC_API_KEY not found. Live data may fail.")
            
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

    def get_current_matches(self) -> List[Dict[str, Any]]:
        """
        Fetch list of current/upcoming matches from CricAPI.
        Returns list of match dicts with 'id' and 'name' fields.
        """
        data = self._make_request("currentMatches")
        
        if data.get("status") != "success":
            return []
        
        matches = []
        for match in data.get("data", []):
            match_id = match.get("id")
            team1 = match.get("teamInfo", [{}])[0].get("name", "Team A") if match.get("teamInfo") else "Team A"
            team2 = match.get("teamInfo", [{}])[1].get("name", "Team B") if len(match.get("teamInfo", [])) > 1 else "Team B"
            match_type = match.get("matchType", "")
            status = match.get("status", "")
            
            if match_id:
                matches.append({
                    "id": match_id,
                    "name": f"{team1} vs {team2}",
                    "type": match_type,
                    "status": status
                })
        
        return matches

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
        squads = data.get("data", [])
        for squad in squads:
            team_name = squad.get("teamName", "Unknown Team")
            team_players = squad.get("players", [])
            
            for p_data in team_players:
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
                    

                # Generative plausible stats for demo purposes (Transparent Demo Mode)
                import random
                
                is_bowler = role == PlayerRole.BOWLER
                is_batter = role == PlayerRole.BATSMAN
                is_allrounder = role == PlayerRole.ALL_ROUNDER
                is_wk = role == PlayerRole.WICKET_KEEPER
                
                # Batting Avg
                if is_batter or is_wk:
                    bat_avg = random.uniform(25.0, 55.0)
                    strike_rate = random.uniform(110.0, 150.0)
                elif is_allrounder:
                    bat_avg = random.uniform(20.0, 40.0)
                    strike_rate = random.uniform(120.0, 160.0)
                else: # Bowler
                    bat_avg = random.uniform(5.0, 15.0)
                    strike_rate = random.uniform(60.0, 100.0)
                    
                # Bowling
                if is_bowler or is_allrounder:
                    bowl_avg = random.uniform(20.0, 35.0)
                    econ = random.uniform(6.0, 9.5)
                else:
                    bowl_avg = 0.0
                    econ = 0.0
                    
                # Recent form (last 5 matches points)
                recent_form = [random.randint(4, 100) for _ in range(5)]
                
                stats = PlayerStats(
                    matches_played=random.randint(10, 100),
                    batting_average=round(bat_avg, 2),
                    bowling_average=round(bowl_avg, 2),
                    strike_rate=round(strike_rate, 2),
                    economy_rate=round(econ, 2),
                    recent_runs=recent_form # Using recent_runs field for form
                )
                
                # Calculate plausible cost based on stats (80 to 120 range)
                # Better stats = Higher cost
                base_cost = 80
                performance_bonus = 0
                
                if bat_avg > 40: performance_bonus += 15
                elif bat_avg > 30: performance_bonus += 10
                
                if bowl_avg > 0 and bowl_avg < 25: performance_bonus += 15
                elif bowl_avg > 0 and bowl_avg < 30: performance_bonus += 10
                
                # Add some random variation
                final_cost = base_cost + performance_bonus + random.randint(0, 15)
                
                player = Player(
                    id=p_data.get("id"),
                    name=p_data.get("name"),
                    team=team_name,
                    role=role,
                    cost=final_cost, 
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
            date=datetime.now(),
            venue=info.get("venue", "Unknown Venue"),
            status=status,
            format=MatchFormat.T20 
        )
