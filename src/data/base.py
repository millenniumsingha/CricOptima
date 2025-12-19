from abc import ABC, abstractmethod
from typing import List, Optional
from src.models.player import Player
from src.models.match import Match
from src.models.team import Team

class BaseDataProvider(ABC):
    """
    Abstract Interface for fetching cricket data.
    Allows switching between Mock data (for demo/testing) and Live API data.
    """
    
    @abstractmethod
    def get_players(self, match_id: Optional[str] = None) -> List[Player]:
        """
        Fetch list of players for a match.
        
        Args:
            match_id: ID of the match to fetch players for.
                     If None, can return a default pool (for Mock).
        """
        pass

    @abstractmethod
    def get_match_info(self, match_id: str) -> Match:
        """
        Get details about a specific match.
        """
        pass
