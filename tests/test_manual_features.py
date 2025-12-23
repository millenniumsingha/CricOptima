import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_local_logic():
    print("Testing Comparator...")
    try:
        from src.scoring.comparator import TeamComparator
        c = TeamComparator()
        # Mock data
        ta = {"name": "A", "predicted_points": 100}
        tb = {"name": "B", "predicted_points": 50}
        res = c.compare_teams(ta, tb)
        print("Comparator Result:", res)
    except Exception as e:
        print("Comparator Failed:", e)

    print("\nTesting Simulator...")
    try:
        from src.scoring.simulator import MatchSimulator
        s = MatchSimulator(iterations=100)
        res = s.simulate_match(ta, tb)
        print("Simulator Result Keys:", res.keys())
        print("Win Prob A:", res['team_a']['win_probability'])
    except Exception as e:
        print("Simulator Failed:", e)

if __name__ == "__main__":
    test_local_logic()
