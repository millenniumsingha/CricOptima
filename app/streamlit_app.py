import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import requests
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.player import PlayerPool, PlayerRole
from src.models.team import Team, TeamConstraints
from src.optimizer.team_builder import TeamOptimizer
from src.ml.predictor import get_predictor
from src.ml.train import train_model
from src.data.mock_provider import MockDataProvider
from src.data.live_provider import LiveDataProvider
from src.config import settings

# Page config
st.set_page_config(
    page_title="CricOptima - Fantasy Cricket Optimizer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    load_css(css_path)



@st.cache_resource
def load_data(match_id: str = None):
    """Load player data and ML model."""
    if settings.DATA_SOURCE == "live":
        provider = LiveDataProvider()
        # Use provided match_id or default
        mid = match_id or "t20_match_01" 
        players = provider.get_players(match_id=mid)
    else:
        provider = MockDataProvider()
        players = provider.get_players()

    model_loaded = False
    
    # Try to load predictions
    try:
        predictor = get_predictor()
        predictor.load()
        players = predictor.enrich_players_with_predictions(players)
        model_loaded = True
    except FileNotFoundError:
        # Auto-train logic for deployment
        with st.spinner("🚀 Performing initial setup & training ML model..."):
            train_model(use_sample_data=True)
            
            # Retry loading
            try:
                predictor = get_predictor()
                predictor.load()
                players = predictor.enrich_players_with_predictions(players)
                model_loaded = True
            except Exception as e:
                st.error(f"Failed to auto-train model: {e}")
                
    if not model_loaded:
        # Fallback if training fails
        for p in players:
            p.predicted_points = p.stats.batting_average + p.stats.recent_avg_wickets * 10
            p.prediction_confidence = 0.5
            
    return PlayerPool(players=players), model_loaded


API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"

def login_user(username, password):
    """Login user via API or Direct DB."""
    # Try API first
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/token",
            data={"username": username, "password": password},
            timeout=1
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback to Direct DB
    from src.auth import authenticate_user, create_access_token
    
    db = get_db_session()
    try:
        user = authenticate_user(db, username, password)
        if not user:
            return None
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()

def get_db_session():
    # Lazy import to avoid startup errors on Cloud (if dependencies missing)
    from src.db import SessionLocal
    return SessionLocal()

def register_user(username, password):
    """Register user via API or Direct DB."""
    # Try API first
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"username": username, "password": password},
            timeout=1
        )
        return response.status_code == 200
    except:
        pass
        
    # Fallback to Direct DB
    db = get_db_session()
    from src.db import User as DBUser
    from src.auth import get_password_hash
    try:
        if db.query(DBUser).filter(DBUser.username == username).first():
            return False # Already exists
        
        hashed_password = get_password_hash(password)
        db_user = DBUser(username=username, hashed_password=hashed_password)
        db.add(db_user)
        db.commit()
        return True
    except Exception as e:
        print(f"Direct register failed: {e}")
        return False
    finally:
        db.close()

def save_team_api(token, name, team_data):
    """Save team via API or Direct DB."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{API_BASE_URL}/teams/",
            json={"name": name, "team_data": team_data},
            headers=headers,
            timeout=1
        )
        return response.status_code == 200
    except:
        pass

    # Direct DB (simplified, assumes token is valid username for now in fallback)
    # Ideally should decode token, but for now assuming direct use
    from jose import jwt, JWTError
    from src.auth import SECRET_KEY, ALGORITHM
    from src.db import User as DBUser, SavedTeam as DBSavedTeam
    
    db = get_db_session()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username: return False
        
        user = db.query(DBUser).filter(DBUser.username == username).first()
        if not user: return False
        
        # Check limit
        if len(user.saved_teams) >= 5: return False # Hard limit for fallback
        
        import json
        db_team = DBSavedTeam(
            name=name,
            team_data=json.dumps(team_data),
            user_id=user.id
        )
        db.add(db_team)
        db.commit()
        return True
    except Exception as e:
        print(f"Direct save failed: {e}")
        return False
    finally:
        db.close()

def get_my_teams_api(token):
    """Get teams via API or Direct DB."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_BASE_URL}/teams/",
            headers=headers,
            timeout=1
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
        
    # Direct DB
    from jose import jwt
    from src.auth import SECRET_KEY, ALGORITHM
    from src.db import User as DBUser
    import json
    
    db = get_db_session()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = db.query(DBUser).filter(DBUser.username == username).first()
        if not user: return []
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "team_data": json.loads(t.team_data),
                "created_at": t.created_at.isoformat()
            }
            for t in user.saved_teams
        ]
    except:
        return []
    finally:
        db.close()

def delete_team_api(token, team_id):
    """Delete team via API or Direct DB."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        requests.delete(
            f"{API_BASE_URL}/teams/{team_id}",
            headers=headers,
            timeout=1
        )
        return True
    except:
        pass
        
    # Direct DB
    from src.db import SavedTeam as DBSavedTeam
    db = get_db_session()
    try:
        # Simplification: In direct mode we trust the ID belongs to user or verify via join
        # For full correctness we should verify ownership
        team = db.query(DBSavedTeam).filter(DBSavedTeam.id == team_id).first()
        if team:
            db.delete(team)
            db.commit()
            return True
        return False
    except:
        return False
    finally:
        db.close()

def compare_teams_api(team_a_data, team_b_data):
    """Compare two teams."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/teams/compare/",
            json={"team_a": team_a_data, "team_b": team_b_data}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def simulate_match_api(team_a_data, team_b_data, iterations=1000):
    """Run simulation."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/teams/simulate/",
            json={"team_a": team_a_data, "team_b": team_b_data, "iterations": iterations}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def main():
    # Header
    st.markdown('<p class="main-header">🏏 CricOptima</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Fantasy Cricket Team Optimizer</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar Settings
    with st.sidebar:
        st.header("👤 User Account")
        
        # Check session state
        if 'token' not in st.session_state:
            auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
            
            with auth_tab1:
                l_user = st.text_input("Username", key="l_user")
                l_pass = st.text_input("Password", type="password", key="l_pass")
                if st.button("Login"):
                    token_data = login_user(l_user, l_pass)
                    if token_data:
                        st.session_state['token'] = token_data['access_token']
                        st.session_state['username'] = l_user
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            
            with auth_tab2:
                r_user = st.text_input("Username", key="r_user")
                r_pass = st.text_input("Password", type="password", key="r_pass")
                if st.button("Register"):
                    if register_user(r_user, r_pass):
                        st.success("Registered! Please login.")
                    else:
                        st.error("Registration failed")
        else:
            st.success(f"Welcome, {st.session_state['username']}!")
            if st.button("Logout"):
                del st.session_state['token']
                del st.session_state['username']
                st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Data Source Toggle
        use_live_data = st.toggle("Use Live Data (CricAPI)", value=False)
        
        if use_live_data:
            settings.DATA_SOURCE = "live"
            api_key = st.text_input("CricAPI Key", type="password", help="Get free key from cricapi.com")
            if api_key:
                settings.CRIC_API_KEY = api_key
                
                # Fetch matches button
                if st.button("🔄 Fetch Current Matches"):
                    try:
                        # Version 2.1.0: Using new provider file to bypass caching issues
                        from src.data.cric_data_provider_v2 import LiveDataProvider
                        
                        import os
                        os.environ["CRIC_API_KEY"] = api_key
                        
                        provider = LiveDataProvider()
                        matches = provider.get_current_matches()
                        
                        if matches:
                            st.session_state['available_matches'] = matches
                            st.success(f"Found {len(matches)} matches!")
                        else:
                            st.warning("No matches found. Check API key or try again.")
                    except Exception as e:
                        st.error(f"Error fetching matches: {str(e)}")
                
                # Match selector
                if 'available_matches' in st.session_state and st.session_state['available_matches']:
                    matches = st.session_state['available_matches']
                    match_options = {f"{m['name']} ({m['type']})": m['id'] for m in matches}
                    selected_match_name = st.selectbox(
                        "Select Match",
                        options=list(match_options.keys()),
                        help="Select a match to load players from"
                    )
                    selected_match_id = match_options.get(selected_match_name)
                else:
                    st.info("👆 Click 'Fetch Current Matches' to see available matches")
                    selected_match_id = None
            else:
                st.warning("⚠️ Please enter API Key")
                selected_match_id = None
        else:
            settings.DATA_SOURCE = "mock"
            selected_match_id = None
        
        # Budget
        budget = st.slider("Budget", 500, 1500, settings.BUDGET_LIMIT, step=50)

    # Load data
    with st.spinner("Loading data..."):
        # Reload provider if source changed
        player_pool, model_loaded = load_data(match_id=selected_match_id)
    
    # Sidebar Constraints
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Constraints")
        
        col1, col2 = st.columns(2)
        with col1:
            min_bat = st.number_input("Min Batsmen", 1, 5, 3)
            min_bowl = st.number_input("Min Bowlers", 1, 5, 3)
        with col2:
            max_bat = st.number_input("Max Batsmen", 3, 6, 5)
            max_bowl = st.number_input("Max Bowlers", 3, 6, 5)
        
        st.markdown("---")
        
        # Model status
        if model_loaded:
            st.success("✅ ML Model Loaded")
        else:
             if use_live_data and not api_key:
                st.warning("waiting for key...")
             else:
                st.warning("⚠️ ML Model not trained")
                st.caption("Auto-training...")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Auto-Optimize", 
        "⚔️ Head-to-Head",
        "🏃 Player Pool", 
        "📈 Analytics",
        "📂 My Teams",
        "ℹ️ About"
    ])
    
    # Tab 1: Auto-Optimize
    with tab1:
        st.header("Build Optimal Team")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            team_name = st.text_input("Team Name", "My Dream XI")
            
            if st.button("🚀 Generate Optimal Team", type="primary", use_container_width=True):
                constraints = TeamConstraints(
                    budget=budget,
                    min_batsmen=min_bat,
                    max_batsmen=max_bat,
                    min_bowlers=min_bowl,
                    max_bowlers=max_bowl
                )
                
                optimizer = TeamOptimizer(constraints)
                
                with st.spinner("Optimizing team selection..."):
                    result = optimizer.optimize(player_pool, team_name)
                
                st.session_state['optimized_team'] = result
                st.success("✅ Team optimized successfully!")
        
        # Display optimized team
        if 'optimized_team' in st.session_state:
            result = st.session_state['optimized_team']
            team = result.team
            
            st.markdown("---")
            st.subheader(f"🏆 {team.name}")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Points", f"{result.total_predicted_points:.1f}")
            m2.metric("Total Cost", f"{result.total_cost}")
            m3.metric("Budget Left", f"{result.budget_remaining}")
            m4.metric("Value Score", f"{result.optimization_score:.3f}")
            
            # Player table
            st.markdown("### Selected Players")
            
            player_data = []
            for p in team.players:
                badges = ""
                if p.id == team.captain_id:
                    badges = "👑 C"
                elif p.id == team.vice_captain_id:
                    badges = "⭐ VC"
                
                player_data.append({
                    "": badges,
                    "Player": p.name,
                    "Team": p.team,
                    "Role": p.role,
                    "Cost": p.cost,
                    "Pred. Pts": f"{p.predicted_points:.1f}" if p.predicted_points else "N/A",
                    "Confidence": f"{p.prediction_confidence:.0%}" if p.prediction_confidence else "N/A"
                })
            
            df = pd.DataFrame(player_data)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Role distribution chart
            col1, col2 = st.columns(2)
            
            with col1:
                role_counts = team.role_counts
                if role_counts:
                    role_df = pd.DataFrame({
                        "Role": list(role_counts.keys()),
                        "Count": list(role_counts.values())
                    })
                    fig_roles = px.pie(
                        role_df,
                        values="Count",
                        names="Role",
                        title="Role Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig_roles, use_container_width=True)
                else:
                    st.info("No role data available")
            
            with col2:
                team_counts = team.team_counts
                if team_counts:
                    team_df = pd.DataFrame({
                        "Team": list(team_counts.keys()),
                        "Players": list(team_counts.values())
                    })
                    fig_teams = px.bar(
                        team_df,
                        x="Team",
                        y="Players",
                        title="Team Distribution"
                    )
                    st.plotly_chart(fig_teams, use_container_width=True)
                else:
                    st.info("No team data available")
            
            # Save Team Button (Authenticated)
            if 'token' in st.session_state:
                st.markdown("### 💾 Save Team")
                if st.button("Save to My Account"):
                    # Serialize team data
                    team_json = {
                        "name": team.name,
                        "players": [p.id for p in team.players],
                        "total_cost": result.total_cost,
                        "predicted_points": result.total_predicted_points
                    }
                    if save_team_api(st.session_state['token'], team.name, team_json):
                        st.success("Team saved to your account!")
                    else:
                        st.error("Failed to save team.")
            else:
                st.info("💡 Login to save this team to your account.")
    
    # Tab 2: Head-to-Head
    with tab2:
        st.header("⚔️ Head-to-Head Comparison")
        
        if 'token' not in st.session_state:
             st.warning("🔒 Please login to use the Head-to-Head feature.")
        else:
            my_teams = get_my_teams_api(st.session_state['token'])
            if not my_teams:
                st.info("You need to save at least one team to use this feature.")
            else:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("🟦 Your Team")
                    team_a_idx = st.selectbox(
                        "Select Team A", 
                        range(len(my_teams)), 
                        format_func=lambda i: my_teams[i]['name']
                    )
                    team_a = my_teams[team_a_idx]
                    st.metric("Points", f"{team_a['team_data'].get('predicted_points', 0):.1f}")

                with c2:
                    st.subheader("🟥 Opponent")
                    # For now just compare against another saved team or allow same team
                    team_b_idx = st.selectbox(
                        "Select Team B", 
                        range(len(my_teams)), 
                        format_func=lambda i: my_teams[i]['name'],
                        key="tb_select"
                    )
                    team_b = my_teams[team_b_idx]
                    st.metric("Points", f"{team_b['team_data'].get('predicted_points', 0):.1f}")
                
                if st.button("🥊 Fight!", type="primary", use_container_width=True):
                    result = compare_teams_api(team_a['team_data'], team_b['team_data'])
                    
                    if result:
                        st.markdown("---")
                        
                        # Win Prob
                        prob = result['win_probability_a']
                        diff = result['point_diff']
                        
                        if prob > 0.5:
                            st.success(f"🏆 {result['team_a']['name']} Wins!")
                            st.balloons()
                        elif prob < 0.5:
                            st.error(f"💀 {result['team_b']['name']} Wins!")
                        else:
                            st.info("🤝 It's a Tie!")
                        
                        # Progress bar for probability
                        st.write(f"**Win Probability ({result['team_a']['name']})**")
                        st.progress(prob)
                        st.caption(f"{prob:.1%} chance of winning")
                        
                        st.info(f"Point Difference: {diff:.1f}")

                st.markdown("---")
                st.subheader("🎲 Monte Carlo Simulation")
                st.caption("Simulate 1,000 matches considering player point variance.")
                
                if st.button("Run Simulation (1000x)"):
                    with st.spinner("Simulating..."):
                        sim_result = simulate_match_api(team_a['team_data'], team_b['team_data'])
                    
                    if sim_result:
                        res_a = sim_result['team_a']
                        res_b = sim_result['team_b']
                        
                        # Metrics
                        m1, m2 = st.columns(2)
                        m1.metric(f"Win Probability ({res_a['name']})", f"{res_a['win_probability']:.1%}")
                        m2.metric(f"Win Probability ({res_b['name']})", f"{res_b['win_probability']:.1%}")
                        
                        # Histogram
                        import plotly.figure_factory as ff
                        
                        hist_data = [
                            sim_result['simulation_data']['scores_a'],
                            sim_result['simulation_data']['scores_b']
                        ]
                        group_labels = [res_a['name'], res_b['name']]
                        
                        fig_dist = ff.create_distplot(
                            hist_data, group_labels, bin_size=10, 
                            show_rug=False, 
                            colors=['#3b82f6', '#ef4444']
                        )
                        fig_dist.update_layout(title_text="Score Distribution (1000 Matches)")
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Stats Table
                        st.markdown("### 📊 Detailed Stats")
                        stats_df = pd.DataFrame([
                            {"Team": res_a['name'], "Mean": res_a['stats']['mean'], "Min": res_a['stats']['min'], "Max": res_a['stats']['max']},
                            {"Team": res_b['name'], "Mean": res_b['stats']['mean'], "Min": res_b['stats']['min'], "Max": res_b['stats']['max']}
                        ])
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    
    # Tab 3: Player Pool
    with tab3:
        st.header("Available Players")
        
        # Filters
        with st.expander("🔍 Filter & Sort Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                role_filter = st.selectbox(
                    "Filter by Role",
                    ["All"] + [r.value for r in PlayerRole]
                )
            
            with col2:
                teams = sorted(set(p.team for p in player_pool.players))
                team_filter = st.selectbox("Filter by Team", ["All"] + teams)
            
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Predicted Points", "Cost", "Value Score", "Name"]
                )
        
        # Apply filters
        players = player_pool.players
        
        if role_filter != "All":
            players = [p for p in players if p.role == role_filter]
        if team_filter != "All":
            players = [p for p in players if p.team == team_filter]
        
        # Sort
        if sort_by == "Predicted Points":
            players = sorted(players, key=lambda p: p.predicted_points or 0, reverse=True)
        elif sort_by == "Cost":
            players = sorted(players, key=lambda p: p.cost, reverse=True)
        elif sort_by == "Value Score":
            players = sorted(players, key=lambda p: p.value_score, reverse=True)
        else:
            players = sorted(players, key=lambda p: p.name)
        
        # Display
        player_data = []
        for p in players:
            player_data.append({
                "Name": p.name,
                "Team": p.team,
                "Role": p.role,
                "Cost": p.cost,
                "Batting Avg": f"{p.stats.batting_average:.1f}",
                "Strike Rate": f"{p.stats.strike_rate:.1f}",
                "Pred. Points": f"{p.predicted_points:.1f}" if p.predicted_points else "N/A",
                "Value": f"{p.value_score:.3f}",
                "Recent Form": str(p.stats.recent_runs)
            })
        
        df = pd.DataFrame(player_data)
        st.dataframe(df, hide_index=True, use_container_width=True, height=500)
    
    # Tab 4: Analytics
    with tab4:
        st.header("Player Analytics")
        
        # Scatter plot: Cost vs Predicted Points
        if player_pool.players:
            plot_data = pd.DataFrame([{
                "Cost": p.cost,
                "Predicted Points": p.predicted_points or 0,
                "Role": p.role,
                "Name": p.name
            } for p in player_pool.players])
            
            fig_scatter = px.scatter(
                plot_data,
                x="Cost",
                y="Predicted Points",
                color="Role",
                hover_name="Name",
                title="Cost vs Predicted Points by Role"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No player data available for analytics. Please check your data source.")
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Top 10 by Predicted Points")
            top_points = sorted(
                player_pool.players,
                key=lambda p: p.predicted_points or 0,
                reverse=True
            )[:10]
            
            for i, p in enumerate(top_points, 1):
                st.write(f"{i}. **{p.name}** ({p.role}) - {p.predicted_points:.1f} pts")
        
        with col2:
            st.subheader("💰 Top 10 by Value Score")
            top_value = sorted(
                player_pool.players,
                key=lambda p: p.value_score,
                reverse=True
            )[:10]
            
            for i, p in enumerate(top_value, 1):
                st.write(f"{i}. **{p.name}** ({p.role}) - {p.value_score:.3f}")

    # Tab 5: My Teams
    with tab5:
        st.header("📂 My Saved Teams")
        
        if 'token' in st.session_state:
            my_teams = get_my_teams_api(st.session_state['token'])
            
            if not my_teams:
                st.info("No saved teams found.")
            else:
                for t in my_teams:
                    with st.expander(f"{t['name']} (Created: {t['created_at'][:10]})"):
                        data = t['team_data']
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Points", f"{data.get('predicted_points', 0):.1f}")
                        c2.metric("Cost", f"{data.get('total_cost', 0)}")
                        
                        if st.button("Delete Team", key=f"del_{t['id']}"):
                            if delete_team_api(st.session_state['token'], t['id']):
                                st.success("Deleted!")
                                st.rerun()
        else:
            st.warning("🔒 Please login to view your saved teams.")
    
    # Tab 6: About
    with tab6:
        st.header("About CricOptima")
        
        st.markdown("""
        ### 🏏 What is CricOptima?
        
        CricOptima is an AI-powered fantasy cricket team optimizer that helps you 
        build the best possible team within your budget constraints.
        
        ### 🧠 How It Works
        
        1. **ML Predictions**: Uses machine learning to predict player fantasy points
           based on historical performance, recent form, and other features.
        
        2. **Optimization**: Applies constraint optimization algorithms to select
           the best team within budget and role requirements.
        
        3. **Smart Recommendations**: Suggests captain and vice-captain based on
           predicted performance.
        
        ### 📊 Features
        
        - **Auto Team Builder**: Generate optimal team with one click
        - **Player Analytics**: Explore player stats and predictions
        - **Custom Constraints**: Adjust budget and role requirements
        - **Value Analysis**: Find undervalued players with high potential
        
        ### 🛠️ Technology Stack
        
        - **ML Model**: Gradient Boosting for performance prediction
        - **Optimization**: Greedy constraint satisfaction algorithm
        - **Backend**: FastAPI REST API
        - **Frontend**: Streamlit dashboard
        - **Data**: IPL player statistics
        
        ---
        
        *Built with ❤️ for cricket fans*
        """)


if __name__ == "__main__":
    main()
