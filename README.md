# 🏏 CricOptima

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

**AI-Powered Fantasy Cricket Team Optimizer**

Build optimal fantasy cricket teams using machine learning predictions and constraint optimization algorithms.

![CricOptima Dashboard](images/fantasy_cricket1.jpg)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Predictions** | Gradient Boosting model predicts player fantasy points |
| 🎯 **Smart Optimization** | Builds best team within budget and role constraints |
| 📊 **Analytics Dashboard** | Explore player stats, value scores, and predictions |
| 🔌 **REST API** | Full-featured FastAPI backend with Swagger docs |
| 🐳 **Docker Ready** | One-command deployment with docker-compose |

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/millenniumsingha/CricOptima.git
cd CricOptima

# Train ML model and start services
docker-compose --profile training up train
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API Docs:  http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train ML model
python -m src.ml.train

# Start API (terminal 1)
uvicorn api.main:app --reload

# Start Dashboard (terminal 2)
streamlit run app/streamlit_app.py
```

## 📊 How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Player Data   │────▶│   ML Predictor  │────▶│   Predictions   │
│   (Stats/Form)  │     │ (Gradient Boost)│     │  (Points/Conf)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │    Optimizer    │◀─────────────┘
                        │  (Constraints)  │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   Optimal XI    │
                        │ (Best Team)     │
                        └─────────────────┘
```

### ML Features Used

- Recent batting/bowling averages
- Strike rate & economy rate
- Form trend (improving/declining)
- Consistency score
- Matches played (experience)

### Optimization Constraints

- Budget limit (default: 1000 points)
- Team size: 11 players
- Min 3 batsmen, 3 bowlers, 1 all-rounder, 1 wicket-keeper
- Max 7 players from same team

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/players` | List all players with filters |
| GET | `/players/{id}` | Get single player details |
| POST | `/optimize` | Build optimal team |
| GET | `/predictions` | Get ML predictions |
| POST | `/teams/validate` | Validate team selection |
| GET | `/health` | Health check |

### Example: Get Optimal Team

```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{"budget": 1000, "team_name": "My Dream XI"}'
```

## 📁 Project Structure

```
CricOptima/
├── src/
│   ├── models/          # Data models (Player, Team, Match)
│   ├── scoring/         # Fantasy points calculator
│   ├── optimizer/       # Team optimization algorithm
│   ├── ml/              # ML prediction model
│   └── data/            # Data layer & sample data
├── api/                 # FastAPI backend
├── app/                 # Streamlit dashboard
├── tests/               # Test suite
├── ml_models/           # Trained models
└── legacy/              # Original project files
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=api
```

## 🎯 Fantasy Scoring Rules

| Category | Points |
|----------|--------|
| Run | 0.5 |
| Four | +1 bonus |
| Six | +2 bonus |
| 50 runs | +10 bonus |
| 100 runs | +20 bonus |
| Wicket | 10 |
| 3-wicket haul | +5 bonus |
| 5-wicket haul | +10 bonus |
| Catch/Stumping/Run-out | 10 |

## 🚧 Roadmap

- [ ] Live cricket data integration (CricAPI)
- [ ] User authentication & team saving
- [ ] Head-to-head matchup predictions
- [ ] Mobile-responsive design
- [ ] Historical match simulation

## 📜 License

MIT License

## 🙏 Acknowledgments

- Original project from Internshala Python Training
- IPL teams and player data
- scikit-learn, FastAPI, and Streamlit communities

---

*Built with ❤️ for cricket fans and data enthusiasts*
