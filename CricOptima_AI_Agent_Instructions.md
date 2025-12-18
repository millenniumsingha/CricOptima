# AI AGENT TASK: Transform Fantasy Cricket → CricOptima

## MISSION
Transform https://github.com/millenniumsingha/My_Python_FantasyCricket into a modern AI-powered fantasy cricket optimizer.

**First: Rename repository to `CricOptima` on GitHub**

---

## EXECUTION PHASES

### PHASE 1: Structure Setup
```bash
# Create directories
mkdir -p src/models src/scoring src/optimizer src/ml src/data src/utils
mkdir -p api/routes app tests notebooks data ml_models legacy

# Move original files
mv my_fantasy_cricket.py legacy/
mv match_evaluation.py legacy/
mv cricket_match.db legacy/

# Create init files
touch src/__init__.py src/models/__init__.py src/scoring/__init__.py
touch src/optimizer/__init__.py src/ml/__init__.py src/data/__init__.py
touch api/__init__.py tests/__init__.py ml_models/.gitkeep
```

### PHASE 2: Create Files (Priority Order)

**Tier 1 - Core (Required for app to work):**
1. `requirements.txt`
2. `src/config.py`
3. `src/models/player.py`
4. `src/models/team.py`
5. `src/models/match.py`
6. `src/scoring/calculator.py`
7. `src/optimizer/team_builder.py`
8. `src/ml/features.py`
9. `src/ml/predictor.py`
10. `src/ml/train.py`
11. `src/data/sample_data.py`

**Tier 2 - API & Frontend:**
12. `api/schemas.py`
13. `api/main.py`
14. `app/streamlit_app.py`

**Tier 3 - DevOps & Docs:**
15. `Dockerfile`
16. `docker-compose.yml`
17. `.gitignore`
18. `.env.example`
19. `README.md`

**Tier 4 - Quality:**
20. `tests/test_scoring.py`
21. `tests/test_optimizer.py`
22. `tests/test_api.py`

### PHASE 3: Validate
```bash
# Install
pip install -r requirements.txt

# Train ML model (IMPORTANT - must run before API/Streamlit)
python -m src.ml.train

# Test
pytest tests/ -v

# Start API
uvicorn api.main:app --port 8000 &

# Verify API
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Start Streamlit
streamlit run app/streamlit_app.py --server.port 8501
```

### PHASE 4: Commit
```bash
git add .
git commit -m "Transform to CricOptima - AI-powered fantasy cricket optimizer

- ML player prediction (Gradient Boosting)
- Constraint-based team optimization
- FastAPI REST API
- Streamlit dashboard
- Docker deployment
- Original files preserved in legacy/"

git push origin master
```

---

## EXPECTED FINAL STRUCTURE
```
CricOptima/
├── src/
│   ├── models/        (player.py, team.py, match.py)
│   ├── scoring/       (calculator.py)
│   ├── optimizer/     (team_builder.py)
│   ├── ml/            (features.py, predictor.py, train.py)
│   └── data/          (sample_data.py)
├── api/               (main.py, schemas.py)
├── app/               (streamlit_app.py)
├── tests/             (test_*.py)
├── ml_models/         (trained model files)
├── legacy/            (original project files)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## SUCCESS CRITERIA
- [ ] Repository renamed to CricOptima
- [ ] `python -m src.ml.train` completes successfully
- [ ] `pytest tests/ -v` - all tests pass
- [ ] API responds at http://localhost:8000/docs
- [ ] Streamlit loads at http://localhost:8501
- [ ] Team optimization returns 11 players within budget
- [ ] Docker builds without errors
- [ ] Original files preserved in `legacy/`

---

## KEY FEATURES TO VERIFY

### ML Predictions Working:
```bash
curl http://localhost:8000/predictions?top_n=5
# Should return players with predicted_points > 0
```

### Team Optimization Working:
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"budget": 1000}'
# Should return team with 11 players, total_cost <= 1000
```

---

## REFERENCE
See `CricOptima_Upgrade_Guide.md` for complete file contents.

---

## NOTES FOR AI AGENT
1. The ML model MUST be trained before starting API/Streamlit
2. Sample data includes real IPL player names for realism
3. Original scoring logic from `match_evaluation.py` is preserved in `src/scoring/calculator.py`
4. All original files go to `legacy/` folder - don't delete them
5. Keep existing `images/` folder for README screenshots
