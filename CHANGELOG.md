# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-24

### Added
- **Live Data Integration**: Real-time player data fetching via CricAPI (#11).
- **Demo Mode**: "Transparent Demo Mode" simulation for testing without API keys (#58).
- **Deployment**: Docker support and GitHub Actions CI/CD pipeline (#15, #27).
- **Mobile Support**: Fully responsive Streamlit UI (#22).
- **User Auth**: Secure JWT-based authentication with bcrypt hashing (#34).
- **Team Management**: Functionality to save and load optimized teams.
- **Comparison**: Head-to-Head win probability predictions (#37).
- **Simulation**: Monte Carlo simulation (1000x matches) for variance analysis (#40).

### Fixed
- Registration error handling (#63).
- Database initialization errors on startup (#65).
- Team saving serialization issues (#67).
- Head-to-Head fallback logic (#70).
- Variable costs in demo simulation (#59).
- Plotly scatter plots crashing on empty player pools (#48).
- API key passing and type annotation fixes (#51, #52).

### Changed
- Bumped `actions/setup-python`, `actions/checkout` and `actions/labeler` versions.
- Optimized Docker cache invalidation for Streamlit Cloud.
- Refactored CSS to external file.

## [1.0.0] - 2025-12-19

### Added
- Initial release of CricOptima platform.
- Core ML models for player performance prediction (Gradient Boosting).
- Linear calibration for optimization constraints.
- FastAPI backend with Swagger documentation.
- Basic Streamlit dashboard for team visualization.
- Project restructuring and modular architecture.
- Unit tests and initial test suite.
