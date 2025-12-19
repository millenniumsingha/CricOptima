from sqlalchemy import create_engine, Column, Integer, String, JSON, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

# Database URL
DATABASE_URL = "sqlite:///./cricoptima.db"

# Setup Engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Session Local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base Model
Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # Relationships
    teams = relationship("SavedTeam", back_populates="owner")


class SavedTeam(Base):
    """Model to store optimized teams."""
    __tablename__ = "saved_teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="My Dream XI")
    team_data = Column(JSON, nullable=False)  # Stores the full team JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    owner = relationship("User", back_populates="teams")


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
