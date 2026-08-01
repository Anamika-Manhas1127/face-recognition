from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.backend.config.settings import settings

# Create engine with connect_args for SQLite multi-thread compatibility
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency injection function to yield database session and close it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
