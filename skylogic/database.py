"""
SkyLogic MAS — Database Engine & Session Management
Supports SQLite (default) and PostgreSQL/PostGIS.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from skylogic.config import settings

# SQLite needs check_same_thread=False for FastAPI
connect_args = {}
engine_kwargs = {
    "echo": False,
}

if settings.is_sqlite:
    connect_args["check_same_thread"] = False
    engine_kwargs["connect_args"] = connect_args
else:
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 10
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(settings.database_url, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called on app startup."""
    from skylogic.models import user, patch, annotation  # noqa: F401
    Base.metadata.create_all(bind=engine)
