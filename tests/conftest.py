"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.main import app
from app.database import Base


# Test database setup
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()
    yield db_session
    db_session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """Create test client"""
    def override_get_db():
        try:
            yield db
        finally:
            db.close()
    
    from app.main import get_db
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def sample_event():
    """Sample event for testing"""
    return {
        "event_id": "test-event-001",
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_000001",
        "event_type": "ENTRY",
        "timestamp": "2026-03-03T14:00:00Z",
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.95,
        "metadata": {}
    }


@pytest.fixture(scope="function")
def sample_store_layout():
    """Sample store layout for testing"""
    return {
        "store_id": "STORE_BLR_002",
        "store_name": "Brigade Road - Bangalore",
        "city": "Bangalore",
        "zones": {
            "ENTRY": {
                "name": "Entry/Exit Threshold",
                "polygon": [(100, 50), (150, 50), (150, 100), (100, 100)]
            },
            "MAIN_FLOOR": {
                "name": "Main Shopping Floor",
                "polygon": [(200, 100), (800, 100), (800, 500), (200, 500)]
            },
            "SKINCARE": {
                "name": "Skincare Zone",
                "polygon": [(200, 150), (400, 150), (400, 300), (200, 300)]
            },
            "BILLING": {
                "name": "Billing Counter Area",
                "polygon": [(600, 350), (800, 350), (800, 500), (600, 500)]
            }
        },
        "camera_zones": {
            "CAM_ENTRY_01": ["ENTRY"],
            "CAM_MAIN_FLOOR_01": ["MAIN_FLOOR", "SKINCARE", "BILLING"],
            "CAM_BILLING_01": ["BILLING"]
        }
    }