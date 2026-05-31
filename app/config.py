"""
Application Configuration
Load settings from environment variables
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings from .env file"""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Store Intelligence API"
    api_version: str = "1.0.0"
    
    # Database
    database_url: str = "sqlite:///./store.db"
    
    # Detection
    yolo_model: str = "yolov8m"
    detection_confidence: float = 0.5
    dwell_interval_ms: int = 30000
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Store
    default_store_id: str = "STORE_BLR_002"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()