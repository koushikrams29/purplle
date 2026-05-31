"""
Store Intelligence API
Main FastAPI application
"""

from fastapi import FastAPI
from datetime import datetime
from app.config import settings
from app.database import init_db

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description="Real-time retail analytics from CCTV footage",
    version=settings.api_version,
    debug=settings.debug
)


@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("🚀 Store Intelligence API starting...")
    print(f"   Environment: {settings.environment}")
    print(f"   Database: {settings.database_url}")
    print(f"   YOLO Model: {settings.yolo_model}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    print("🛑 Store Intelligence API shutting down...")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "store-intelligence-api",
        "version": settings.api_version,
        "environment": settings.environment
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Store Intelligence API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )