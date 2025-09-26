#!/usr/bin/env python3
"""
Main entry point for the Hybrid Identity Monitoring & Deepfake-Resistant Verification System
"""
import sys
import os
import argparse
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings
from models.database import create_tables
from services.deepfake_detector import DeepfakeDetector
from services.identity_monitor import IdentityMonitor
from services.continuous_monitoring import ContinuousIdentityMonitor
from deployment.hybrid_deployment import HybridDeploymentManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/identity_monitoring.log')
        ]
    )

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'data/videos',
        'data/images',
        'data/models',
        'logs',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

async def initialize_services():
    """Initialize all services"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize services
        deepfake_detector = DeepfakeDetector(settings.deepfake_model_path)
        identity_monitor = IdentityMonitor()
        continuous_monitor = ContinuousIdentityMonitor(identity_monitor, deepfake_detector)
        hybrid_deployment = HybridDeploymentManager()
        
        # Initialize hybrid deployment
        await hybrid_deployment.initialize_hybrid_deployment()
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        return False

def run_api_server():
    """Run the FastAPI server"""
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

def run_dashboard():
    """Run the Streamlit dashboard"""
    import subprocess
    import sys
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_dashboard.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def run_tests():
    """Run the test suite"""
    import subprocess
    import sys
    
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hybrid Identity Monitoring & Deepfake-Resistant Verification System"
    )
    
    parser.add_argument(
        "command",
        choices=["api", "dashboard", "test", "init", "all"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help="Host to bind the API server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Port to bind the API server to"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    create_directories()
    
    if args.command == "init":
        logger.info("Initializing services...")
        success = asyncio.run(initialize_services())
        if success:
            logger.info("Initialization completed successfully")
        else:
            logger.error("Initialization failed")
            sys.exit(1)
    
    elif args.command == "api":
        logger.info("Starting API server...")
        run_api_server()
    
    elif args.command == "dashboard":
        logger.info("Starting dashboard...")
        run_dashboard()
    
    elif args.command == "test":
        logger.info("Running tests...")
        run_tests()
    
    elif args.command == "all":
        logger.info("Starting all services...")
        
        # Initialize services first
        success = asyncio.run(initialize_services())
        if not success:
            logger.error("Failed to initialize services")
            sys.exit(1)
        
        # Start both API and dashboard
        import threading
        
        # Start API server in a thread
        api_thread = threading.Thread(target=run_api_server)
        api_thread.daemon = True
        api_thread.start()
        
        # Start dashboard in main thread
        run_dashboard()

if __name__ == "__main__":
    main()
