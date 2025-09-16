"""
Logging setup for the email classifier app.
"""
import logging
import os
from datetime import datetime

def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Returns a logger with console and file output.
    """
    # Make logs folder if missing
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Skip if already configured
    if logger.handlers:
        return logger
    
    # File handler with daily log file
    log_filename = f"logs/{datetime.now().strftime('%Y%m%d')}_spam_detector.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for live messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Common format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
