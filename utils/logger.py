import logging
import sys
from datetime import datetime
from pathlib import Path
import os


def get_logger(name=None, level=logging.INFO, log_file=None, format_string=None):
    """
    Create and configure a logger instance.
    
    Args:
        name (str, optional): Logger name. Defaults to calling module name.
        level (int, optional): Logging level. Defaults to INFO.
        log_file (str, optional): Path to log file. If None, logs to console only.
        format_string (str, optional): Custom format string for log messages.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Use provided name or get the calling module's name
    if name is None:
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'root')
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Reset handlers if they exist
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Default format string
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure the log file is writable
            if log_path.exists() and not os.access(log_path, os.W_OK):
                logger.warning(f"Log file {log_file} is not writable. Falling back to console logging only.")
            else:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}. Falling back to console logging only.")
    
    return logger


def setup_project_logger(project_name, log_dir="logs", level=logging.INFO):
    """
    Set up a project-wide logger with file output.
    
    Args:
        project_name (str): Name of the project
        log_dir (str): Directory to store log files
        level (int): Logging level
    
    Returns:
        logging.Logger: Configured project logger
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"{project_name}_{timestamp}.log"
        
        return get_logger(
            name=project_name,
            level=level,
            log_file=str(log_file)
        )
    except Exception as e:
        # Fallback to console-only logging if file setup fails
        print(f"Warning: Failed to set up file logging: {e}. Using console logging only.")
        return get_logger(name=project_name, level=level)


# Pre-configured loggers for common use cases
def get_debug_logger(name=None):
    """Get a logger configured for debug output."""
    return get_logger(name=name, level=logging.DEBUG)


def get_error_logger(name=None, log_file="logs/errors.log"):
    """Get a logger configured for error logging."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return get_logger(
            name=name,
            level=logging.ERROR,
            log_file=log_file,
            format_string='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
    except Exception as e:
        print(f"Warning: Failed to set up error logging: {e}. Using console logging only.")
        return get_logger(
            name=name,
            level=logging.ERROR,
            format_string='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )