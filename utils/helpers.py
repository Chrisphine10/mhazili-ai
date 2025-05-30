"""
Helper utilities and common functions for the project.
"""
import os
import json
import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry a function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read JSON file and return parsed data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Write data to JSON file."""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Read text file and return content."""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_text(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> None:
    """Write text content to file."""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension without the dot."""
    return Path(file_path).suffix.lstrip('.')


def is_file_older_than(file_path: Union[str, Path], hours: int) -> bool:
    """Check if file is older than specified hours."""
    file_time = Path(file_path).stat().st_mtime
    current_time = time.time()
    return (current_time - file_time) > (hours * 3600)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with dot notation support."""
    keys = key.split('.')
    value = dictionary
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def timestamp(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime(format_string)


def format_bytes(bytes_size: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Get environment variable with optional default and requirement check."""
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    
    return value


def validate_email(email: str) -> bool:
    """Basic email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def clean_string(text: str, remove_chars: str = None) -> str:
    """Clean string by removing specified characters and extra whitespace."""
    if remove_chars:
        for char in remove_chars:
            text = text.replace(char, '')
    
    # Remove extra whitespace
    return ' '.join(text.split())


class ConfigManager:
    """Simple configuration management class."""
    
    def __init__(self, config_file: Union[str, Path] = "config.json"):
        self.config_file = Path(config_file)
        self._config = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            self._config = read_json(self.config_file)
    
    def save(self) -> None:
        """Save configuration to file."""
        write_json(self._config, self.config_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return safe_get(self._config, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save()


# Example usage functions
def setup_project_structure(project_name: str) -> None:
    """Create standard project directory structure."""
    dirs = [
        f"{project_name}/src",
        f"{project_name}/tests",
        f"{project_name}/docs",
        f"{project_name}/data",
        f"{project_name}/logs",
        f"{project_name}/config"
    ]
    
    for dir_path in dirs:
        ensure_dir(dir_path)
        print(f"Created directory: {dir_path}")


# Commonly used constants
DEFAULT_ENCODING = 'utf-8'
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"