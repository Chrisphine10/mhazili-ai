"""
File Manager Service for AI Desktop Agent
Handles file and directory operations with safety checks and logging
"""

import os
import shutil
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import subprocess
import platform
import stat
import hashlib
import json
from dataclasses import dataclass, asdict
import threading
import time

# Project imports
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FileInfo:
    """Information about a file or directory"""
    name: str
    path: str  
    size: Optional[int] = None
    modified: Optional[datetime] = None
    created: Optional[datetime] = None
    is_directory: bool = False
    is_hidden: bool = False
    permissions: Optional[str] = None
    mime_type: Optional[str] = None
    extension: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class FileOperation:
    """Record of file operation for logging and undo"""
    operation: str  # create, delete, move, copy, rename
    source_path: str
    target_path: Optional[str] = None
    timestamp: datetime = None
    success: bool = False
    error_message: Optional[str] = None
    backup_path: Optional[str] = None

class FileManager:
    """
    Comprehensive file management service with safety features
    Handles navigation, CRUD operations, search, and file monitoring
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize file manager with configuration"""
        self.config = self._load_config(config_path)
        
        # Safety settings
        self.safe_mode = self.config.get("safe_mode", True)
        self.restricted_paths = set(self.config.get("restricted_paths", [
            "/system", "/windows", "/program files", "/program files (x86)",
            "/boot", "/etc", "/proc", "/sys", "/dev"
        ]))
        self.allowed_extensions = set(self.config.get("allowed_extensions", [
            ".txt", ".doc", ".docx", ".pdf", ".jpg", ".png", ".gif", ".mp4", 
            ".mp3", ".zip", ".json", ".xml", ".csv", ".xlsx", ".py", ".js", ".html"
        ]))
        
        # Working directory and history
        self.current_directory = Path.cwd()
        self.directory_history = [self.current_directory]
        self.history_index = 0
        
        # Operation tracking
        self.operations_log: List[FileOperation] = []
        self.backup_directory = Path(self.config.get("backup_directory", "backups"))
        self.backup_directory.mkdir(exist_ok=True)
        
        # Monitoring and caching
        self.file_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_seconds", 300)  # 5 minutes
        self.monitoring_enabled = self.config.get("enable_monitoring", False)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"File manager initialized. Current directory: {self.current_directory}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('file_manager', {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def _is_safe_path(self, path: Union[str, Path]) -> bool:
        """Check if path is safe to operate on"""
        if not self.safe_mode:
            return True
        
        path_str = str(Path(path).resolve()).lower()
        
        # Check against restricted paths
        for restricted in self.restricted_paths:
            if restricted.lower() in path_str:
                return False
        
        return True
    
    def _is_allowed_extension(self, path: Union[str, Path]) -> bool:
        """Check if file extension is allowed"""
        if not self.safe_mode:
            return True
        
        extension = Path(path).suffix.lower()
        return extension in self.allowed_extensions or extension == ""
    
    def _create_backup(self, path: Union[str, Path]) -> Optional[str]:
        """Create backup of file before destructive operation"""
        try:
            source_path = Path(path)
            if not source_path.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.name}_{timestamp}.backup"
            backup_path = self.backup_directory / backup_name
            
            if source_path.is_file():
                shutil.copy2(source_path, backup_path)
            else:
                shutil.copytree(source_path, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def _log_operation(self, operation: FileOperation):
        """Log file operation"""
        operation.timestamp = datetime.now()
        self.operations_log.append(operation)
        
        # Keep only recent operations
        max_operations = self.config.get("max_operations_log", 1000)
        if len(self.operations_log) > max_operations:
            self.operations_log = self.operations_log[-max_operations:]
    
    def navigate_to(self, path: Union[str, Path]) -> bool:
        """Navigate to specified directory"""
        try:
            target_path = Path(path).resolve()
            
            if not target_path.exists():
                logger.error(f"Path does not exist: {target_path}")
                return False
            
            if not target_path.is_dir():
                logger.error(f"Path is not a directory: {target_path}")
                return False
            
            if not self._is_safe_path(target_path):
                logger.error(f"Access to path denied: {target_path}")
                return False
            
            # Update current directory and history
            self.current_directory = target_path
            
            # Add to history if it's a new location
            if (self.history_index == len(self.directory_history) - 1 and 
                self.directory_history[-1] != target_path):
                self.directory_history.append(target_path)
                self.history_index += 1
            
            logger.info(f"Navigated to: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate to {path}: {e}")
            return False
    
    def go_back(self) -> bool:
        """Go back in directory history"""
        if self.history_index > 0:
            self.history_index -= 1
            self.current_directory = self.directory_history[self.history_index]
            logger.info(f"Went back to: {self.current_directory}")
            return True
        return False
    
    def go_forward(self) -> bool:
        """Go forward in directory history"""
        if self.history_index < len(self.directory_history) - 1:
            self.history_index += 1
            self.current_directory = self.directory_history[self.history_index]  
            logger.info(f"Went forward to: {self.current_directory}")
            return True
        return False
    
    def list_directory(
        self,
        directory: Optional[Union[str, Path]] = None,
        include_hidden: bool = False,
        sort_by: str = "name",
        reverse: bool = False
    ) -> List[FileInfo]:
        """List contents of directory with detailed information"""
        try:
            target_dir = Path(directory) if directory else self.current_directory
            
            if not target_dir.exists() or not target_dir.is_dir():
                logger.error(f"Invalid directory: {target_dir}")
                return []
            
            if not self._is_safe_path(target_dir):
                logger.error(f"Access denied to directory: {target_dir}")
                return []
            
            files = []
            
            for item in target_dir.iterdir():
                try:
                    # Skip hidden files unless requested
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    
                    stat_info = item.stat()
                    
                    file_info = FileInfo(
                        name=item.name,
                        path=str(item),
                        size=stat_info.st_size if item.is_file() else None,
                        modified=datetime.fromtimestamp(stat_info.st_mtime),
                        created=datetime.fromtimestamp(stat_info.st_ctime),
                        is_directory=item.is_dir(),
                        is_hidden=item.name.startswith('.'),
                        permissions=oct(stat_info.st_mode)[-3:],
                        mime_type=mimetypes.guess_type(str(item))[0] if item.is_file() else None,
                        extension=item.suffix.lower() if item.is_file() else None
                    )
                    
                    files.append(file_info)
                    
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not access {item}: {e}")
                    continue
            
            # Sort files
            sort_key_map = {
                "name": lambda x: x.name.lower(),
                "size": lambda x: x.size or 0,
                "modified": lambda x: x.modified or datetime.min,
                "created": lambda x: x.created or datetime.min,
                "extension": lambda x: x.extension or ""
            }
            
            if sort_by in sort_key_map:
                files.sort(key=sort_key_map[sort_by], reverse=reverse)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list directory: {e}")
            return []
    
    def create_file(
        self,
        filename: str,
        content: str = "",
        directory: Optional[Union[str, Path]] = None
    ) -> bool:
        """Create a new file with optional content"""
        try:
            target_dir = Path(directory) if directory else self.current_directory
            file_path = target_dir / filename
            
            if not self._is_safe_path(file_path):
                logger.error(f"Cannot create file in restricted location: {file_path}")
                return False
            
            if not self._is_allowed_extension(file_path):
                logger.error(f"File extension not allowed: {file_path}")
                return False
            
            # Check if file already exists
            if file_path.exists():
                logger.error(f"File already exists: {file_path}")
                return False
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log operation
            operation = FileOperation(
                operation="create",
                source_path=str(file_path),
                success=True
            )
            self._log_operation(operation)
            
            logger.info(f"File created: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create file {filename}: {e}")
            operation = FileOperation(
                operation="create",
                source_path=str(file_path) if 'file_path' in locals() else filename,
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def create_directory(
        self,
        dirname: str,
        directory: Optional[Union[str, Path]] = None
    ) -> bool:
        """Create a new directory"""
        try:
            target_dir = Path(directory) if directory else self.current_directory
            dir_path = target_dir / dirname
            
            if not self._is_safe_path(dir_path):
                logger.error(f"Cannot create directory in restricted location: {dir_path}")
                return False
            
            if dir_path.exists():
                logger.error(f"Directory already exists: {dir_path}")
                return False
            
            dir_path.mkdir(parents=True, exist_ok=False)
            
            # Log operation
            operation = FileOperation(
                operation="create",
                source_path=str(dir_path),
                success=True
            )
            self._log_operation(operation)
            
            logger.info(f"Directory created: {dir_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directory {dirname}: {e}")
            operation = FileOperation(
                operation="create",
                source_path=str(dir_path) if 'dir_path' in locals() else dirname,
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read content from a text file"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File does not exist: {path}")
                return None
            
            if not path.is_file():
                logger.error(f"Path is not a file: {path}")
                return None
            
            if not self._is_safe_path(path):
                logger.error(f"Access denied to file: {path}")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"File read: {path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def write_file(
        self,
        file_path: Union[str, Path],
        content: str,
        create_backup: bool = True
    ) -> bool:
        """Write content to a file, with optional backup"""
        try:
            path = Path(file_path)
            
            if not self._is_safe_path(path):
                logger.error(f"Cannot write to restricted location: {path}")
                return False
            
            if not self._is_allowed_extension(path):
                logger.error(f"File extension not allowed: {path}")
                return False
            
            backup_path = None
            if create_backup and path.exists():
                backup_path = self._create_backup(path)
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log operation
            operation = FileOperation(
                operation="write",
                source_path=str(path),
                success=True,
                backup_path=backup_path
            )
            self._log_operation(operation)
            
            logger.info(f"File written: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            operation = FileOperation(
                operation="write",
                source_path=str(file_path),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def delete_file(self, file_path: Union[str, Path], create_backup: bool = True) -> bool:
        """Delete a file with optional backup"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File does not exist: {path}")
                return False
            
            if not path.is_file():
                logger.error(f"Path is not a file: {path}")
                return False
            
            if not self._is_safe_path(path):
                logger.error(f"Cannot delete file in restricted location: {path}")
                return False
            
            backup_path = None
            if create_backup:
                backup_path = self._create_backup(path)
            
            path.unlink()
            
            # Log operation
            operation = FileOperation(
                operation="delete",
                source_path=str(path),
                success=True,
                backup_path=backup_path
            )
            self._log_operation(operation)
            
            logger.info(f"File deleted: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            operation = FileOperation(
                operation="delete",
                source_path=str(file_path),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def delete_directory(
        self,
        dir_path: Union[str, Path],
        create_backup: bool = True,
        recursive: bool = False
    ) -> bool:
        """Delete a directory with optional backup"""
        try:
            path = Path(dir_path)
            
            if not path.exists():
                logger.error(f"Directory does not exist: {path}")
                return False
            
            if not path.is_dir():
                logger.error(f"Path is not a directory: {path}")
                return False
            
            if not self._is_safe_path(path):
                logger.error(f"Cannot delete directory in restricted location: {path}")
                return False
            
            # Check if directory is empty
            if not recursive and any(path.iterdir()):
                logger.error(f"Directory is not empty: {path}")
                return False
            
            backup_path = None
            if create_backup:
                backup_path = self._create_backup(path)
            
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
            
            # Log operation
            operation = FileOperation(
                operation="delete",
                source_path=str(path),
                success=True,
                backup_path=backup_path
            )
            self._log_operation(operation)
            
            logger.info(f"Directory deleted: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete directory {dir_path}: {e}")
            operation = FileOperation(
                operation="delete",
                source_path=str(dir_path),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def copy_file(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path]
    ) -> bool:
        """Copy a file to a new location"""
        try:
            source = Path(source_path)
            target = Path(target_path)
            
            if not source.exists():
                logger.error(f"Source file does not exist: {source}")
                return False
            
            if not source.is_file():
                logger.error(f"Source is not a file: {source}")
                return False
            
            if not self._is_safe_path(source) or not self._is_safe_path(target):
                logger.error(f"Access denied to source or target path")
                return False
            
            if not self._is_allowed_extension(target):
                logger.error(f"Target file extension not allowed: {target}")
                return False
            
            # Create parent directories if needed
            target.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, target)
            
            # Log operation
            operation = FileOperation(
                operation="copy",
                source_path=str(source),
                target_path=str(target),
                success=True
            )
            self._log_operation(operation)
            
            logger.info(f"File copied: {source} -> {target}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file {source_path} to {target_path}: {e}")
            operation = FileOperation(
                operation="copy",
                source_path=str(source_path),
                target_path=str(target_path),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def move_file(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        create_backup: bool = True
    ) -> bool:
        """Move/rename a file"""
        try:
            source = Path(source_path)
            target = Path(target_path)
            
            if not source.exists():
                logger.error(f"Source file does not exist: {source}")
                return False
            
            if not self._is_safe_path(source) or not self._is_safe_path(target):
                logger.error(f"Access denied to source or target path")
                return False
            
            if not self._is_allowed_extension(target):
                logger.error(f"Target file extension not allowed: {target}")
                return False
            
            backup_path = None
            if create_backup and target.exists():
                backup_path = self._create_backup(target)
            
            # Create parent directories if needed
            target.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(target))
            
            # Log operation
            operation = FileOperation(
                operation="move",
                source_path=str(source),
                target_path=str(target),
                success=True,
                backup_path=backup_path
            )
            self._log_operation(operation)
            
            logger.info(f"File moved: {source} -> {target}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file {source_path} to {target_path}: {e}")
            operation = FileOperation(
                operation="move",
                source_path=str(source_path),
                target_path=str(target_path),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def search_files(
        self,
        pattern: str,
        directory: Optional[Union[str, Path]] = None,
        recursive: bool = True,
        case_sensitive: bool = False,
        search_content: bool = False
    ) -> List[FileInfo]:
        """Search for files by name pattern or content"""
        try:
            search_dir = Path(directory) if directory else self.current_directory
            
            if not search_dir.exists() or not search_dir.is_dir():
                logger.error(f"Invalid search directory: {search_dir}")
                return []
            
            if not self._is_safe_path(search_dir):
                logger.error(f"Access denied to search directory: {search_dir}")
                return []
            
            results = []
            search_pattern = pattern if case_sensitive else pattern.lower()
            
            def search_in_directory(dir_path: Path):
                try:
                    for item in dir_path.iterdir():
                        if not self._is_safe_path(item):
                            continue
                        
                        # Search by filename
                        item_name = item.name if case_sensitive else item.name.lower()
                        if search_pattern in item_name:
                            stat_info = item.stat()
                            file_info = FileInfo(
                                name=item.name,
                                path=str(item),
                                size=stat_info.st_size if item.is_file() else None,
                                modified=datetime.fromtimestamp(stat_info.st_mtime),
                                created=datetime.fromtimestamp(stat_info.st_ctime),
                                is_directory=item.is_dir(),
                                is_hidden=item.name.startswith('.'),
                                permissions=oct(stat_info.st_mode)[-3:],
                                mime_type=mimetypes.guess_type(str(item))[0] if item.is_file() else None,
                                extension=item.suffix.lower() if item.is_file() else None
                            )
                            results.append(file_info)
                        
                        # Search content if requested
                        if search_content and item.is_file() and self._is_allowed_extension(item):
                            try:
                                with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    content_search = content if case_sensitive else content.lower()
                                    if search_pattern in content_search:
                                        # Add to results if not already found by filename
                                        if not any(r.path == str(item) for r in results):
                                            stat_info = item.stat()
                                            file_info = FileInfo(
                                                name=item.name,
                                                path=str(item),
                                                size=stat_info.st_size,
                                                modified=datetime.fromtimestamp(stat_info.st_mtime),
                                                created=datetime.fromtimestamp(stat_info.st_ctime),
                                                is_directory=False,
                                                is_hidden=item.name.startswith('.'),
                                                permissions=oct(stat_info.st_mode)[-3:],
                                                mime_type=mimetypes.guess_type(str(item))[0],
                                                extension=item.suffix.lower()
                                            )
                                            results.append(file_info)
                            except Exception as e:
                                logger.debug(f"Could not search content in {item}: {e}")
                        
                        # Recursive search
                        if recursive and item.is_dir():
                            search_in_directory(item)
                            
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not search in directory {dir_path}: {e}")
            
            search_in_directory(search_dir)
            logger.info(f"Search completed. Found {len(results)} matches for '{pattern}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_file_info(self, file_path: Union[str, Path], calculate_checksum: bool = False) -> Optional[FileInfo]:
        """Get detailed information about a file or directory"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"Path does not exist: {path}")
                return None
            
            if not self._is_safe_path(path):
                logger.error(f"Access denied to path: {path}")
                return None
            
            stat_info = path.stat()
            
            file_info = FileInfo(
                name=path.name,
                path=str(path),
                size=stat_info.st_size if path.is_file() else None,
                modified=datetime.fromtimestamp(stat_info.st_mtime),
                created=datetime.fromtimestamp(stat_info.st_ctime),
                is_directory=path.is_dir(),
                is_hidden=path.name.startswith('.'),
                permissions=oct(stat_info.st_mode)[-3:],
                mime_type=mimetypes.guess_type(str(path))[0] if path.is_file() else None,
                extension=path.suffix.lower() if path.is_file() else None
            )
            
            # Calculate checksum if requested and it's a file
            if calculate_checksum and path.is_file():
                try:
                    with open(path, 'rb') as f:
                        file_hash = hashlib.sha256()
                        for chunk in iter(lambda: f.read(4096), b""):
                            file_hash.update(chunk)
                        file_info.checksum = file_hash.hexdigest()
                except Exception as e:
                    logger.warning(f"Could not calculate checksum for {path}: {e}")
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return None
    
    def open_file_external(self, file_path: Union[str, Path]) -> bool:
        """Open file with default system application"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File does not exist: {path}")
                return False
            
            if not self._is_safe_path(path):
                logger.error(f"Access denied to file: {path}")
                return False
            
            system = platform.system()
            
            if system == "Windows":
                os.startfile(str(path))
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(path)])
            else:  # Linux and others
                subprocess.run(["xdg-open", str(path)])
            
            logger.info(f"Opened file externally: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return False
    
    def get_operations_log(self, limit: Optional[int] = None) -> List[Dict]:
        """Get recent file operations log"""
        operations = self.operations_log[-limit:] if limit else self.operations_log
        return [asdict(op) for op in operations]
    
    def clear_operations_log(self):
        """Clear the operations log"""
        self.operations_log.clear()
        logger.info("Operations log cleared")
    
    def get_current_directory(self) -> str:
        """Get current working directory"""
        return str(self.current_directory)
    
    def get_directory_size(self, directory: Optional[Union[str, Path]] = None) -> int:
        """Calculate total size of directory and its contents"""
        try:
            target_dir = Path(directory) if directory else self.current_directory
            
            if not target_dir.exists() or not target_dir.is_dir():
                logger.error(f"Invalid directory: {target_dir}")
                return 0
            
            if not self._is_safe_path(target_dir):
                logger.error(f"Access denied to directory: {target_dir}")
                return 0
            
            total_size = 0
            
            for item in target_dir.rglob('*'):
                try:
                    if item.is_file() and self._is_safe_path(item):
                        total_size += item.stat().st_size
                except (OSError, PermissionError):
                    continue
            
            return total_size
            
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
            return 0
    
    def cleanup_backups(self, max_age_days: int = 30):
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0
            
            for backup_file in self.backup_directory.iterdir():
                try:
                    if backup_file.is_file():
                        file_modified = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        if file_modified < cutoff_date:
                            backup_file.unlink()
                            removed_count += 1
                            logger.debug(f"Removed old backup: {backup_file}")
                except Exception as e:
                    logger.warning(f"Could not remove backup file {backup_file}: {e}")
            
            logger.info(f"Cleanup completed. Removed {removed_count} old backup files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup backups: {e}")
            return 0
    
    def export_operations_log(self, file_path: Union[str, Path]) -> bool:
        """Export operations log to JSON file"""
        try:
            path = Path(file_path)
            
            if not self._is_safe_path(path):
                logger.error(f"Cannot export to restricted location: {path}")
                return False
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            log_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_operations": len(self.operations_log),
                "operations": [asdict(op) for op in self.operations_log]
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            logger.info(f"Operations log exported to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export operations log: {e}")
            return False
    
    def import_operations_log(self, file_path: Union[str, Path]) -> bool:
        """Import operations log from JSON file"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"Import file does not exist: {path}")
                return False
            
            if not self._is_safe_path(path):
                logger.error(f"Access denied to import file: {path}")
                return False
            
            with open(path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            imported_operations = []
            for op_dict in log_data.get("operations", []):
                try:
                    # Convert timestamp string back to datetime
                    if op_dict.get("timestamp"):
                        op_dict["timestamp"] = datetime.fromisoformat(op_dict["timestamp"])
                    
                    operation = FileOperation(**op_dict)
                    imported_operations.append(operation)
                except Exception as e:
                    logger.warning(f"Could not import operation: {e}")
                    continue
            
            self.operations_log.extend(imported_operations)
            logger.info(f"Imported {len(imported_operations)} operations from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import operations log: {e}")
            return False
    
    def validate_path_security(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Validate path security and return detailed analysis"""
        try:
            path_obj = Path(path).resolve()
            
            analysis = {
                "path": str(path_obj),
                "exists": path_obj.exists(),
                "is_safe": self._is_safe_path(path_obj),
                "is_allowed_extension": self._is_allowed_extension(path_obj),
                "is_restricted": False,
                "restricted_reason": None,
                "permissions": None,
                "warnings": []
            }
            
            # Check against restricted paths
            path_str = str(path_obj).lower()
            for restricted in self.restricted_paths:
                if restricted.lower() in path_str:
                    analysis["is_restricted"] = True
                    analysis["restricted_reason"] = f"Path contains restricted directory: {restricted}"
                    break
            
            # Get permissions if file exists
            if path_obj.exists():
                try:
                    stat_info = path_obj.stat()
                    analysis["permissions"] = oct(stat_info.st_mode)[-3:]
                except Exception as e:
                    analysis["warnings"].append(f"Could not read permissions: {e}")
            
            # Additional security checks
            if path_obj.exists():
                if path_obj.is_symlink():
                    analysis["warnings"].append("Path is a symbolic link")
                
                if path_obj.name.startswith('.'):
                    analysis["warnings"].append("Path is hidden")
            
            return analysis
            
        except Exception as e:
            return {
                "path": str(path),
                "error": str(e),
                "is_safe": False
            }
    
    def batch_operation(
        self,
        operation: str,
        file_paths: List[Union[str, Path]],
        target_directory: Optional[Union[str, Path]] = None,
        create_backup: bool = True
    ) -> Dict[str, Any]:
        """Perform batch operations on multiple files"""
        results = {
            "successful": [],
            "failed": [],
            "total": len(file_paths),
            "operation": operation
        }
        
        try:
            with self.lock:
                for file_path in file_paths:
                    try:
                        success = False
                        
                        if operation == "delete":
                            success = self.delete_file(file_path, create_backup)
                        elif operation == "copy" and target_directory:
                            target_path = Path(target_directory) / Path(file_path).name
                            success = self.copy_file(file_path, target_path)
                        elif operation == "move" and target_directory:
                            target_path = Path(target_directory) / Path(file_path).name
                            success = self.move_file(file_path, target_path, create_backup)
                        else:
                            results["failed"].append({
                                "path": str(file_path),
                                "error": "Unsupported operation or missing target directory"
                            })
                            continue
                        
                        if success:
                            results["successful"].append(str(file_path))
                        else:
                            results["failed"].append({
                                "path": str(file_path),
                                "error": "Operation failed (check logs for details)"
                            })
                            
                    except Exception as e:
                        results["failed"].append({
                            "path": str(file_path),
                            "error": str(e)
                        })
            
            logger.info(f"Batch {operation} completed: {len(results['successful'])} successful, {len(results['failed'])} failed")
            return results
            
        except Exception as e:
            logger.error(f"Batch operation failed: {e}")
            results["error"] = str(e)
            return results
    
    def get_disk_usage(self, path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
        """Get disk usage statistics for a path"""
        try:
            target_path = Path(path) if path else self.current_directory
            
            if not target_path.exists():
                logger.error(f"Path does not exist: {target_path}")
                return {}
            
            if not self._is_safe_path(target_path):
                logger.error(f"Access denied to path: {target_path}")
                return {}
            
            # Get disk usage
            disk_usage = shutil.disk_usage(target_path)
            
            return {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "path": str(target_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {}
    
    def create_archive(
        self,
        archive_path: Union[str, Path],
        source_paths: List[Union[str, Path]],
        archive_format: str = "zip"
    ) -> bool:
        """Create archive from multiple files/directories"""
        try:
            archive_path_obj = Path(archive_path)
            
            if not self._is_safe_path(archive_path_obj):
                logger.error(f"Cannot create archive in restricted location: {archive_path_obj}")
                return False
            
            # Validate all source paths
            valid_sources = []
            for source_path in source_paths:
                source = Path(source_path)
                if source.exists() and self._is_safe_path(source):
                    valid_sources.append(source)
                else:
                    logger.warning(f"Skipping invalid source path: {source}")
            
            if not valid_sources:
                logger.error("No valid source paths found")
                return False
            
            # Create parent directories if needed
            archive_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            if archive_format.lower() == "zip":
                import zipfile
                with zipfile.ZipFile(archive_path_obj, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for source in valid_sources:
                        if source.is_file():
                            zipf.write(source, source.name)
                        elif source.is_dir():
                            for file_path in source.rglob('*'):
                                if file_path.is_file() and self._is_safe_path(file_path):
                                    arcname = file_path.relative_to(source.parent)
                                    zipf.write(file_path, arcname)
            
            elif archive_format.lower() in ["tar", "tar.gz", "tgz"]:
                import tarfile
                mode = "w:gz" if archive_format.lower() in ["tar.gz", "tgz"] else "w"
                with tarfile.open(archive_path_obj, mode) as tarf:
                    for source in valid_sources:
                        tarf.add(source, arcname=source.name)
            
            else:
                logger.error(f"Unsupported archive format: {archive_format}")
                return False
            
            # Log operation
            operation = FileOperation(
                operation="create_archive",
                source_path=str(valid_sources[0]) if valid_sources else "",
                target_path=str(archive_path_obj),
                success=True
            )
            self._log_operation(operation)
            
            logger.info(f"Archive created: {archive_path_obj}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            operation = FileOperation(
                operation="create_archive",
                source_path=str(source_paths[0]) if source_paths else "",
                target_path=str(archive_path),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def extract_archive(
        self,
        archive_path: Union[str, Path],
        extract_to: Optional[Union[str, Path]] = None
    ) -> bool:
        """Extract archive to specified directory"""
        try:
            archive_path_obj = Path(archive_path)
            
            if not archive_path_obj.exists():
                logger.error(f"Archive does not exist: {archive_path_obj}")
                return False
            
            if not self._is_safe_path(archive_path_obj):
                logger.error(f"Access denied to archive: {archive_path_obj}")
                return False
            
            extract_dir = Path(extract_to) if extract_to else self.current_directory
            
            if not self._is_safe_path(extract_dir):
                logger.error(f"Cannot extract to restricted location: {extract_dir}")
                return False
            
            # Create extraction directory if needed
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine archive type and extract
            if archive_path_obj.suffix.lower() == ".zip":
                import zipfile
                with zipfile.ZipFile(archive_path_obj, 'r') as zipf:
                    zipf.extractall(extract_dir)
            
            elif archive_path_obj.suffix.lower() in [".tar", ".gz", ".tgz"]:
                import tarfile
                with tarfile.open(archive_path_obj, 'r:*') as tarf:
                    tarf.extractall(extract_dir)
            
            else:
                logger.error(f"Unsupported archive format: {archive_path_obj.suffix}")
                return False
            
            # Log operation
            operation = FileOperation(
                operation="extract_archive",
                source_path=str(archive_path_obj),
                target_path=str(extract_dir),
                success=True
            )
            self._log_operation(operation)
            
            logger.info(f"Archive extracted: {archive_path_obj} -> {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract archive: {e}")
            operation = FileOperation(
                operation="extract_archive",
                source_path=str(archive_path),
                target_path=str(extract_to) if extract_to else str(self.current_directory),
                success=False,
                error_message=str(e)
            )
            self._log_operation(operation)
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information related to file operations"""
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "current_directory": str(self.current_directory),
                "safe_mode": self.safe_mode,
                "restricted_paths": list(self.restricted_paths),
                "allowed_extensions": list(self.allowed_extensions),
                "backup_directory": str(self.backup_directory),
                "operations_count": len(self.operations_log),
                "cache_size": len(self.file_cache),
                "monitoring_enabled": self.monitoring_enabled
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """Clean shutdown of file manager"""
        try:
            # Clear cache
            self.file_cache.clear()
            
            # Export operations log if configured
            if self.config.get("export_log_on_shutdown", False):
                log_file = self.backup_directory / f"operations_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.export_operations_log(log_file)
            
            # Cleanup old backups
            if self.config.get("cleanup_on_shutdown", True):
                self.cleanup_backups()
            
            logger.info("File manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Utility functions for external use
def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def is_text_file(file_path: Union[str, Path]) -> bool:
    """Check if file is likely a text file"""
    try:
        path = Path(file_path)
        
        # Check by extension first
        text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml', 
                          '.md', '.rst', '.log', '.cfg', '.ini', '.conf', '.csv', '.sql'}
        
        if path.suffix.lower() in text_extensions:
            return True
        
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type.startswith('text/'):
            return True
        
        # Check file content (first 1024 bytes)
        if path.exists() and path.is_file():
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' not in chunk:  # No null bytes usually means text
                    return True
        
        return False
        
    except Exception:
        return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize file manager
    fm = FileManager()
    
    # Example operations
    print(f"Current directory: {fm.get_current_directory()}")
    
    # List current directory
    files = fm.list_directory()
    print(f"Found {len(files)} items in current directory")
    
    # Create a test file
    if fm.create_file("test_file.txt", "Hello, World!"):
        print("Test file created successfully")
    
    # Read the file
    content = fm.read_file("test_file.txt")
    if content:
        print(f"File content: {content}")
    
    # Get file info
    file_info = fm.get_file_info("test_file.txt", calculate_checksum=True)
    if file_info:
        print(f"File info: {file_info}")
    
    # Search for files
    results = fm.search_files("test", search_content=True)
    print(f"Search found {len(results)} matches")
    
    # Clean up
    fm.delete_file("test_file.txt")
    fm.shutdown()