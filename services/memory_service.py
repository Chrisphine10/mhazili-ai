"""
Memory Service - Persistent state management and learning system
Handles task memory, user preferences, execution history, and pattern recognition
for improved automation performance over time.
"""

import os
import json
import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import sqlite3
from enum import Enum


class MemoryType(Enum):
    """Types of memory entries."""
    TASK_EXECUTION = "task_execution"
    USER_CORRECTION = "user_correction"
    SYSTEM_PATTERN = "system_pattern"
    APPLICATION_STATE = "application_state"
    SUCCESSFUL_TASK = "successful_task"
    FAILED_TASK = "failed_task"
    USER_PREFERENCE = "user_preference"


class TaskStatus(Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MemoryEntry:
    """Base structure for memory entries."""
    id: str
    memory_type: MemoryType
    timestamp: float
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: float = None
    expiry_date: Optional[float] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


@dataclass
class TaskMemory:
    """Memory structure for task execution history."""
    task_id: str
    user_input: str
    intent_analysis: Dict[str, Any]
    execution_steps: List[Dict[str, Any]]
    status: TaskStatus
    start_time: float
    end_time: Optional[float] = None
    success_rate: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    user_feedback: Optional[str] = None
    corrections_applied: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get task execution duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def is_successful(self) -> bool:
        """Check if task was successful."""
        return self.status == TaskStatus.COMPLETED and self.success_rate > 0.7


@dataclass
class UserCorrection:
    """Structure for user corrections and feedback."""
    correction_id: str
    original_task_id: str
    step_number: int
    original_action: Dict[str, Any]
    corrected_action: Dict[str, Any]
    user_explanation: str
    timestamp: float
    applied: bool = False
    effectiveness_score: Optional[float] = None


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def save_entry(self, entry: MemoryEntry) -> bool:
        """Save a memory entry."""
        pass
    
    @abstractmethod
    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        pass
    
    @abstractmethod
    async def search_entries(self, **kwargs) -> List[MemoryEntry]:
        """Search memory entries by criteria."""
        pass
    
    @abstractmethod
    async def delete_entry(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        pass


class JSONMemoryStorage(MemoryStorage):
    """JSON file-based memory storage."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()  # Use asyncio lock instead of threading lock
        self.logger = logging.getLogger(f"{__name__}.JSONMemoryStorage")
        
        # Initialize file if it doesn't exist or is corrupted
        if not self.file_path.exists() or not self._is_valid_json():
            self._save_data({})
    
    def _is_valid_json(self) -> bool:
        """Check if the JSON file is valid."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
    
    def _load_data(self) -> Dict[str, Dict]:
        """Load data from JSON file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading memory data: {e}")
            # If file is corrupted, backup and create new
            if self.file_path.exists():
                backup_path = self.file_path.with_suffix('.json.bak')
                try:
                    self.file_path.rename(backup_path)
                    self.logger.info(f"Backed up corrupted memory file to {backup_path}")
                except Exception as backup_error:
                    self.logger.error(f"Failed to backup corrupted file: {backup_error}")
            return {}
    
    def _save_data(self, data: Dict[str, Dict]) -> bool:
        """Save data to JSON file."""
        try:
            # Create a temporary file
            temp_path = self.file_path.with_suffix('.json.tmp')
            
            # Write to temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # If write was successful, replace the original file
            temp_path.replace(self.file_path)
            return True
        except Exception as e:
            self.logger.error(f"Error saving memory data: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    self.logger.error(f"Failed to clean up temp file: {cleanup_error}")
            return False
    
    async def save_entry(self, entry: MemoryEntry) -> bool:
        """Save a memory entry to JSON file."""
        async with self.lock:
            data = self._load_data()
            # Convert entry to dict and handle enum serialization
            entry_dict = asdict(entry)
            entry_dict['memory_type'] = entry_dict['memory_type'].value  # Convert MemoryType to string
            data[entry.id] = entry_dict
            return self._save_data(data)
    
    async def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        async with self.lock:
            data = self._load_data()
            entry_data = data.get(entry_id)
            
            if entry_data:
                # Update access tracking
                entry_data['access_count'] += 1
                entry_data['last_accessed'] = time.time()
                self._save_data(data)
                
                # Convert back to MemoryEntry
                entry_data['memory_type'] = MemoryType(entry_data['memory_type'])
                return MemoryEntry(**entry_data)
            
            return None
    
    async def search_entries(self, **kwargs) -> List[MemoryEntry]:
        """Search memory entries by criteria."""
        async with self.lock:
            data = self._load_data()
            results = []
            
            for entry_data in data.values():
                match = True
                
                # Filter by memory type
                if 'memory_type' in kwargs:
                    if entry_data['memory_type'] != kwargs['memory_type'].value:
                        match = False
                
                # Filter by tags
                if 'tags' in kwargs and match:
                    required_tags = kwargs['tags']
                    if not all(tag in entry_data.get('tags', []) for tag in required_tags):
                        match = False
                
                # Filter by time range
                if 'after' in kwargs and match:
                    if entry_data['timestamp'] < kwargs['after']:
                        match = False
                
                if 'before' in kwargs and match:
                    if entry_data['timestamp'] > kwargs['before']:
                        match = False
                
                # Filter by importance threshold
                if 'min_importance' in kwargs and match:
                    if entry_data['importance'] < kwargs['min_importance']:
                        match = False
                
                if match:
                    entry_data['memory_type'] = MemoryType(entry_data['memory_type'])
                    results.append(MemoryEntry(**entry_data))
            
            # Sort by relevance (importance * recency)
            results.sort(key=lambda x: x.importance * (1.0 / (time.time() - x.timestamp + 1)), reverse=True)
            
            return results
    
    async def delete_entry(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        async with self.lock:
            data = self._load_data()
            if entry_id in data:
                del data[entry_id]
                return self._save_data(data)
            return False
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self.lock:
            data = self._load_data()
            current_time = time.time()
            expired_count = 0
            
            entries_to_delete = []
            for entry_id, entry_data in data.items():
                if entry_data.get('expiry_date') and entry_data['expiry_date'] < current_time:
                    entries_to_delete.append(entry_id)
            
            for entry_id in entries_to_delete:
                del data[entry_id]
                expired_count += 1
            
            if expired_count > 0:
                self._save_data(data)
                self.logger.info(f"Cleaned up {expired_count} expired memory entries")
            
            return expired_count


class SQLiteMemoryStorage(MemoryStorage):
    """SQLite database-based memory storage for better performance."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.SQLiteMemoryStorage")
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL,
                    tags TEXT,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL,
                    expiry_date REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance)')
            conn.commit()
    
    def save_entry(self, entry: MemoryEntry) -> bool:
        """Save a memory entry to SQLite database."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO memory_entries 
                        (id, memory_type, timestamp, data, tags, importance, access_count, last_accessed, expiry_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.id,
                        entry.memory_type.value,
                        entry.timestamp,
                        json.dumps(entry.data),
                        json.dumps(entry.tags),
                        entry.importance,
                        entry.access_count,
                        entry.last_accessed,
                        entry.expiry_date
                    ))
                    conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error saving memory entry: {e}")
            return False
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT * FROM memory_entries WHERE id = ?', (entry_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        # Update access tracking
                        conn.execute('''
                            UPDATE memory_entries 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE id = ?
                        ''', (time.time(), entry_id))
                        conn.commit()
                        
                        # Convert row to MemoryEntry
                        return MemoryEntry(
                            id=row[0],
                            memory_type=MemoryType(row[1]),
                            timestamp=row[2],
                            data=json.loads(row[3]),
                            tags=json.loads(row[4] or '[]'),
                            importance=row[5],
                            access_count=row[6] + 1,
                            last_accessed=time.time(),
                            expiry_date=row[8]
                        )
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving memory entry: {e}")
            return None
    
    def search_entries(self, **kwargs) -> List[MemoryEntry]:
        """Search memory entries by criteria."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    query = 'SELECT * FROM memory_entries WHERE 1=1'
                    params = []
                    
                    if 'memory_type' in kwargs:
                        query += ' AND memory_type = ?'
                        params.append(kwargs['memory_type'].value)
                    
                    if 'after' in kwargs:
                        query += ' AND timestamp >= ?'
                        params.append(kwargs['after'])
                    
                    if 'before' in kwargs:
                        query += ' AND timestamp <= ?'
                        params.append(kwargs['before'])
                    
                    if 'min_importance' in kwargs:
                        query += ' AND importance >= ?'
                        params.append(kwargs['min_importance'])
                    
                    query += ' ORDER BY importance * (1.0 / (? - timestamp + 1)) DESC'
                    params.append(time.time())
                    
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        entry = MemoryEntry(
                            id=row[0],
                            memory_type=MemoryType(row[1]),
                            timestamp=row[2],
                            data=json.loads(row[3]),
                            tags=json.loads(row[4] or '[]'),
                            importance=row[5],
                            access_count=row[6],
                            last_accessed=row[7],
                            expiry_date=row[8]
                        )
                        
                        # Filter by tags if specified
                        if 'tags' in kwargs:
                            required_tags = kwargs['tags']
                            if all(tag in entry.tags for tag in required_tags):
                                results.append(entry)
                        else:
                            results.append(entry)
                    
                    return results
        except Exception as e:
            self.logger.error(f"Error searching memory entries: {e}")
            return []
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('DELETE FROM memory_entries WHERE id = ?', (entry_id,))
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Error deleting memory entry: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'DELETE FROM memory_entries WHERE expiry_date IS NOT NULL AND expiry_date < ?',
                        (time.time(),)
                    )
                    conn.commit()
                    expired_count = cursor.rowcount
                    
                    if expired_count > 0:
                        self.logger.info(f"Cleaned up {expired_count} expired memory entries")
                    
                    return expired_count
        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")
            return 0


class MemoryService:
    """
    Main memory service for managing task history, user preferences, and learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        storage_type = config.get('storage_type', 'json')
        if storage_type == 'sqlite':
            db_path = config.get('db_path', 'memory.db')
            self.storage = SQLiteMemoryStorage(db_path)
        else:
            file_path = config.get('file_path', 'memory.json')
            self.storage = JSONMemoryStorage(file_path)
        
        # Memory management settings
        self.max_memory_entries = config.get('max_memory_entries', 10000)
        self.cleanup_interval = config.get('cleanup_interval', 3600)  # 1 hour
        self.default_expiry_days = config.get('default_expiry_days', 30)
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        # Pattern recognition cache
        self.pattern_cache = {}
        self.cache_expiry = time.time() + 300  # 5 minutes
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_old_memories()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def generate_id(self, content: str) -> str:
        """Generate unique ID for memory entry."""
        return hashlib.md5(f"{content}_{time.time()}".encode()).hexdigest()
    
    def store_task_execution(self, task_memory: TaskMemory) -> bool:
        """Store task execution memory."""
        entry = MemoryEntry(
            id=self.generate_id(task_memory.task_id),
            memory_type=MemoryType.TASK_EXECUTION,
            timestamp=task_memory.start_time,
            data=asdict(task_memory),
            tags=[
                task_memory.status.value,
                "task_execution",
                *self._extract_task_tags(task_memory.user_input)
            ],
            importance=self._calculate_task_importance(task_memory),
            expiry_date=time.time() + (self.default_expiry_days * 86400)
        )
        
        return self.storage.save_entry(entry)
    
    async def store_execution(self, task_description: str, result: Dict[str, Any], executor: Optional[str] = None) -> bool:
        """
        Store task execution results in memory.
        
        Args:
            task_description: The original task description
            result: Dictionary containing execution results
            executor: Optional executor type used
            
        Returns:
            bool: Success status
        """
        try:
            # Create task memory entry
            task_memory = TaskMemory(
                task_id=self.generate_id(task_description),
                user_input=task_description,
                intent_analysis={},  # Would be populated by LLM in future
                execution_steps=[{
                    'action': 'execute',
                    'params': {'text': task_description},
                    'result': result
                }],
                status=TaskStatus.COMPLETED if result.get('success') else TaskStatus.FAILED,
                start_time=time.time(),
                end_time=time.time(),
                success_rate=1.0 if result.get('success') else 0.0,
                error_messages=[result.get('message')] if not result.get('success') else []
            )
            
            # Convert task memory to dict and handle enum serialization
            memory_dict = asdict(task_memory)
            memory_dict['status'] = memory_dict['status'].value  # Convert TaskStatus to string
            
            # Create memory entry with string values for enums
            entry = MemoryEntry(
                id=self.generate_id(task_memory.task_id),
                memory_type=MemoryType.TASK_EXECUTION,
                timestamp=task_memory.start_time,
                data=memory_dict,
                tags=[
                    task_memory.status.value,
                    "task_execution",
                    *self._extract_task_tags(task_memory.user_input)
                ],
                importance=self._calculate_task_importance(task_memory),
                expiry_date=time.time() + (self.default_expiry_days * 86400)
            )
            
            # Store in memory using the storage's save_entry method
            return await self.storage.save_entry(entry)
            
        except Exception as e:
            self.logger.error(f"Failed to store execution: {e}")
            return False
    
    def store_user_correction(self, correction: UserCorrection) -> bool:
        """Store user correction for learning."""
        entry = MemoryEntry(
            id=correction.correction_id,
            memory_type=MemoryType.USER_CORRECTION,
            timestamp=correction.timestamp,
            data=asdict(correction),
            tags=["user_correction", "learning", correction.original_task_id],
            importance=0.9,  # High importance for corrections
            expiry_date=time.time() + (90 * 86400)  # 90 days
        )
        
        return self.storage.save_entry(entry)
    
    def store_user_preference(self, preference_key: str, preference_value: Any, 
                            context: Optional[str] = None) -> bool:
        """Store user preference."""
        entry = MemoryEntry(
            id=self.generate_id(f"pref_{preference_key}"),
            memory_type=MemoryType.USER_PREFERENCE,
            timestamp=time.time(),
            data={
                'key': preference_key,
                'value': preference_value,
                'context': context
            },
            tags=["preference", preference_key],
            importance=0.8,
            expiry_date=None  # Preferences don't expire
        )
        
        return self.storage.save_entry(entry)
    
    def get_similar_tasks(self, user_input: str, limit: int = 5) -> List[TaskMemory]:
        """Find similar successful tasks for guidance."""
        # Search for successful task executions
        entries = self.storage.search_entries(
            memory_type=MemoryType.TASK_EXECUTION,
            min_importance=0.6,
            after=time.time() - (30 * 86400)  # Last 30 days
        )
        
        similar_tasks = []
        input_words = set(user_input.lower().split())
        
        for entry in entries[:limit * 2]:  # Get more to filter better
            task_data = entry.data
            if task_data.get('status') == TaskStatus.COMPLETED.value:
                task_words = set(task_data.get('user_input', '').lower().split())
                similarity = len(input_words & task_words) / len(input_words | task_words)
                
                if similarity > 0.3:  # 30% similarity threshold
                    task_memory = TaskMemory(**task_data)
                    similar_tasks.append((task_memory, similarity))
        
        # Sort by similarity and return top results
        similar_tasks.sort(key=lambda x: x[1], reverse=True)
        return [task for task, _ in similar_tasks[:limit]]
    
    def get_user_corrections_for_task(self, task_pattern: str) -> List[UserCorrection]:
        """Get user corrections that might apply to current task."""
        entries = self.storage.search_entries(
            memory_type=MemoryType.USER_CORRECTION,
            min_importance=0.7
        )
        
        corrections = []
        for entry in entries:
            correction = UserCorrection(**entry.data)
            corrections.append(correction)
        
        return corrections
    
    def get_user_preferences(self, context: Optional[str] = None) -> Dict[str, Any]:
        """Get user preferences, optionally filtered by context."""
        entries = self.storage.search_entries(
            memory_type=MemoryType.USER_PREFERENCE
        )
        
        preferences = {}
        for entry in entries:
            pref_data = entry.data
            if not context or pref_data.get('context') == context:
                preferences[pref_data['key']] = pref_data['value']
        
        return preferences
    
    def learn_from_execution(self, task_memory: TaskMemory) -> Dict[str, Any]:
        """Learn patterns from task execution."""
        patterns = {
            'common_failures': self._identify_failure_patterns(task_memory),
            'success_factors': self._identify_success_factors(task_memory),
            'timing_patterns': self._analyze_timing_patterns(task_memory),
            'app_patterns': self._analyze_application_patterns(task_memory)
        }
        
        # Store identified patterns
        if any(patterns.values()):
            pattern_entry = MemoryEntry(
                id=self.generate_id(f"pattern_{task_memory.task_id}"),
                memory_type=MemoryType.SYSTEM_PATTERN,
                timestamp=time.time(),
                data=patterns,
                tags=["pattern", "learning", task_memory.status.value],
                importance=0.7,
                expiry_date=time.time() + (60 * 86400)  # 60 days
            )
            self.storage.save_entry(pattern_entry)
        
        return patterns
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        all_entries = self.storage.search_entries()
        
        stats = {
            'total_entries': len(all_entries),
            'by_type': {},
            'importance_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'recent_activity': 0
        }
        
        recent_threshold = time.time() - 86400  # Last 24 hours
        
        for entry in all_entries:
            # Count by type
            type_name = entry.memory_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1
            
            # Importance distribution
            if entry.importance >= 0.8:
                stats['importance_distribution']['high'] += 1
            elif entry.importance >= 0.5:
                stats['importance_distribution']['medium'] += 1
            else:
                stats['importance_distribution']['low'] += 1
            
            # Recent activity
            if entry.timestamp >= recent_threshold:
                stats['recent_activity'] += 1
        
        return stats
    
    def cleanup_old_memories(self) -> Dict[str, int]:
        """Clean up old and unused memories."""
        results = {
            'expired_removed': 0,
            'low_importance_removed': 0,
            'unused_removed': 0
        }
        
        # Remove expired entries
        results['expired_removed'] = self.storage.cleanup_expired()
        
        # Remove low importance, old entries if memory is full
        all_entries = self.storage.search_entries()
        if len(all_entries) > self.max_memory_entries:
            # Sort by importance and age, remove least important old entries
            old_threshold = time.time() - (7 * 86400)  # 7 days old
            candidates = [
                entry for entry in all_entries 
                if entry.timestamp < old_threshold and entry.importance < 0.3
            ]
            
            candidates.sort(key=lambda x: (x.importance, x.last_accessed))
            
            for entry in candidates[:len(all_entries) - self.max_memory_entries]:
                if self.storage.delete_entry(entry.id):
                    results['low_importance_removed'] += 1
        
        self.logger.info(f"Memory cleanup completed: {results}")
        return results
    
    def _extract_task_tags(self, user_input: str) -> List[str]:
        """Extract relevant tags from user input."""
        tags = []
        input_lower = user_input.lower()
        
        # Common application keywords
        apps = ['chrome', 'firefox', 'notepad', 'excel', 'word', 'outlook', 'whatsapp', 'telegram']
        for app in apps:
            if app in input_lower:
                tags.append(app)
        
        # Common action keywords
        actions = ['open', 'close', 'search', 'type', 'click', 'scroll', 'copy', 'paste', 'download']
        for action in actions:
            if action in input_lower:
                tags.append(action)
        
        return tags
    
    def _calculate_task_importance(self, task_memory: TaskMemory) -> float:
        """Calculate importance score for task memory."""
        importance = 0.5  # Base importance
        
        # Successful tasks are more important
        if task_memory.is_successful:
            importance += 0.3
        
        # Tasks with user corrections are important
        if task_memory.corrections_applied:
            importance += 0.2
        
        # Longer tasks might be more complex/important
        if task_memory.duration > 60:  # More than 1 minute
            importance += 0.1
        
        # Tasks with user feedback are important
        if task_memory.user_feedback:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _identify_failure_patterns(self, task_memory: TaskMemory) -> List[str]:
        """Identify common failure patterns."""
        patterns = []
        
        if task_memory.status == TaskStatus.FAILED:
            for error in task_memory.error_messages:
                if 'timeout' in error.lower():
                    patterns.append('timeout_issues')
                elif 'element not found' in error.lower():
                    patterns.append('element_detection_issues')
                elif 'permission' in error.lower():
                    patterns.append('permission_issues')
        
        return patterns
    
    def _identify_success_factors(self, task_memory: TaskMemory) -> List[str]:
        """Identify factors that contributed to success."""
        factors = []
        
        if task_memory.is_successful:
            if task_memory.duration < 30:  # Quick execution
                factors.append('quick_execution')
            
            if not task_memory.error_messages:
                factors.append('error_free')
            
            if task_memory.success_rate > 0.9:
                factors.append('high_accuracy')
        
        return factors
    
    def _analyze_timing_patterns(self, task_memory: TaskMemory) -> Dict[str, Any]:
        """Analyze timing patterns in task execution."""
        return {
            'total_duration': task_memory.duration,
            'steps_count': len(task_memory.execution_steps),
            'avg_step_duration': task_memory.duration / max(len(task_memory.execution_steps), 1)
        }
    
    def _analyze_application_patterns(self, task_memory: TaskMemory) -> Dict[str, Any]:
        """Analyze application usage patterns."""
        apps_used = []
        
        for step in task_memory.execution_steps:
            if 'application' in step:
                apps_used.append(step['application'])
        
        return {
            'applications_used': list(set(apps_used)),
            'primary_application': max(set(apps_used), key=apps_used.count) if apps_used else None
        }
    
    def clear_memory(self) -> bool:
        """Clear all memory entries."""
        try:
            if isinstance(self.storage, JSONMemoryStorage):
                return self.storage._save_data({})
            elif isinstance(self.storage, SQLiteMemoryStorage):
                return self.storage._init_db()
            return False
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return False


# Factory function for easy initialization
def create_memory_service(config: Dict[str, Any]) -> MemoryService:
    """Create and initialize memory service."""
    return MemoryService(config)


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Example configuration
    config = {
        'storage_type': 'json',  # or 'sqlite'
        'memory_file': 'data/memory.json',
        'db_path': 'data/memory.db',
        'max_memory_entries': 1000,
        'cleanup_interval': 3600,
        'default_expiry_days': 30
    }
    
    # Create memory service
    memory_service = create_memory_service(config)
    
    # Example task memory
    task_memory = TaskMemory(
        task_id="test_task_001",
        user_input="Open Chrome and search for AI news",
        intent_analysis={"action": "search", "application": "chrome", "query": "AI news"},
        execution_steps=[
            {"action": "open_app", "target": "chrome", "timestamp": time.time()},
            {"action": "type", "text": "AI news", "timestamp": time.time() + 1}
        ],
        status=TaskStatus.COMPLETED,
        start_time=time.time(),
        end_time=time.time() + 30,
        success_rate=0.9
    )
    
    # Store task execution
    memory_service.store_task_execution(task_memory)
    
    # Store user preference
    memory_service.store_user_preference("default_browser", "chrome", "web_browsing")
    
    # Example user correction
    correction = UserCorrection(
        correction_id="correction_001",
        original_task_id="test_task_001",
        step_number=1,
        original_action={"action": "click", "coordinates": [100, 200]},
        corrected_action={"action": "click", "coordinates": [150, 220]},
        user_explanation="The button was slightly to the right",
        timestamp=time.time()
    )
    
    # Store user correction
    memory_service.store_user_correction(correction)
    
    # Get similar tasks
    similar_tasks = memory_service.get_similar_tasks("Open browser and search")
    print(f"Found {len(similar_tasks)} similar tasks")
    
    # Get memory statistics
    stats = memory_service.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    # Get user preferences
    preferences = memory_service.get_user_preferences("web_browsing")
    print(f"User preferences: {preferences}")
    
    print("Memory service example completed successfully!")