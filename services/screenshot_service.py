"""
Screenshot Service for AI Desktop Agent
Handles screen capture, storage, and management for task logging and ML training
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import threading
from dataclasses import dataclass, asdict
import json

# Third-party imports
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyautogui
import pygetwindow as gw

# Project imports
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ScreenshotMetadata:
    """Metadata for screenshot files"""
    filename: str
    timestamp: datetime
    task_id: Optional[str] = None
    step_number: Optional[int] = None
    window_title: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None
    file_size: Optional[int] = None
    annotations: Optional[List[Dict]] = None
    tags: Optional[List[str]] = None

class ScreenshotService:
    """
    Service for capturing, managing, and organizing screenshots
    Supports multiple formats, quality levels, and annotation features
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize screenshot service with configuration"""
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.base_dir = Path(self.config.get("screenshot_base_dir", "screenshots"))
        self.sessions_dir = self.base_dir / "sessions"
        self.archive_dir = self.base_dir / "archive"
        self.temp_dir = self.base_dir / "temp"
        
        self._create_directories()
        
        # Screenshot settings
        self.default_format = self.config.get("screenshot_format", "PNG")
        self.default_quality = self.config.get("screenshot_quality", 95)
        self.auto_cleanup_days = self.config.get("auto_cleanup_days", 7)
        self.max_screenshots_per_session = self.config.get("max_screenshots_per_session", 1000)
        
        # Current session
        self.current_session_id = self._generate_session_id()
        self.current_session_dir = self.sessions_dir / self.current_session_id
        self.current_session_dir.mkdir(exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = self.current_session_dir / "metadata.json"
        self.screenshots_metadata: List[ScreenshotMetadata] = []
        
        # Threading for async operations
        self.cleanup_thread = None
        self._setup_auto_cleanup()
        
        logger.info(f"Screenshot service initialized. Session: {self.current_session_id}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('screenshot_service', {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.base_dir, self.sessions_dir, self.archive_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def capture_full_screen(
        self,
        task_id: Optional[str] = None,
        step_number: Optional[int] = None,
        annotation: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Capture full screen screenshot
        
        Args:
            task_id: Associated task identifier
            step_number: Step number in task execution
            annotation: Text annotation for the screenshot
            tags: List of tags for categorization
        
        Returns:
            Path to saved screenshot file or None if failed
        """
        try:
            # Capture screenshot
            screenshot = pyautogui.screenshot()
            
            # Generate filename
            timestamp = datetime.now()
            filename = self._generate_filename("fullscreen", timestamp, task_id, step_number)
            filepath = self.current_session_dir / filename
            
            # Add annotation if provided
            if annotation:
                screenshot = self._add_text_annotation(screenshot, annotation)
            
            # Save screenshot
            screenshot.save(str(filepath), format=self.default_format, quality=self.default_quality)
            
            # Create metadata
            metadata = ScreenshotMetadata(
                filename=filename,
                timestamp=timestamp,
                task_id=task_id,
                step_number=step_number,
                resolution=(screenshot.width, screenshot.height),
                file_size=filepath.stat().st_size,
                tags=tags or []
            )
            
            self.screenshots_metadata.append(metadata)
            self._save_metadata()
            
            logger.info(f"Full screen captured: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to capture full screen: {e}")
            return None
    
    def capture_window(
        self,
        window_title: Optional[str] = None,
        task_id: Optional[str] = None,
        step_number: Optional[int] = None,
        annotation: Optional[str] = None
    ) -> Optional[str]:
        """
        Capture specific window screenshot
        
        Args:
            window_title: Title of window to capture (None for active window)
            task_id: Associated task identifier
            step_number: Step number in task execution
            annotation: Text annotation for the screenshot
        
        Returns:
            Path to saved screenshot file or None if failed
        """
        try:
            # Get target window
            if window_title:
                windows = gw.getWindowsWithTitle(window_title)
                if not windows:
                    logger.warning(f"Window '{window_title}' not found")
                    return None
                target_window = windows[0]
            else:
                target_window = gw.getActiveWindow()
                if not target_window:
                    logger.warning("No active window found")
                    return None
            
            # Get window coordinates
            left, top, width, height = target_window.left, target_window.top, target_window.width, target_window.height
            
            # Capture window area
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Generate filename
            timestamp = datetime.now()
            filename = self._generate_filename("window", timestamp, task_id, step_number)
            filepath = self.current_session_dir / filename
            
            # Add annotation if provided
            if annotation:
                screenshot = self._add_text_annotation(screenshot, annotation)
            
            # Save screenshot
            screenshot.save(str(filepath), format=self.default_format, quality=self.default_quality)
            
            # Create metadata
            metadata = ScreenshotMetadata(
                filename=filename,
                timestamp=timestamp,
                task_id=task_id,
                step_number=step_number,
                window_title=target_window.title,
                resolution=(screenshot.width, screenshot.height),
                file_size=filepath.stat().st_size
            )
            
            self.screenshots_metadata.append(metadata)
            self._save_metadata()
            
            logger.info(f"Window captured: {filename} ({target_window.title})")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to capture window: {e}")
            return None
    
    def capture_region(
        self,
        region: Tuple[int, int, int, int],
        task_id: Optional[str] = None,
        step_number: Optional[int] = None,
        annotation: Optional[str] = None
    ) -> Optional[str]:
        """
        Capture specific screen region
        
        Args:
            region: (left, top, width, height) coordinates
            task_id: Associated task identifier
            step_number: Step number in task execution
            annotation: Text annotation for the screenshot
        
        Returns:
            Path to saved screenshot file or None if failed
        """
        try:
            left, top, width, height = region
            
            # Capture region
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Generate filename
            timestamp = datetime.now()
            filename = self._generate_filename("region", timestamp, task_id, step_number)
            filepath = self.current_session_dir / filename
            
            # Add annotation if provided
            if annotation:
                screenshot = self._add_text_annotation(screenshot, annotation)
            
            # Save screenshot
            screenshot.save(str(filepath), format=self.default_format, quality=self.default_quality)
            
            # Create metadata
            metadata = ScreenshotMetadata(
                filename=filename,
                timestamp=timestamp,
                task_id=task_id,
                step_number=step_number,
                resolution=(screenshot.width, screenshot.height),
                file_size=filepath.stat().st_size
            )
            
            self.screenshots_metadata.append(metadata)
            self._save_metadata()
            
            logger.info(f"Region captured: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to capture region: {e}")
            return None
    
    def annotate_screenshot(
        self,
        screenshot_path: str,
        annotations: List[Dict],
        save_as_new: bool = True
    ) -> Optional[str]:
        """
        Add annotations to existing screenshot
        
        Args:
            screenshot_path: Path to existing screenshot
            annotations: List of annotation dictionaries with type, coordinates, text, etc.
            save_as_new: Whether to save as new file or overwrite
        
        Returns:
            Path to annotated screenshot
        """
        try:
            # Load image
            image = Image.open(screenshot_path)
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Apply annotations
            for annotation in annotations:
                ann_type = annotation.get('type', 'text')
                
                if ann_type == 'text':
                    x, y = annotation.get('position', (10, 10))
                    text = annotation.get('text', '')
                    color = annotation.get('color', 'red')
                    draw.text((x, y), text, fill=color, font=font)
                
                elif ann_type == 'rectangle':
                    coords = annotation.get('coordinates', (0, 0, 100, 100))
                    color = annotation.get('color', 'red')
                    width = annotation.get('width', 2)
                    for i in range(width):
                        draw.rectangle([coords[0]+i, coords[1]+i, coords[2]-i, coords[3]-i], outline=color)
                
                elif ann_type == 'circle':
                    center_x, center_y = annotation.get('center', (50, 50))
                    radius = annotation.get('radius', 20)
                    color = annotation.get('color', 'red')
                    width = annotation.get('width', 2)
                    bbox = [center_x-radius, center_y-radius, center_x+radius, center_y+radius]
                    for i in range(width):
                        draw.ellipse([bbox[0]+i, bbox[1]+i, bbox[2]-i, bbox[3]-i], outline=color)
                
                elif ann_type == 'arrow':
                    start = annotation.get('start', (0, 0))
                    end = annotation.get('end', (50, 50))
                    color = annotation.get('color', 'red')
                    width = annotation.get('width', 2)
                    draw.line([start, end], fill=color, width=width)
                    # Simple arrowhead
                    draw.polygon([end, (end[0]-10, end[1]-5), (end[0]-10, end[1]+5)], fill=color)
            
            # Save annotated image
            if save_as_new:
                base_name = Path(screenshot_path).stem
                new_name = f"{base_name}_annotated.{self.default_format.lower()}"
                new_path = Path(screenshot_path).parent / new_name
            else:
                new_path = screenshot_path
            
            image.save(str(new_path), format=self.default_format, quality=self.default_quality)
            
            logger.info(f"Screenshot annotated: {new_path}")
            return str(new_path)
            
        except Exception as e:
            logger.error(f"Failed to annotate screenshot: {e}")
            return None
    
    def get_session_screenshots(self, session_id: Optional[str] = None) -> List[ScreenshotMetadata]:
        """Get all screenshots from a session"""
        if session_id is None:
            return self.screenshots_metadata
        
        session_dir = self.sessions_dir / session_id
        metadata_file = session_dir / "metadata.json"
        
        if not metadata_file.exists():
            return []
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return [ScreenshotMetadata(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load session metadata: {e}")
            return []
    
    def delete_screenshot(self, filename: str) -> bool:
        """Delete a specific screenshot and its metadata"""
        try:
            filepath = self.current_session_dir / filename
            if filepath.exists():
                filepath.unlink()
                
                # Remove from metadata
                self.screenshots_metadata = [
                    meta for meta in self.screenshots_metadata 
                    if meta.filename != filename
                ]
                self._save_metadata()
                
                logger.info(f"Screenshot deleted: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete screenshot: {e}")
            return False
    
    def cleanup_old_screenshots(self, days_old: int = None) -> int:
        """Clean up screenshots older than specified days"""
        if days_old is None:
            days_old = self.auto_cleanup_days
        
        cleanup_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        try:
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                # Check session creation time
                try:
                    session_date = datetime.strptime(session_dir.name.split('_')[1] + '_' + session_dir.name.split('_')[2], '%Y%m%d_%H%M%S')
                    if session_date < cleanup_date:
                        # Move to archive or delete
                        archive_path = self.archive_dir / session_dir.name
                        session_dir.rename(archive_path)
                        cleaned_count += 1
                        logger.info(f"Archived old session: {session_dir.name}")
                except (IndexError, ValueError):
                    # Skip directories with invalid names
                    continue
            
            logger.info(f"Cleanup completed. Archived {cleaned_count} sessions.")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup screenshots: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        try:
            total_size = 0
            total_files = 0
            
            for session_dir in self.sessions_dir.iterdir():
                if session_dir.is_dir():
                    for file in session_dir.rglob("*.png"):
                        total_size += file.stat().st_size
                        total_files += 1
                    for file in session_dir.rglob("*.jpg"):
                        total_size += file.stat().st_size
                        total_files += 1
            
            return {
                'total_screenshots': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'active_sessions': len(list(self.sessions_dir.iterdir())),
                'current_session_screenshots': len(self.screenshots_metadata)
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def _generate_filename(
        self,
        capture_type: str,
        timestamp: datetime,
        task_id: Optional[str] = None,
        step_number: Optional[int] = None
    ) -> str:
        """Generate standardized filename"""
        base = f"{capture_type}_{timestamp.strftime('%H%M%S_%f')[:-3]}"
        
        if task_id:
            base += f"_task{task_id}"
        
        if step_number is not None:
            base += f"_step{step_number:03d}"
        
        return f"{base}.{self.default_format.lower()}"
    
    def _add_text_annotation(self, image: Image.Image, text: str) -> Image.Image:
        """Add text annotation to image"""
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add background rectangle for text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = 10, 10
        draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill='black', outline='white')
        draw.text((x, y), text, fill='white', font=font)
        
        return image
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            metadata_dict = [asdict(meta) for meta in self.screenshots_metadata]
            # Convert datetime objects to strings
            for meta in metadata_dict:
                if isinstance(meta['timestamp'], datetime):
                    meta['timestamp'] = meta['timestamp'].isoformat()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _setup_auto_cleanup(self):
        """Setup automatic cleanup thread"""
        if self.auto_cleanup_days > 0:
            def cleanup_worker():
                while True:
                    time.sleep(24 * 60 * 60)  # Run daily
                    self.cleanup_old_screenshots()
            
            self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            self.cleanup_thread.start()
    
    def __del__(self):
        """Cleanup on service destruction"""
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread:
            self.cleanup_thread = None
    
    def capture_screenshot(
        self,
        task_id: Optional[str] = None,
        step_number: Optional[int] = None,
        annotation: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Capture screenshot (alias for capture_full_screen)
        
        Args:
            task_id: Associated task identifier
            step_number: Step number in task execution
            annotation: Text annotation for the screenshot
            tags: List of tags for categorization
        
        Returns:
            Path to saved screenshot file or None if failed
        """
        return self.capture_full_screen(
            task_id=task_id,
            step_number=step_number,
            annotation=annotation,
            tags=tags
        )