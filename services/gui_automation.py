"""
GUI Automation Service
Handles mouse, keyboard, window focus operations using PyAutoGUI and pynput
"""

import time
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    import pyautogui
    import pynput
    from pynput import mouse, keyboard
    import psutil
    import win32gui
    import win32con
    import win32process
except ImportError as e:
    logging.warning(f"GUI automation dependencies not available: {e}")
    pyautogui = None
    pynput = None


class ClickType(Enum):
    LEFT = "left"
    RIGHT = "right" 
    DOUBLE = "double"
    MIDDLE = "middle"


class KeyAction(Enum):
    PRESS = "press"
    HOLD = "hold"
    RELEASE = "release"
    TYPE = "type"


@dataclass
class WindowInfo:
    """Information about a window"""
    handle: int
    title: str
    class_name: str
    pid: int
    position: Tuple[int, int]
    size: Tuple[int, int]
    is_visible: bool
    is_active: bool


@dataclass
class MouseAction:
    """Mouse action configuration"""
    x: int
    y: int
    click_type: ClickType = ClickType.LEFT
    duration: float = 0.1
    clicks: int = 1


@dataclass
class KeyboardAction:
    """Keyboard action configuration"""
    action: KeyAction
    key_or_text: str
    duration: float = 0.1
    modifiers: List[str] = None


class GUIAutomationService:
    """
    Service for GUI automation including mouse, keyboard, and window management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Safety settings
        self.fail_safe = self.config.get('fail_safe', True)
        self.pause_duration = self.config.get('pause_duration', 0.1)
        self.screenshot_on_action = self.config.get('screenshot_on_action', False)
        
        # Initialize PyAutoGUI settings
        if pyautogui:
            pyautogui.FAILSAFE = self.fail_safe
            pyautogui.PAUSE = self.pause_duration
            
        self._setup_hotkeys()
        
    def _setup_hotkeys(self):
        """Setup emergency hotkeys for safety"""
        if not pynput:
            return
            
        def on_emergency_stop():
            self.logger.warning("Emergency stop activated!")
            # Could implement emergency stop logic here
            
        # Emergency stop with Ctrl+Alt+Q
        self.hotkey_listener = keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+q': on_emergency_stop
        })
        
    # =================== MOUSE OPERATIONS ===================
    
    def click(self, x: int, y: int, click_type: ClickType = ClickType.LEFT, 
              duration: float = 0.1, clicks: int = 1) -> bool:
        """
        Perform mouse click at specified coordinates
        
        Args:
            x, y: Screen coordinates
            click_type: Type of click to perform
            duration: Duration to hold click
            clicks: Number of clicks
            
        Returns:
            bool: Success status
        """
        if not pyautogui:
            self.logger.error("PyAutoGUI not available")
            return False
            
        try:
            self.logger.info(f"Clicking at ({x}, {y}) with {click_type.value}")
            
            if self.screenshot_on_action:
                self._take_screenshot_before_action("click", x, y)
                
            if click_type == ClickType.LEFT:
                pyautogui.click(x, y, clicks=clicks, duration=duration, button='left')
            elif click_type == ClickType.RIGHT:
                pyautogui.click(x, y, clicks=clicks, duration=duration, button='right')
            elif click_type == ClickType.DOUBLE:
                pyautogui.doubleClick(x, y, duration=duration)
            elif click_type == ClickType.MIDDLE:
                pyautogui.click(x, y, clicks=clicks, duration=duration, button='middle')
                
            return True
            
        except Exception as e:
            self.logger.error(f"Click failed: {e}")
            return False
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
             duration: float = 1.0, button: str = 'left') -> bool:
        """
        Drag from start to end coordinates
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            duration: Duration of drag operation
            button: Mouse button to use
            
        Returns:
            bool: Success status
        """
        if not pyautogui:
            return False
            
        try:
            self.logger.info(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration, button=button)
            return True
        except Exception as e:
            self.logger.error(f"Drag failed: {e}")
            return False
    
    def scroll(self, x: int, y: int, direction: int, clicks: int = 3) -> bool:
        """
        Scroll at specified coordinates
        
        Args:
            x, y: Coordinates to scroll at
            direction: Positive for up, negative for down
            clicks: Number of scroll clicks
            
        Returns:
            bool: Success status
        """
        if not pyautogui:
            return False
            
        try:
            self.logger.info(f"Scrolling at ({x}, {y}), direction: {direction}")
            pyautogui.scroll(direction * clicks, x=x, y=y)
            return True
        except Exception as e:
            self.logger.error(f"Scroll failed: {e}")
            return False
    
    # =================== KEYBOARD OPERATIONS ===================
    
    def type_text(self, text: str, interval: float = 0.01) -> bool:
        """
        Type text with specified interval between characters
        
        Args:
            text: Text to type
            interval: Interval between keystrokes
            
        Returns:
            bool: Success status
        """
        if not pyautogui:
            return False
            
        try:
            self.logger.info(f"Typing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            pyautogui.typewrite(text, interval=interval)
            return True
        except Exception as e:
            self.logger.error(f"Type text failed: {e}")
            return False
    
    def press_key(self, key: str, presses: int = 1, interval: float = 0.0) -> bool:
        """
        Press a key or key combination
        
        Args:
            key: Key to press (e.g., 'enter', 'ctrl+c', 'alt+tab')
            presses: Number of times to press
            interval: Interval between presses
            
        Returns:
            bool: Success status
        """
        if not pyautogui:
            return False
            
        try:
            self.logger.info(f"Pressing key: {key} ({presses} times)")
            
            if '+' in key:
                # Handle key combinations
                keys = key.split('+')
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(key, presses=presses, interval=interval)
                
            return True
        except Exception as e:
            self.logger.error(f"Key press failed: {e}")
            return False
    
    def hold_key(self, key: str, duration: float = 1.0) -> bool:
        """
        Hold a key for specified duration
        
        Args:
            key: Key to hold
            duration: Duration to hold key
            
        Returns:
            bool: Success status
        """
        if not pyautogui:
            return False
            
        try:
            self.logger.info(f"Holding key: {key} for {duration}s")
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            return True
        except Exception as e:
            self.logger.error(f"Hold key failed: {e}")
            return False
    
    # =================== WINDOW OPERATIONS ===================
    
    def get_active_window(self) -> Optional[WindowInfo]:
        """
        Get information about the currently active window
        
        Returns:
            WindowInfo: Information about active window, None if failed
        """
        try:
            if win32gui:
                hwnd = win32gui.GetForegroundWindow()
                return self._get_window_info(hwnd)
        except Exception as e:
            self.logger.error(f"Get active window failed: {e}")
        return None
    
    def get_windows_by_title(self, title_pattern: str) -> List[WindowInfo]:
        """
        Find windows by title pattern
        
        Args:
            title_pattern: Pattern to match in window title
            
        Returns:
            List[WindowInfo]: List of matching windows
        """
        windows = []
        if not win32gui:
            return windows
            
        def enum_handler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if title_pattern.lower() in window_title.lower():
                    window_info = self._get_window_info(hwnd)
                    if window_info:
                        windows.append(window_info)
        
        try:
            win32gui.EnumWindows(enum_handler, None)
        except Exception as e:
            self.logger.error(f"Enumerate windows failed: {e}")
            
        return windows
    
    def activate_window(self, window_handle: int) -> bool:
        """
        Activate (bring to front) a window by handle
        
        Args:
            window_handle: Window handle to activate
            
        Returns:
            bool: Success status
        """
        if not win32gui:
            return False
            
        try:
            self.logger.info(f"Activating window handle: {window_handle}")
            win32gui.SetForegroundWindow(window_handle)
            return True
        except Exception as e:
            self.logger.error(f"Activate window failed: {e}")
            return False
    
    def minimize_window(self, window_handle: int) -> bool:
        """
        Minimize a window
        
        Args:
            window_handle: Window handle to minimize
            
        Returns:
            bool: Success status
        """
        if not win32gui:
            return False
            
        try:
            win32gui.ShowWindow(window_handle, win32con.SW_MINIMIZE)
            return True
        except Exception as e:
            self.logger.error(f"Minimize window failed: {e}")
            return False
    
    def maximize_window(self, window_handle: int) -> bool:
        """
        Maximize a window
        
        Args:
            window_handle: Window handle to maximize
            
        Returns:
            bool: Success status
        """
        if not win32gui:
            return False
            
        try:
            win32gui.ShowWindow(window_handle, win32con.SW_MAXIMIZE)
            return True
        except Exception as e:
            self.logger.error(f"Maximize window failed: {e}")
            return False
    
    def _get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """
        Get detailed information about a window
        
        Args:
            hwnd: Window handle
            
        Returns:
            WindowInfo: Window information, None if failed
        """
        if not win32gui:
            return None
            
        try:
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            
            # Get process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            return WindowInfo(
                handle=hwnd,
                title=title,
                class_name=class_name,
                pid=pid,
                position=(rect[0], rect[1]),
                size=(rect[2] - rect[0], rect[3] - rect[1]),
                is_visible=win32gui.IsWindowVisible(hwnd),
                is_active=hwnd == win32gui.GetForegroundWindow()
            )
        except Exception as e:
            self.logger.error(f"Get window info failed: {e}")
            return None
    
    # =================== UTILITY OPERATIONS ===================
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get screen dimensions
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        if pyautogui:
            return pyautogui.size()
        return (1920, 1080)  # Default fallback
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """
        Get current mouse position
        
        Returns:
            Tuple[int, int]: (x, y) coordinates
        """
        if pyautogui:
            return pyautogui.position()
        return (0, 0)
    
    def wait(self, seconds: float) -> None:
        """
        Wait for specified duration
        
        Args:
            seconds: Duration to wait
        """
        self.logger.info(f"Waiting for {seconds} seconds")
        time.sleep(seconds)
    
    def _take_screenshot_before_action(self, action: str, x: int = None, y: int = None) -> None:
        """
        Take screenshot before performing action (for debugging)
        
        Args:
            action: Action being performed
            x, y: Coordinates (if applicable)
        """
        try:
            if pyautogui:
                timestamp = int(time.time())
                filename = f"action_{action}_{timestamp}.png"
                screenshot = pyautogui.screenshot()
                screenshot.save(filename)
                self.logger.debug(f"Screenshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
    
    # =================== HIGH-LEVEL OPERATIONS ===================
    
    def open_application(self, app_name: str) -> bool:
        """
        Open an application by name
        
        Args:
            app_name: Name of application to open
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Opening application: {app_name}")
            
            # Press Windows key and type app name
            self.press_key('win')
            self.wait(0.5)
            self.type_text(app_name)
            self.wait(1.0)
            self.press_key('enter')
            
            return True
        except Exception as e:
            self.logger.error(f"Open application failed: {e}")
            return False
    
    def close_active_window(self) -> bool:
        """
        Close the currently active window
        
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("Closing active window")
            return self.press_key('alt+f4')
        except Exception as e:
            self.logger.error(f"Close window failed: {e}")
            return False
    
    def copy_to_clipboard(self) -> bool:
        """
        Copy selected content to clipboard
        
        Returns:
            bool: Success status
        """
        return self.press_key('ctrl+c')
    
    def paste_from_clipboard(self) -> bool:
        """
        Paste content from clipboard
        
        Returns:
            bool: Success status
        """
        return self.press_key('ctrl+v')
    
    def select_all(self) -> bool:
        """
        Select all content
        
        Returns:
            bool: Success status
        """
        return self.press_key('ctrl+a')
    
    def undo(self) -> bool:
        """
        Perform undo operation
        
        Returns:
            bool: Success status
        """
        return self.press_key('ctrl+z')
    
    def redo(self) -> bool:
        """
        Perform redo operation
        
        Returns:
            bool: Success status
        """
        return self.press_key('ctrl+y')


# =================== EXAMPLE USAGE ===================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    gui = GUIAutomationService({
        'fail_safe': True,
        'pause_duration': 0.1,
        'screenshot_on_action': True
    })
    
    # Example operations
    try:
        # Get screen info
        screen_size = gui.get_screen_size()
        print(f"Screen size: {screen_size}")
        
        # Get current mouse position
        mouse_pos = gui.get_mouse_position()
        print(f"Mouse position: {mouse_pos}")
        
        # Get active window
        active_window = gui.get_active_window()
        if active_window:
            print(f"Active window: {active_window.title}")
        
        # Example: Open Notepad and type text
        # gui.open_application("notepad")
        # gui.wait(2)
        # gui.type_text("Hello from GUI Automation!")
        
    except Exception as e:
        print(f"Example failed: {e}")