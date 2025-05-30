import logging
import traceback
import sys
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import pyautogui


class ErrorSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorType(Enum):
    AUTOMATION_ERROR = "AUTOMATION_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    FILE_ERROR = "FILE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorHandler:
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the ErrorHandler with logging configuration.
        
        Args:
            log_file: Optional path to log file. If None, uses default error.log
        """
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file or "logs/error.log"
        self.error_count = 0
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration for error handling."""
        try:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Create file handler if log file is specified
            if self.log_file:
                import os
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(logging.ERROR)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    def handle_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle and log exceptions with detailed information.
        
        Args:
            exception: The exception that occurred
            context: Optional context information about where the error occurred
            
        Returns:
            Dict containing error details
        """
        self.error_count += 1
        
        # Determine error type and severity
        error_type = self._classify_error(exception)
        severity = self._determine_severity(exception, error_type)
        
        # Create error details
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'error_id': f"ERR_{self.error_count:04d}",
            'type': error_type.value,
            'severity': severity.value,
            'message': str(exception),
            'exception_type': type(exception).__name__,
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        # Log the error
        self._log_error(error_details)
        
        # Take recovery actions if possible
        self._attempt_recovery(exception, error_type, severity)
        
        return error_details
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Classify the error type based on the exception."""
        exception_type = type(exception).__name__
        
        # PyAutoGUI specific errors
        if 'pyautogui' in exception_type.lower() or isinstance(exception, pyautogui.FailSafeException):
            return ErrorType.AUTOMATION_ERROR
            
        # System and OS errors
        if exception_type in ['OSError', 'SystemError', 'PermissionError']:
            return ErrorType.SYSTEM_ERROR
            
        # Network related errors
        if exception_type in ['ConnectionError', 'TimeoutError', 'URLError', 'HTTPError']:
            return ErrorType.NETWORK_ERROR
            
        # File operations
        if exception_type in ['FileNotFoundError', 'IOError', 'IsADirectoryError']:
            return ErrorType.FILE_ERROR
            
        # Validation errors
        if exception_type in ['ValueError', 'TypeError', 'KeyError', 'IndexError']:
            return ErrorType.VALIDATION_ERROR
            
        # Timeout errors
        if 'timeout' in exception_type.lower():
            return ErrorType.TIMEOUT_ERROR
            
        return ErrorType.UNKNOWN_ERROR
    
    def _determine_severity(self, exception: Exception, error_type: ErrorType) -> ErrorSeverity:
        """Determine the severity of the error."""
        exception_type = type(exception).__name__
        
        # Critical errors that should stop execution
        if exception_type in ['SystemExit', 'KeyboardInterrupt', 'MemoryError']:
            return ErrorSeverity.CRITICAL
            
        # High severity errors
        if error_type in [ErrorType.SYSTEM_ERROR, ErrorType.FILE_ERROR]:
            return ErrorSeverity.HIGH
            
        # Medium severity errors
        if error_type in [ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
            
        # Low severity errors (recoverable)
        if error_type in [ErrorType.AUTOMATION_ERROR, ErrorType.VALIDATION_ERROR]:
            return ErrorSeverity.LOW
            
        return ErrorSeverity.MEDIUM
    
    def _log_error(self, error_details: Dict[str, Any]):
        """Log error details with appropriate level."""
        severity = error_details['severity']
        message = f"[{error_details['error_id']}] {error_details['type']}: {error_details['message']}"
        
        if severity == ErrorSeverity.CRITICAL.value:
            self.logger.critical(message)
            self.logger.critical(f"Traceback: {error_details['traceback']}")
        elif severity == ErrorSeverity.HIGH.value:
            self.logger.error(message)
            self.logger.error(f"Context: {error_details['context']}")
        elif severity == ErrorSeverity.MEDIUM.value:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _attempt_recovery(self, exception: Exception, error_type: ErrorType, severity: ErrorSeverity):
        """Attempt to recover from certain types of errors."""
        try:
            if error_type == ErrorType.AUTOMATION_ERROR:
                # Reset mouse position to safe area
                try:
                    pyautogui.moveTo(100, 100)
                    self.logger.info("Reset mouse position for recovery")
                except:
                    pass
                    
            elif error_type == ErrorType.SYSTEM_ERROR and severity != ErrorSeverity.CRITICAL:
                # Wait a moment for system to recover
                import time
                time.sleep(1)
                self.logger.info("Waited for system recovery")
                
        except Exception as recovery_error:
            self.logger.warning(f"Recovery attempt failed: {recovery_error}")
    
    def handle_validation_error(self, field_name: str, value: Any, expected_type: type = None) -> Dict[str, Any]:
        """
        Handle validation errors with specific context.
        
        Args:
            field_name: Name of the field that failed validation
            value: The invalid value
            expected_type: Expected type for the value
            
        Returns:
            Dict containing validation error details
        """
        context = {
            'field_name': field_name,
            'invalid_value': str(value),
            'value_type': type(value).__name__
        }
        
        if expected_type:
            context['expected_type'] = expected_type.__name__
            
        error_msg = f"Validation failed for field '{field_name}': expected {expected_type.__name__ if expected_type else 'valid value'}, got {type(value).__name__}"
        validation_error = ValueError(error_msg)
        
        return self.handle_exception(validation_error, context)
    
    def handle_automation_error(self, action: str, params: Dict[str, Any], original_exception: Exception) -> Dict[str, Any]:
        """
        Handle automation-specific errors with action context.
        
        Args:
            action: The automation action that failed
            params: Parameters used for the action
            original_exception: The original exception that occurred
            
        Returns:
            Dict containing automation error details
        """
        context = {
            'failed_action': action,
            'action_params': params,
            'automation_tool': 'pyautogui'
        }
        
        return self.handle_exception(original_exception, context)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        return {
            'total_errors': self.error_count,
            'log_file': self.log_file,
            'handler_initialized': datetime.now().isoformat()
        }
    
    def reset_error_count(self):
        """Reset the error counter."""
        self.error_count = 0
        self.logger.info("Error count reset")
    
    def is_critical_error(self, exception: Exception) -> bool:
        """Check if an error is critical and should stop execution."""
        error_type = self._classify_error(exception)
        severity = self._determine_severity(exception, error_type)
        return severity == ErrorSeverity.CRITICAL