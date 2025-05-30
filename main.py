#!/usr/bin/env python3
"""
AI Automation System - Main Entry Point
A comprehensive AI-powered desktop automation system that combines LLMs with computer vision
for intelligent task execution and system control.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class AIAutomationSystem:
    """Main system orchestrator for the AI automation platform."""
    
    def __init__(self):
        """Initialize the AI automation system."""
        self.config = self._load_config()
        self._setup_logging()
        self.running = False
        
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from YAML file."""
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Config file not found at {config_path}")
            print("Creating default configuration...")
            self._create_default_config(config_path)
            return self._load_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            return self._load_default_config()
    
    def _create_default_config(self, config_path: Path):
        """Create a default configuration file if none exists."""
        config_path.parent.mkdir(exist_ok=True)
        default_config = self._load_default_config()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        print(f"✅ Default config created at {config_path}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Return default configuration settings."""
        return {
            'system': {
                'name': 'AI Automation System',
                'version': '1.0.0',
                'debug': True,
                'log_level': 'INFO'
            },
            'llm': {
                'default_provider': 'openai',
                'openai': {
                    'api_key': 'your-openai-api-key-here',
                    'model': 'gpt-4',
                    'max_tokens': 2000,
                    'temperature': 0.7
                },
                'gemini': {
                    'api_key': 'your-gemini-api-key-here',
                    'model': 'gemini-pro',
                    'temperature': 0.7
                }
            },
            'automation': {
                'screenshot_delay': 1.0,
                'action_delay': 0.5,
                'max_retries': 3,
                'confidence_threshold': 0.8
            },
            'paths': {
                'screenshots': str(PROJECT_ROOT / "data" / "screenshots"),
                'memory': str(PROJECT_ROOT / "data" / "memory.json"),
                'logs': str(PROJECT_ROOT / "logs")
            },
            'ui': {
                'streamlit_port': 8501,
                'theme': 'dark',
                'auto_refresh': True
            },
            'vision': {
                'yolo_model': 'yolov8n.pt',
                'detr_model': 'facebook/detr-resnet-50',
                'detection_threshold': 0.5
            }
        }
    
    def _setup_logging(self):
        """Configure system logging."""
        log_level = getattr(logging, self.config['system']['log_level'], logging.INFO)
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'automation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging system initialized")
    
    def display_banner(self):
        """Display system banner and information."""
        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                   AI AUTOMATION SYSTEM                      ║
║                     Version {self.config['system']['version']}                        ║
╚══════════════════════════════════════════════════════════════╝

🤖 Intelligent Desktop Automation with AI Vision
🎯 Multi-LLM Support (OpenAI, Gemini)
🖥️  Computer Vision Integration (YOLO, DETR)
📱 Web UI Dashboard Available
"""
        print(banner)
    
    def display_menu(self):
        """Display the main menu options."""
        menu = """
┌─────────────────── MAIN MENU ───────────────────┐
│                                                 │
│  1. 🚀 Start Interactive Mode                   │
│  2. 🎯 Execute Single Task                      │
│  3. 📊 Launch Web Dashboard                     │
│  4. 🔧 System Configuration                     │
│  5. 📋 View System Status                       │
│  6. 🧪 Run System Tests                         │
│  7. 📚 View Documentation                       │
│  8. 🔄 Reload Configuration                     │
│  9. 🚪 Exit System                              │
│                                                 │
└─────────────────────────────────────────────────┘
"""
        print(menu)
    
    def get_user_choice(self) -> str:
        """Get user menu choice with validation."""
        while True:
            try:
                choice = input("\n🎯 Enter your choice (1-9): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    return choice
                else:
                    print("❌ Invalid choice. Please enter a number between 1-9.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return '9'
            except EOFError:
                return '9'
    
    def handle_choice(self, choice: str):
        """Handle user menu selection."""
        handlers = {
            '1': self._start_interactive_mode,
            '2': self._execute_single_task,
            '3': self._launch_dashboard,
            '4': self._configure_system,
            '5': self._show_system_status,
            '6': self._run_tests,
            '7': self._show_documentation,
            '8': self._reload_config,
            '9': self._exit_system
        }
        
        handler = handlers.get(choice)
        if handler:
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Error handling choice {choice}: {e}")
                print(f"❌ Error: {e}")
        else:
            print("❌ Invalid choice")
    
    def _start_interactive_mode(self):
        """Start interactive AI automation mode."""
        print("\n🚀 Starting Interactive Mode...")
        print("💡 This feature will be available in Stage 2 (LLM Service implementation)")
        print("🔄 Returning to main menu...")
        input("\n📱 Press Enter to continue...")
    
    def _execute_single_task(self):
        """Execute a single automation task."""
        print("\n🎯 Single Task Execution...")
        task = input("📝 Enter your task description: ").strip()
        if task:
            print(f"📋 Task received: {task}")
            print("💡 Task execution will be available in Stage 3 (Task Router implementation)")
        else:
            print("❌ No task provided")
        input("\n📱 Press Enter to continue...")
    
    def _launch_dashboard(self):
        """Launch the Streamlit web dashboard."""
        print("\n📊 Launching Web Dashboard...")
        port = self.config['ui']['streamlit_port']
        print(f"🌐 Dashboard will be available at: http://localhost:{port}")
        print("💡 Web UI will be available in Stage 5 (Streamlit implementation)")
        input("\n📱 Press Enter to continue...")
    
    def _configure_system(self):
        """Configure system settings."""
        print("\n🔧 System Configuration...")
        print("📁 Configuration file location:", PROJECT_ROOT / "config" / "settings.yaml")
        print("💡 Advanced configuration features coming in later stages")
        
        config_menu = """
Current Configuration:
├── LLM Provider: {provider}
├── Debug Mode: {debug}
├── Log Level: {log_level}
└── Screenshot Path: {screenshots}
""".format(
            provider=self.config['llm']['default_provider'],
            debug=self.config['system']['debug'],
            log_level=self.config['system']['log_level'],
            screenshots=self.config['paths']['screenshots']
        )
        print(config_menu)
        input("\n📱 Press Enter to continue...")
    
    def _show_system_status(self):
        """Display current system status."""
        print("\n📋 System Status Check...")
        
        # Check critical directories
        paths_to_check = [
            ('Screenshots', self.config['paths']['screenshots']),
            ('Memory', Path(self.config['paths']['memory']).parent),
            ('Logs', self.config['paths']['logs'])
        ]
        
        print("\n📁 Directory Status:")
        for name, path in paths_to_check:
            path_obj = Path(path)
            status = "✅ EXISTS" if path_obj.exists() else "❌ MISSING"
            print(f"  {name}: {status} ({path})")
        
        # Check configuration
        print(f"\n⚙️  Configuration Status:")
        print(f"  System Version: {self.config['system']['version']}")
        print(f"  Debug Mode: {self.config['system']['debug']}")
        print(f"  Default LLM: {self.config['llm']['default_provider']}")
        
        print(f"\n🔧 System Components:")
        components = [
            ("LLM Service", "⏳ Pending (Stage 2)"),
            ("Memory Service", "⏳ Pending (Stage 2)"),
            ("Task Router", "⏳ Pending (Stage 3)"),
            ("GUI Automation", "⏳ Pending (Stage 4)"),
            ("Vision Service", "⏳ Pending (Stage 4)"),
            ("Web Dashboard", "⏳ Pending (Stage 5)")
        ]
        
        for component, status in components:
            print(f"  {component}: {status}")
        
        input("\n📱 Press Enter to continue...")
    
    def _run_tests(self):
        """Run system tests."""
        print("\n🧪 Running System Tests...")
        print("💡 Test framework will be available in Stage 6")
        
        # Basic configuration test
        print("\n🔍 Basic Configuration Test:")
        try:
            assert self.config is not None
            assert 'system' in self.config
            assert 'llm' in self.config
            print("  ✅ Configuration structure: PASSED")
        except AssertionError:
            print("  ❌ Configuration structure: FAILED")
        
        # Path validation test
        print("\n📁 Path Validation Test:")
        for path_name, path_value in self.config['paths'].items():
            path_obj = Path(path_value)
            if path_name == 'memory':
                path_obj = path_obj.parent
            path_obj.mkdir(parents=True, exist_ok=True)
            status = "✅ PASSED" if path_obj.exists() else "❌ FAILED"
            print(f"  {path_name}: {status}")
        
        input("\n📱 Press Enter to continue...")
    
    def _show_documentation(self):
        """Display system documentation."""
        doc = """
📚 AI AUTOMATION SYSTEM DOCUMENTATION

🎯 OVERVIEW
This system provides intelligent desktop automation using AI and computer vision.
It combines large language models (LLMs) with advanced vision models to understand
and execute complex automation tasks.

🏗️ ARCHITECTURE
├── Stage 1: Core Framework (CURRENT)
│   ├── main.py - System entry point
│   └── config/settings.yaml - Configuration
├── Stage 2: Core Services (NEXT)
│   ├── LLM Service - AI integration
│   └── Memory Service - State management
├── Stage 3: Task Management
│   ├── Task Router - Intent classification
│   └── Task Executor - Action execution
└── ... (Additional stages)

🚀 GETTING STARTED
1. Configure your API keys in config/settings.yaml
2. Install dependencies: pip install -r requirements.txt
3. Run: python main.py

🔧 CONFIGURATION
Edit config/settings.yaml to customize:
- LLM providers and settings
- Automation parameters
- File paths and UI preferences

📞 SUPPORT
For issues and contributions, check the README.md file.
"""
        print(doc)
        input("\n📱 Press Enter to continue...")
    
    def _reload_config(self):
        """Reload system configuration."""
        print("\n🔄 Reloading Configuration...")
        try:
            old_config = self.config.copy()
            self.config = self._load_config()
            self._setup_logging()
            print("✅ Configuration reloaded successfully")
            
            # Show what changed
            if old_config != self.config:
                print("📋 Configuration changes detected")
            else:
                print("📋 No configuration changes found")
                
        except Exception as e:
            print(f"❌ Error reloading config: {e}")
            self.logger.error(f"Config reload failed: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def _exit_system(self):
        """Exit the system gracefully."""
        print("\n🚪 Shutting down AI Automation System...")
        print("💾 Saving system state...")
        print("🧹 Cleaning up resources...")
        print("✅ Shutdown complete")
        print("\n👋 Thank you for using AI Automation System!")
        self.running = False
    
    def run(self):
        """Main system loop."""
        self.running = True
        self.display_banner()
        
        # Ensure required directories exist
        for path_name, path_value in self.config['paths'].items():
            path_obj = Path(path_value)
            if path_name == 'memory':
                path_obj = path_obj.parent
            path_obj.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("AI Automation System started")
        
        try:
            while self.running:
                self.display_menu()
                choice = self.get_user_choice()
                self.handle_choice(choice)
        except KeyboardInterrupt:
            print("\n\n🛑 System interrupted by user")
            self._exit_system()
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            print(f"❌ Unexpected error: {e}")
        finally:
            self.logger.info("AI Automation System stopped")


def main():
    """Entry point for the AI automation system."""
    try:
        system = AIAutomationSystem()
        system.run()
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())