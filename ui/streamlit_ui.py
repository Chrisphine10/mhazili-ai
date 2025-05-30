"""
Main Streamlit Dashboard for AI Agent
Provides interactive interface for task management, monitoring, and control
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
from pathlib import Path
import sys
import os
import logging

import yaml

from utils.logger import get_logger

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ui.components import (
    render_sidebar,
    render_task_input,
    render_execution_status,
    render_memory_panel,
    render_logs_panel,
    render_settings_modal,
    render_metrics_dashboard
)

# Import your services (these will be available after previous stages)
try:
    from services.llm_service import LLMService
    from services.memory_service import MemoryService
    from controllers.task_router import TaskRouter
    from controllers.task_executor import TaskExecutor
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Initialize logger
logger = get_logger(
    name=__name__,
    level=logging.INFO,
    log_file="logs/streamlit_ui.log",
    format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Page configuration
st.set_page_config(
    page_title="AI Desktop Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .status-container {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-idle {
        background-color: #f0f2f6;
        border-left: 4px solid #007bff;
    }
    
    .status-running {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    .status-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = 'idle'
    
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
    
    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = []
    
    if 'memory_data' not in st.session_state:
        st.session_state.memory_data = []
    
    if 'settings_open' not in st.session_state:
        st.session_state.settings_open = False
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'services_initialized' not in st.session_state:
        st.session_state.services_initialized = False

def initialize_services():
    """Initialize AI services"""
    if not st.session_state.services_initialized:
        try:
            with open("config/settings.yaml", "r") as f:
                config = yaml.safe_load(f)
                
            with st.spinner("Initializing AI services..."):
                st.session_state.llm_service = LLMService()
                st.session_state.memory_service = MemoryService(config.get('memory', {}))
                st.session_state.task_router = TaskRouter()
                st.session_state.task_executor = TaskExecutor()
                st.session_state.services_initialized = True
                logger.info("Services initialized successfully")
                return True
        except Exception as e:
            st.error(f"Failed to initialize services: {str(e)}")
            logger.error(f"Service initialization failed: {e}")
            return False
    return True

async def execute_task(task_description: str):
    """Execute a task asynchronously"""
    try:
        logger.info(f"Starting task execution: {task_description}")
        st.session_state.agent_status = 'running'
        st.session_state.current_task = task_description
        
        # Add to execution logs
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'info',
            'message': f"Starting task: {task_description}"
        }
        st.session_state.execution_logs.append(log_entry)
                
        # Route task
        logger.info("Routing task...")
        task_data = await st.session_state.task_router.route_task(task_description)
        logger.info(f"Task routed with data: {task_data}")
        
        if not task_data:
            error_msg = "Task routing failed - no task data returned"
            logger.error(error_msg)
            st.session_state.agent_status = 'error'
            st.session_state.execution_logs.append({
                'timestamp': datetime.now(),
                'type': 'error',
                'message': error_msg
            })
            return {'success': False, 'message': error_msg}
        
        # Execute task
        logger.info("Executing task...")
        try:            
            result = await st.session_state.task_executor.execute_task(task_data)
            logger.info(f"Task execution result: {result}")
        except Exception as exec_error:
            error_msg = f"Task execution failed: {str(exec_error)}"
            logger.error(error_msg, exc_info=True)
            st.session_state.agent_status = 'error'
            st.session_state.execution_logs.append({
                'timestamp': datetime.now(),
                'type': 'error',
                'message': error_msg
            })
            return {'success': False, 'message': error_msg}
        
        # Update memory
        try:
            logger.info("Storing execution in memory...")
            await st.session_state.memory_service.store_execution(
                task_description, result, task_data.get('executor')
            )
        except Exception as mem_error:
            logger.error(f"Failed to store execution in memory: {mem_error}", exc_info=True)
            # Don't fail the task if memory storage fails
        
        # Update session state
        success = result.get('success', False)
        st.session_state.agent_status = 'success' if success else 'error'
        st.session_state.task_history.append({
            'task': task_description,
            'result': result,
            'timestamp': datetime.now(),
            'status': 'success' if success else 'error'
        })
        
        # Add completion log
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'success' if success else 'error',
            'message': f"Task completed: {result.get('message', 'No message')}"
        }
        st.session_state.execution_logs.append(log_entry)
        
        return result
        
    except Exception as e:
        error_msg = f"Task execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        st.session_state.agent_status = 'error'
        log_entry = {
            'timestamp': datetime.now(),
            'type': 'error',
            'message': error_msg
        }
        st.session_state.execution_logs.append(log_entry)
        
        return {'success': False, 'message': error_msg}

def main():
    """Main application logic"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Mhazili - AI Desktop Agent</h1>
        <p>Intelligent task automation and desktop control</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize services
    if not initialize_services():
        st.stop()
    
    # Sidebar
    render_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Control", "📊 Status", "🧠 Memory", "📝 Logs"])
    
    with tab1:
        st.header("Task Control Panel")
        
        # Task input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            task_result = render_task_input()
            
        with col2:
            st.subheader("Quick Actions")
            
            if st.button("🛑 Emergency Stop", type="secondary", use_container_width=True):
                st.session_state.agent_status = 'idle'
                st.session_state.current_task = None
                st.success("Agent stopped successfully")
            
            if st.button("🔄 Reset Memory", type="secondary", use_container_width=True):
                if st.session_state.get('memory_service'):
                    st.session_state.memory_service.clear_memory()
                    st.session_state.memory_data = []
                    st.success("Memory cleared")
            
            if st.button("📸 Take Screenshot", type="secondary", use_container_width=True):
                st.info("Screenshot functionality will be available after Stage 7")
        
        # Execute task if submitted
        if task_result and task_result.get('execute'):
            with st.spinner("Executing task..."):
                result = asyncio.run(execute_task(task_result['task']))
                
                if result.get('success'):
                    st.success(f"Task completed successfully: {result.get('message', '')}")
                else:
                    st.error(f"Task failed: {result.get('message', 'Unknown error')}")
    
    with tab2:
        st.header("Execution Status")
        render_execution_status()
        
        # Metrics dashboard
        st.subheader("Performance Metrics")
        render_metrics_dashboard()
    
    with tab3:
        st.header("Memory & Learning")
        render_memory_panel()
    
    with tab4:
        st.header("Execution Logs")
        render_logs_panel()
    
    # Settings modal
    if st.session_state.settings_open:
        render_settings_modal()
    
    # Auto-refresh functionality
    if st.session_state.auto_refresh and st.session_state.agent_status == 'running':
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()