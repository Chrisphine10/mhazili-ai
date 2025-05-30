"""
Reusable Streamlit Components for AI Agent Dashboard
Contains all UI elements used across the application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

def render_sidebar():
    """Render the sidebar with navigation and controls"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/white?text=AI+Agent", width=200)
        
        st.markdown("---")
        
        # Agent status indicator
        status = st.session_state.get('agent_status', 'idle')
        status_colors = {
            'idle': '🟢',
            'running': '🟡',
            'error': '🔴',
            'success': '🟢'
        }
        
        st.markdown(f"**Status:** {status_colors.get(status, '⚪')} {status.title()}")
        
        if st.session_state.get('current_task'):
            st.markdown(f"**Current Task:** {st.session_state.current_task[:50]}...")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("🎛️ Controls")
        
        auto_refresh = st.checkbox(
            "Auto-refresh",
            value=st.session_state.get('auto_refresh', True),
            help="Automatically refresh the interface during task execution"
        )
        st.session_state.auto_refresh = auto_refresh
        
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.settings_open = True
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        
        total_tasks = len(st.session_state.get('task_history', []))
        successful_tasks = len([t for t in st.session_state.get('task_history', []) 
                              if t.get('status') == 'success'])
        
        st.metric("Total Tasks", total_tasks)
        st.metric("Success Rate", f"{(successful_tasks/max(total_tasks,1)*100):.1f}%")
        st.metric("Memory Items", len(st.session_state.get('memory_data', [])))

def render_task_input() -> Optional[Dict]:
    """Render task input form"""
    st.subheader("Enter Task Description")
    
    # Predefined task examples
    example_tasks = [
        "Open Chrome and search for 'AI news'",
        "Take a screenshot and save it to Desktop",
        "Open Calculator and compute 15 * 23",
        "Create a new text file on Desktop",
        "Find and open the latest downloaded file"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Task input methods
        input_method = st.radio(
            "Input Method:",
            ["Type Custom Task", "Select Example"],
            horizontal=True
        )
        
        if input_method == "Type Custom Task":
            task_input = st.text_area(
                "Describe what you want the agent to do:",
                height=100,
                placeholder="E.g., Open Chrome, navigate to Google, search for 'Python tutorials', and take a screenshot"
            )
        else:
            task_input = st.selectbox(
                "Choose an example task:",
                [""] + example_tasks
            )
    
    with col2:
        st.markdown("**Task Options:**")
        
        priority = st.selectbox("Priority", ["Normal", "High", "Low"])
        
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
        
        save_screenshots = st.checkbox("Save Screenshots", value=True)
        
        verbose_logging = st.checkbox("Verbose Logging", value=False)
    
    # Execution controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        execute_button = st.button(
            "🚀 Execute Task",
            type="primary",
            use_container_width=True,
            disabled=not task_input or st.session_state.get('agent_status') == 'running'
        )
    
    with col2:
        validate_button = st.button(
            "✅ Validate Only",
            type="secondary",
            use_container_width=True,
            disabled=not task_input
        )
    
    with col3:
        schedule_button = st.button(
            "⏰ Schedule",
            type="secondary",
            use_container_width=True,
            disabled=not task_input
        )
    
    if execute_button and task_input:
        return {
            'execute': True,
            'task': task_input,
            'priority': priority,
            'timeout': timeout,
            'save_screenshots': save_screenshots,
            'verbose_logging': verbose_logging
        }
    
    if validate_button and task_input:
        st.info("Task validation will be implemented in future iterations")
    
    if schedule_button and task_input:
        st.info("Task scheduling will be implemented in future iterations")
    
    return None

def render_execution_status():
    """Render current execution status and progress"""
    status = st.session_state.get('agent_status', 'idle')
    current_task = st.session_state.get('current_task')
    
    # Status container with custom styling
    status_class = f"status-{status}"
    
    if status == 'idle':
        st.markdown(f"""
        <div class="status-container status-idle">
            <h4>🟢 Agent Ready</h4>
            <p>The agent is ready to receive new tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif status == 'running':
        st.markdown(f"""
        <div class="status-container status-running">
            <h4>🟡 Task in Progress</h4>
            <p><strong>Current Task:</strong> {current_task}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar (simulated)
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
    
    elif status == 'error':
        st.markdown(f"""
        <div class="status-container status-error">
            <h4>🔴 Execution Error</h4>
            <p>The last task encountered an error. Check the logs for details.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif status == 'success':
        st.markdown(f"""
        <div class="status-container status-success">
            <h4>🟢 Task Completed</h4>
            <p>The last task was completed successfully.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent task history
    st.subheader("Recent Tasks")
    
    task_history = st.session_state.get('task_history', [])
    
    if task_history:
        # Display last 5 tasks
        recent_tasks = task_history[-5:]
        
        for i, task in enumerate(reversed(recent_tasks)):
            with st.expander(f"Task {len(task_history) - i}: {task['task'][:50]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {task['status']}")
                    st.write(f"**Timestamp:** {task['timestamp'].strftime('%H:%M:%S')}")
                
                with col2:
                    if task.get('result'):
                        st.json(task['result'])
    else:
        st.info("No tasks executed yet.")

def render_memory_panel():
    """Render memory and learning information"""
    memory_data = st.session_state.get('memory_data', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Recent Memory")
        
        if memory_data:
            for item in memory_data[-10:]:  # Show last 10 items
                with st.expander(f"Memory: {item.get('task', 'Unknown')[:30]}..."):
                    st.write(f"**Type:** {item.get('type', 'Unknown')}")
                    st.write(f"**Timestamp:** {item.get('timestamp', 'Unknown')}")
                    st.write(f"**Success:** {item.get('success', False)}")
                    if item.get('details'):
                        st.json(item['details'])
        else:
            st.info("No memory data available yet.")
    
    with col2:
        st.subheader("🧠 Learning Insights")
        
        if memory_data:
            # Success rate by task type
            task_types = {}
            for item in memory_data:
                task_type = item.get('type', 'Unknown')
                if task_type not in task_types:
                    task_types[task_type] = {'total': 0, 'success': 0}
                
                task_types[task_type]['total'] += 1
                if item.get('success'):
                    task_types[task_type]['success'] += 1
            
            # Create success rate chart
            if task_types:
                df = pd.DataFrame([
                    {
                        'Task Type': k,
                        'Success Rate': (v['success'] / v['total']) * 100,
                        'Total': v['total']
                    }
                    for k, v in task_types.items()
                ])
                
                fig = px.bar(
                    df,
                    x='Task Type',
                    y='Success Rate',
                    title='Success Rate by Task Type',
                    color='Success Rate',
                    color_continuous_scale='viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Learning insights will appear after task execution.")

def render_logs_panel():
    """Render execution logs and debugging information"""
    logs = st.session_state.get('execution_logs', [])
    
    # Log level filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_levels = st.multiselect(
            "Filter by Level:",
            ['info', 'success', 'warning', 'error'],
            default=['info', 'success', 'warning', 'error']
        )
    
    with col2:
        max_logs = st.selectbox("Show logs:", [50, 100, 200, 500], index=0)
    
    with col3:
        if st.button("🗑️ Clear Logs"):
            st.session_state.execution_logs = []
            st.success("Logs cleared!")
    
    # Display logs
    if logs:
        filtered_logs = [log for log in logs if log.get('type') in log_levels]
        recent_logs = filtered_logs[-max_logs:]
        
        for log in reversed(recent_logs):
            log_time = log.get('timestamp', datetime.now()).strftime('%H:%M:%S')
            log_type = log.get('type', 'info')
            log_message = log.get('message', 'No message')
            
            # Color coding for different log types
            if log_type == 'error':
                st.error(f"[{log_time}] {log_message}")
            elif log_type == 'warning':
                st.warning(f"[{log_time}] {log_message}")
            elif log_type == 'success':
                st.success(f"[{log_time}] {log_message}")
            else:
                st.info(f"[{log_time}] {log_message}")
    else:
        st.info("No logs available yet. Execute a task to see logs.")

def render_settings_modal():
    """Render settings configuration modal"""
    st.subheader("⚙️ Agent Settings")
    
    with st.expander("🤖 LLM Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            llm_provider = st.selectbox(
                "LLM Provider:",
                ["OpenAI", "Google Gemini"],
                index=0
            )
            
            model_name = st.selectbox(
                "Model:",
                ["gpt-4", "gpt-3.5-turbo"] if llm_provider == "OpenAI" 
                else ["gemini-pro", "gemini-pro-vision"]
            )
        
        with col2:
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens:", 100, 4000, 1000, 100)
    
    with st.expander("🖥️ Automation Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            screenshot_quality = st.selectbox("Screenshot Quality:", ["High", "Medium", "Low"])
            action_delay = st.slider("Action Delay (seconds):", 0.1, 3.0, 0.5, 0.1)
        
        with col2:
            retry_attempts = st.number_input("Retry Attempts:", 1, 10, 3)
            safety_mode = st.checkbox("Safety Mode", value=True)
    
    with st.expander("💾 Memory & Storage"):
        memory_limit = st.number_input("Memory Limit (items):", 10, 1000, 100, 10)
        auto_save = st.checkbox("Auto-save sessions", value=True)
        log_retention = st.selectbox("Log Retention:", ["1 day", "1 week", "1 month"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            # Save settings logic would go here
            st.success("Settings saved successfully!")
            st.session_state.settings_open = False
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.info("Settings reset to defaults")
    
    with col3:
        if st.button("❌ Cancel"):
            st.session_state.settings_open = False

def render_metrics_dashboard():
    """Render performance metrics and analytics"""
    task_history = st.session_state.get('task_history', [])
    
    if not task_history:
        st.info("Metrics will appear after task execution.")
        return
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_tasks = len(task_history)
    successful_tasks = len([t for t in task_history if t.get('status') == 'success'])
    avg_execution_time = 5.2  # Placeholder - would calculate from actual execution times
    uptime = "99.2%"  # Placeholder
    
    with col1:
        st.metric("Total Tasks", total_tasks, delta=1 if total_tasks > 0 else 0)
    
    with col2:
        success_rate = (successful_tasks / max(total_tasks, 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%", delta=f"{success_rate - 95:.1f}%")
    
    with col3:
        st.metric("Avg. Execution Time", f"{avg_execution_time:.1f}s", delta="-0.3s")
    
    with col4:
        st.metric("Uptime", uptime, delta="0.1%")
    
    # Task execution timeline
    if len(task_history) > 1:
        st.subheader("Task Execution Timeline")
        
        # Create timeline data
        timeline_data = []
        for i, task in enumerate(task_history[-20:]):  # Last 20 tasks
            timeline_data.append({
                'Task': f"Task {i+1}",
                'Timestamp': task['timestamp'],
                'Status': task['status'],
                'Duration': 5 + (i % 10)  # Simulated duration
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create timeline chart
        fig = px.scatter(
            df,
            x='Timestamp',
            y='Task',
            color='Status',
            size='Duration',
            title='Recent Task Execution Timeline',
            color_discrete_map={
                'success': 'green',
                'error': 'red',
                'running': 'orange'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_task_queue():
    """Render task queue management (for future implementation)"""
    st.subheader("📋 Task Queue")
    st.info("Task queue management will be implemented in future iterations.")
    
    # Placeholder for queued tasks
    st.text("• No queued tasks")
    st.text("• Queue management coming soon")

def render_system_monitor():
    """Render system resource monitoring"""
    st.subheader("🖥️ System Monitor")
    
    # Placeholder metrics (would use psutil in real implementation)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU Usage", "23%", delta="-2%")
    
    with col2:
        st.metric("Memory Usage", "1.2 GB", delta="0.1 GB")
    
    with col3:
        st.metric("Disk Usage", "45%", delta="1%")