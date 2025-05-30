import json
import time
import logging
import asyncio
from typing import Dict, Any, List
from services.gui_automation import GUIAutomationService, ClickType, KeyAction
from services.llm_service import LLMService
from services.screenshot_service import ScreenshotService
from utils.logger import get_logger

# Initialize logger
logger = get_logger(
    name=__name__,
    level=logging.INFO,
    log_file="logs/streamlit_ui.log",
    format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TaskExecutor:
    def __init__(self, config_path: str = "config/settings.yaml"):        
        # Initialize services
        self.gui_service = GUIAutomationService({
            'fail_safe': True,
            'pause_duration': 0.1,
            'screenshot_on_action': True
        })
        
        self.screenshot_service = ScreenshotService()
        
        # Initialize AI service
        try:
            self.llm_service = LLMService(config_path)
            self.ai_enabled = True
            logger.info("AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            self.llm_service = None
            self.ai_enabled = False
        
        # Initialize command history
        self.command_history = []
        self.last_error = None
        self.retry_count = 0
        self.max_retries = 3
        
        logger.info("Simple AI Task Executor initialized")

    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """
        Execute a task based on natural language description using AI guidance.
        
        Args:
            task_description: Natural language description of the task or task data dictionary
            
        Returns:
            Dict containing execution results
        """
        if not self.ai_enabled:
            return {
                'success': False,
                'message': 'AI service not available',
                'steps_executed': 0
            }
        
        try:
            # Extract description from input if it's a dictionary
            if isinstance(task_description, dict):
                description = task_description.get('description', '')
                if not description:
                    return {
                        'success': False,
                        'message': 'No task description provided',
                        'steps_executed': 0
                    }
            else:
                description = task_description
            
            logger.info(f"Starting task: {description}")
            
            # Check memory for similar tasks
            memory_check = await self._check_task_memory(description)
            if memory_check.get('found'):
                logger.info(f"Found similar task in memory: {memory_check.get('similar_task')}")
                if memory_check.get('success'):
                    logger.info("Previous successful execution found, reusing steps")
                    steps = memory_check.get('steps', [])
                else:
                    logger.info("Previous failed execution found, attempting with modifications")
                    steps = await self._generate_task_steps(description, memory_check.get('failed_steps', []))
            else:
                steps = await self._generate_task_steps(description)
            
            if not steps:
                return {
                    'success': False,
                    'message': 'Failed to generate task steps',
                    'ai_analysis': 'Could not break down the task'
                }
            
            logger.info(f"Generated {len(steps)} steps for execution")
            
            # Take initial screenshot
            initial_screenshot = self.screenshot_service.capture_screenshot()
            screenshots = [initial_screenshot]
            
            # Step 2: Execute each step with verification
            executed_steps = 0
            execution_log = []
            failed_steps = []
            
            for i, step in enumerate(steps):
                logger.info(f"Executing step {i + 1}: {step.get('description', 'Unknown step')}")
                
                # Execute step with retries
                step_success = False
                for retry in range(self.max_retries):
                    step_result = await self._execute_step(step)
                    
                    # Verify step completion
                    verification_result = await self._verify_step_completion(step, screenshots[-1])
                    
                    if step_result and verification_result:
                        step_success = True
                        break
                    else:
                        logger.warning(f"Step {i + 1} verification failed, attempt {retry + 1}/{self.max_retries}")
                        await asyncio.sleep(1)  # Wait before retry
                
                execution_log.append({
                    'step_number': i + 1,
                    'description': step.get('description', ''),
                    'success': step_success,
                    'timestamp': time.time(),
                    'verification_result': verification_result
                })
                
                if step_success:
                    executed_steps += 1
                    # Small delay between successful steps
                    await asyncio.sleep(0.5)
                else:
                    failed_steps.append(step)
                    logger.error(f"Step {i + 1} failed after {self.max_retries} attempts")
                    break
            
            # Take final screenshot
            final_screenshot = self.screenshot_service.capture_screenshot()
            screenshots.append(final_screenshot)
            
            # Verify overall task completion
            verification_result = await self.verify_task_completion(description, screenshots)
            
            success = executed_steps == len(steps) and verification_result
            
            # Store execution in memory
            await self._store_task_execution(
                description=description,
                steps=steps,
                success=success,
                execution_log=execution_log,
                screenshots=screenshots,
                failed_steps=failed_steps
            )
            
            return {
                'success': success,
                'message': f'Executed {executed_steps} of {len(steps)} steps',
                'total_steps': len(steps),
                'steps_executed': executed_steps,
                'execution_log': execution_log,
                'task_description': description,
                'verification_result': verification_result,
                'screenshots': screenshots,
                'failed_steps': failed_steps,
                'memory_reference': memory_check.get('memory_id')
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                'success': False,
                'message': f'Task execution error: {str(e)}',
                'task_description': task_description if isinstance(task_description, str) else str(task_description)
            }

    async def _check_task_memory(self, task_description: str) -> Dict[str, Any]:
        """Check memory for similar tasks and their execution history."""
        try:
            # Use AI to analyze task similarity
            prompt = f"""
            Compare this task with previous executions in memory:
            Task: {task_description}
            
            Return a JSON response:
            {{
                "found": true/false,
                "similarity_score": 0.0-1.0,
                "similar_task": "description of similar task",
                "success": true/false,
                "steps": [previous steps if successful],
                "failed_steps": [steps that failed],
                "memory_id": "id of memory entry"
            }}
            """
            
            response = self.llm_service.extract_information(prompt, "task_memory")
            
            if not response or not response.content:
                return {"found": False}
            
            try:
                memory_data = json.loads(response.content)
                return memory_data
            except json.JSONDecodeError:
                return {"found": False}
                
        except Exception as e:
            logger.error(f"Failed to check task memory: {e}")
            return {"found": False}

    async def _store_task_execution(self, description: str, steps: List[Dict], 
                                  success: bool, execution_log: List[Dict],
                                  screenshots: List[str], failed_steps: List[Dict]) -> None:
        """Store task execution details in memory."""
        try:
            memory_entry = {
                'task_description': description,
                'steps': steps,
                'success': success,
                'execution_log': execution_log,
                'screenshots': screenshots,
                'failed_steps': failed_steps,
                'timestamp': time.time()
            }
            
            # Store in memory service
            if hasattr(self, 'memory_service'):
                await self.memory_service.store_execution(description, memory_entry)
            
        except Exception as e:
            logger.error(f"Failed to store task execution: {e}")

    async def _verify_step_completion(self, step: Dict[str, Any], screenshot: str) -> bool:
        """Verify if a specific step was completed successfully."""
        try:
            action = step.get('action', '')
            params = step.get('params', {})
            
            # Create verification prompt
            prompt = f"""
            Verify if this step was completed successfully:
            Action: {action}
            Parameters: {json.dumps(params)}
            Screenshot: {screenshot}
            
            Return a JSON response:
            {{
                "success": true/false,
                "confidence": 0.0-1.0,
                "verification_details": "explanation"
            }}
            """
            
            response = self.llm_service.extract_information(prompt, "step_verification")
            
            if not response or not response.content:
                return False
            
            try:
                verification = json.loads(response.content)
                return verification.get('success', False)
            except json.JSONDecodeError:
                return False
                
        except Exception as e:
            logger.error(f"Step verification failed: {e}")
            return False

    async def _execute_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single step using the GUI automation service."""
        try:
            action = step.get('action', '')
            target = step.get('target', '')
            params = step.get('parameters', {})
            
            logger.info(f"Executing: {action} on {target}")
            
            # Skip if duplicate command
            if self.command_history and self.command_history[-1] == step:
                logger.info("Skipping duplicate command")
                return True
            
            # Execute action
            result = False
            if action == 'open_app':
                app_name = params.get('app_name', '')
                if self._is_app_open(app_name):
                    logger.info(f"App {app_name} is already open")
                    return True
                result = self._execute_open_app_action(params)
            elif action == 'click':
                result = self._execute_click_action(params)
            elif action == 'type':
                result = self._execute_type_action(params)
            elif action == 'key_press':
                result = self._execute_key_press_action(params)
            elif action == 'wait':
                result = self._execute_wait_action(params)
            else:
                logger.warning(f"Unknown action: {action}")
                return True
            
            # Handle result
            if result:
                self.command_history.append(step)
                self.retry_count = 0
                
                # Add wait time if specified
                wait_after = params.get('wait_after', 0)
                if wait_after > 0:
                    await asyncio.sleep(wait_after)
            else:
                self.last_error = f"Failed to execute {action} on {target}"
                self.retry_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            self.last_error = str(e)
            self.retry_count += 1
            return False

    def _is_app_open(self, app_name: str) -> bool:
        """Check if an application is already open."""
        try:
            # Take a screenshot
            screenshot = self.screenshot_service.capture_screenshot()
            
            # Use LLM to analyze if the app window is visible
            prompt = f"""Analyze if {app_name} is already open and visible in this screenshot.
Return JSON: {{"is_open": true|false, "confidence": 0.0-1.0}}"""
            
            response = self.llm_service.analyze_intent(prompt)
            if response and response.content:
                try:
                    result = json.loads(response.content)
                    return result.get('is_open', False)
                except json.JSONDecodeError:
                    pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if app is open: {e}")
            return False

    def _execute_click_action(self, params: Dict[str, Any]) -> bool:
        """Execute click action."""
        try:
            x = params.get('x', 500)
            y = params.get('y', 300)
            click_type = params.get('click_type', 'left')
            
            # Convert click type
            if click_type == 'right':
                click_enum = ClickType.RIGHT
            elif click_type == 'double':
                click_enum = ClickType.DOUBLE
            elif click_type == 'middle':
                click_enum = ClickType.MIDDLE
            else:
                click_enum = ClickType.LEFT
            
            return self.gui_service.click(x, y, click_enum)
            
        except Exception as e:
            logger.error(f"Click action failed: {e}")
            return False

    def _execute_type_action(self, params: Dict[str, Any]) -> bool:
        """Execute type action."""
        try:
            text = params.get('text', '')
            interval = params.get('interval', 0.01)
            
            if not text:
                logger.warning("No text provided for type action")
                return True
            
            return self.gui_service.type_text(text, interval)
            
        except Exception as e:
            logger.error(f"Type action failed: {e}")
            return False

    def _execute_key_press_action(self, params: Dict[str, Any]) -> bool:
        """Execute key press action."""
        try:
            key = params.get('key', 'enter')
            presses = params.get('presses', 1)
            interval = params.get('interval', 0.0)
            
            return self.gui_service.press_key(key, presses, interval)
            
        except Exception as e:
            logger.error(f"Key press action failed: {e}")
            return False

    def _execute_wait_action(self, params: Dict[str, Any]) -> bool:
        """Execute wait action."""
        try:
            seconds = params.get('seconds', 1.0)
            self.gui_service.wait(seconds)
            return True
            
        except Exception as e:
            logger.error(f"Wait action failed: {e}")
            return False

    def _execute_open_app_action(self, params: Dict[str, Any]) -> bool:
        """Execute open application action."""
        try:
            app_name = params.get('app_name', '') or params.get('name', '')
            
            if not app_name:
                logger.error("No application name provided")
                return False
            
            return self.gui_service.open_application(app_name)
            
        except Exception as e:
            logger.error(f"Open app action failed: {e}")
            return False

    async def get_ai_suggestions(self, task_description: str) -> Dict[str, Any]:
        """Get AI suggestions for a task without executing it."""
        if not self.ai_enabled:
            return {
                'suggestions': 'AI service not available',
                'feasibility': 'unknown'
            }
        
        try:
            prompt = f"""
            Analyze this automation task and provide suggestions: "{task_description}"
            
            Consider:
            1. Is this task feasible with GUI automation?
            2. What are the main challenges?
            3. What information might be needed from the user?
            4. Are there any risks or considerations?
            5. Alternative approaches?
            
            Provide practical advice and recommendations.
            """
            
            response = self.llm_service.summarize_content(prompt, max_length=500)
            
            return {
                'task_description': task_description,
                'ai_suggestions': response.content,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get AI suggestions: {e}")
            return {
                'suggestions': f'Error getting suggestions: {str(e)}',
                'feasibility': 'unknown'
            }

    async def verify_task_completion(self, task_description: str, screenshots: List[str]) -> bool:
        """
        Verify task completion using screenshots and AI analysis.
        
        Args:
            task_description: Original task description
            screenshots: List of screenshot paths taken during execution
            
        Returns:
            bool: True if task was completed successfully
        """
        if not self.ai_enabled or not screenshots:
            return True  # Skip verification if AI is disabled or no screenshots
            
        try:
            # Create verification prompt
            prompt = f"""
            Analyze these screenshots to verify if the following task was completed successfully:
            Task: {task_description}
            
            Consider:
            1. Are the expected UI elements visible?
            2. Is the text entered correctly?
            3. Are the windows/applications in the expected state?
            4. Are there any error messages or unexpected states?
            
            Return a JSON response:
            {{
                "success": true/false,
                "confidence": 0.0-1.0,
                "issues": ["list of any issues found"],
                "verification_details": "detailed explanation"
            }}
            """
            
            # Get AI analysis
            response = self.llm_service.extract_information(prompt, "verification")
            
            if not response or not response.content:
                logger.error("Empty verification response from AI")
                return True  # Default to success if verification fails
                
            try:
                verification = json.loads(response.content)
                success = verification.get('success', True)
                
                if not success:
                    logger.warning(f"Task verification failed: {verification.get('verification_details', '')}")
                    logger.warning(f"Issues found: {verification.get('issues', [])}")
                
                return success
                
            except json.JSONDecodeError:
                logger.error("Failed to parse verification response")
                return True
                
        except Exception as e:
            logger.error(f"Task verification failed: {e}")
            return True  # Default to success if verification fails

    async def _generate_task_steps(self, task_description: str, 
                                 previous_failed_steps: List[Dict] = None) -> List[Dict[str, Any]]:
        """Generate step-by-step execution plan using AI."""
        try:
            # Simple prompt focused on the task
            prompt = f"""Task: {task_description}

Generate steps to complete this task. Use these actions:
- open_app: To open applications
- click: To click buttons or UI elements
- type: To type text or numbers
- key_press: To press keyboard keys
- wait: To wait between actions

Example for calculator:
1. Open calculator
2. Type first number
3. Click operation button
4. Type second number
5. Click equals

Return steps in this format:
[
    {{"action": "open_app", "target": "app name", "parameters": {{"app_name": "name"}}}},
    {{"action": "type", "target": "input", "parameters": {{"text": "text to type"}}}},
    {{"action": "click", "target": "button", "parameters": {{"x": 500, "y": 300}}}}
]"""
            
            # Use summarize_content instead of analyze_intent
            response = self.llm_service.summarize_content(prompt, max_length=1000)
            
            if not response or not response.content:
                logger.error("Empty response from LLM")
                return []
            
            # Simple JSON parsing
            try:
                content = response.content.strip()
                # Remove markdown if present
                if '```' in content:
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                content = content.strip()
                
                steps = json.loads(content)
                
                # Basic validation
                if not isinstance(steps, list):
                    return []
                
                # Process steps
                processed_steps = []
                for step in steps:
                    if not isinstance(step, dict) or 'action' not in step:
                        continue
                        
                    # Ensure required fields
                    if 'parameters' not in step:
                        step['parameters'] = {}
                    if 'target' not in step:
                        step['target'] = step.get('description', '')
                    
                    # Add description
                    step['description'] = f"Execute {step['action']} on {step['target']}"
                    
                    # Add appropriate delays
                    if step['action'] == 'open_app':
                        step['parameters']['wait_after'] = 2
                    elif step['action'] == 'click':
                        step['parameters']['wait_after'] = 0.5
                    elif step['action'] == 'type':
                        step['parameters']['wait_after'] = 0.2
                    
                    processed_steps.append(step)
                
                if processed_steps:
                    logger.info(f"Generated {len(processed_steps)} steps")
                    return processed_steps
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return []
            
            return []
            
        except Exception as e:
            logger.error(f"Step generation failed: {e}")
            return []

    async def _parse_natural_language_steps(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse natural language response into structured steps."""
        try:
            # Simple parsing - look for numbered steps
            steps = []
            lines = ai_response.split('\n')
            
            current_step = None
            for line in lines:
                line = line.strip()
                
                # Look for numbered steps (1., 2., etc.)
                if line and (line[0].isdigit() or line.startswith('Step')):
                    if current_step:
                        steps.append(current_step)
                    
                    # Extract action from the step description
                    action, params = self._extract_action_from_description(line)
                    current_step = {
                        'description': line,
                        'action': action,
                        'params': params
                    }
                elif current_step and line:
                    # Additional details for current step
                    current_step['description'] += f" {line}"
            
            if current_step:
                steps.append(current_step)
            
            return steps
            
        except Exception as e:
            logger.error(f"Failed to parse natural language steps: {e}")
            return []

    def _extract_action_from_description(self, description: str) -> tuple:
        """Extract action type and parameters from step description."""
        description_lower = description.lower()
        
        # Simple keyword matching
        if 'click' in description_lower:
            return 'click', {'x': 500, 'y': 300}  # Default coordinates
        elif 'type' in description_lower or 'enter' in description_lower:
            return 'type', {'text': 'example text'}
        elif 'press' in description_lower and 'key' in description_lower:
            return 'key_press', {'key': 'enter'}
        elif 'open' in description_lower and 'app' in description_lower:
            return 'open_app', {'app_name': 'notepad'}
        elif 'wait' in description_lower:
            return 'wait', {'seconds': 2}
        elif 'scroll' in description_lower:
            return 'scroll', {'x': 500, 'y': 400, 'direction': 1}
        else:
            return 'wait', {'seconds': 1}  # Default action


# Example usage
async def main():
    """Example usage of the Simple AI Task Executor."""
    executor = TaskExecutor()
    
    # Example task descriptions
    tasks = [
        "Open Notepad and type 'Hello World'",
        "Take a screenshot",
        "Open calculator and calculate 2+2",
        "Copy some text and paste it somewhere else"
    ]
    
    for task in tasks:
        print(f"\n--- Executing Task: {task} ---")
        
        # Get AI suggestions first
        suggestions = await executor.get_ai_suggestions(task)
        print(f"AI Suggestions: {suggestions.get('ai_suggestions', 'None')}")
        
        # Execute the task
        result = await executor.execute_task(task)
        print(f"Result: {result}")
        
        # Wait between tasks
        await asyncio.sleep(2)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(main())