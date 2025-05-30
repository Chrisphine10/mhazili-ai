from typing import Dict, Any, Union, List
from services.llm_service import LLMService
import json
import logging

logger = logging.getLogger(__name__)

class TaskRouter:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.llm_service = LLMService(config_path)

    def parse_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Uses LLM to understand the user's goal with their computer and generate a clear description.
        No rigid task_type classification — just focuses on describing the task meaningfully.
        """
        prompt = (
            f"You are an intelligent task interpreter for computer automation. A user has given a natural language command "
            f"to perform a task using their computer. Analyze the intent and describe the task clearly for an automation system.\n\n"
            f"Examples:\n"
            f"Input: 'Search for Python installation tutorials'\n"
            f"Output: {{\"description\": \"Search for Python installation tutorials on Google.\"}}\n\n"
            f"Input: 'Open Spotify and play my workout playlist'\n"
            f"Output: {{\"description\": \"Launch Spotify and play workout playlist.\"}}\n\n"
            f"Input: 'Uninstall Adobe Reader'\n"
            f"Output: {{\"description\": \"Uninstall Adobe Reader from the system.\"}}\n\n"
            f"Now interpret this input:\n"
            f"Input: '{user_input}'\n"
            f"Output:"
        )
        response = self.llm_service.summarize_content(prompt, max_length=300)

        try:
            parsed = eval(response) if isinstance(response, str) else response
            if not isinstance(parsed, dict) or "description" not in parsed:
                raise ValueError("Invalid format")
        except Exception:
            parsed = {"description": user_input}

        parsed["parameters"] = {}  # extendable in the future
        return parsed

    async def route_task(self, intent_or_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        intent = self.parse_intent(intent_or_input) if isinstance(intent_or_input, str) else intent_or_input

        return {
            "description": intent.get("description", ""),
            "executor": "computer_task_executor",
            "steps": await self._create_task_steps(intent)
        }

    async def _create_task_steps(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to generate task steps for automation from the task description.
        """
        prompt = (
            f"You are a task automation planner for desktop systems. Given a clear task description, "
            f"generate a list of structured steps to be executed programmatically. "
            f"Each step should be a dictionary with:\n"
            f"- 'action': name of the action (e.g., 'open_app', 'type', 'click', 'search_web', etc.)\n"
            f"- 'params': dictionary with any required parameters (e.g., 'path', 'text', 'url')\n"
            f"- Optional 'delay' between steps\n\n"
            f"Example:\n"
            f"Task: Launch Chrome and search for 'latest AI tools'\n"
            f"Output:\n"
            f"[{{\"action\": \"open_browser\", \"params\": {{\"url\": \"https://www.google.com\"}}, \"delay\": 1}},\n"
            f" {{\"action\": \"type\", \"params\": {{\"text\": \"latest AI tools\"}}, \"delay\": 0.5}},\n"
            f" {{\"action\": \"press_key\", \"params\": {{\"key\": \"enter\"}}}}]\n\n"
            f"Task: {intent['description']}\n"
            f"Output:"
        )
        
        try:
            response = self.llm_service.summarize_content(prompt, max_length=500)
            
            if not response or not response.content:
                logger.error("Empty response from LLM")
                return self._get_fallback_steps(intent)
            
            # Try to find JSON array in the response
            content = response.content.strip()
            if '[' in content and ']' in content:
                # Extract just the JSON array part
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                json_str = content[start_idx:end_idx]
                
                try:
                    steps = json.loads(json_str)
                    if isinstance(steps, list) and len(steps) > 0:
                        # Validate each step has required fields
                        validated_steps = []
                        for step in steps:
                            if isinstance(step, dict) and 'action' in step:
                                if 'params' not in step:
                                    step['params'] = {}
                                validated_steps.append(step)
                        
                        if validated_steps:
                            logger.info(f"Successfully generated {len(validated_steps)} steps")
                            return validated_steps
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
            
            logger.error("Could not extract valid steps from LLM response")
            return self._get_fallback_steps(intent)
            
        except Exception as e:
            logger.error(f"Error generating task steps: {e}")
            return self._get_fallback_steps(intent)
    
    def _get_fallback_steps(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback steps when LLM fails."""
        description = intent.get("description", "").lower()
        
        # Basic step generation based on common patterns
        steps = []
        
        if "chrome" in description or "browser" in description:
            steps.append({
                "action": "open_app",
                "params": {"app_name": "chrome"},
                "delay": 1
            })
        
        if "search" in description:
            # Extract search query if possible
            search_text = description.split("search for")[-1].strip().strip("'\"")
            if search_text:
                steps.append({
                    "action": "type",
                    "params": {"text": search_text},
                    "delay": 0.5
                })
                steps.append({
                    "action": "press_key",
                    "params": {"key": "enter"}
                })
        
        if not steps:
            # Default fallback
            steps = [{
                "action": "execute",
                "params": {
                    "text": intent.get("description", ""),
                    "delay": 0.5
                }
            }]
        
        return steps
