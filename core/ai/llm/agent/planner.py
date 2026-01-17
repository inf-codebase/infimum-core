import openai
from core.utils import auto_config
import json
import logging
from .prompts import PLANNER_PROMPTS  # Import the centralized prompts

# Configure logging
logger = logging.getLogger(__name__)

class Planner:
    def __init__(self):
        self.system_prompt = """You are an AI assistant tasked with planning and reasoning about complex tasks.
                                Your job is to break down user queries into actionable steps and identify when tools should be used."""
        self.client = openai.OpenAI(api_key=auto_config.OPENAI_API_KEY)
        self.model = auto_config.OPENAI_MODEL
        logger.info("Initialized Planner")
    
    def decompose_task(self, query):
        """Decompose a task into smaller steps"""
        messages = [
            {"role": "system", "content": self._get_decomposition_prompt()},
            {"role": "user", "content": query}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        try:
            plan = json.loads(response.choices[0].message.content)
            return plan
        except json.JSONDecodeError:
            logger.error("Failed to parse task decomposition")
            return {"steps": []}
    
    def identify_tool_use(self, step):
        """Identify if a step requires using a specific tool"""
        logger.debug(f"Identifying tool use for step: {step}")
        tools_mapping = {
            "search": ["search", "find information", "look up", "research"],
            "calculator": ["calculate", "compute", "math", "equation"],
            "weather": ["weather", "temperature", "forecast", "rain"]
        }
        
        step_lower = step.lower()
        
        for tool_name, keywords in tools_mapping.items():
            if any(keyword in step_lower for keyword in keywords):
                logger.info(f"Identified tool '{tool_name}' for step: {step}")
                return tool_name, step
        
        logger.debug("No tool identified for step")
        return None, None
    
    def evaluate_step_completion(self, observation, step):
        """Evaluate if a single step is complete based on observation"""
        messages = [
            {"role": "system", "content": self._get_step_evaluation_prompt()},
            {"role": "user", "content": f"Step: {step}\nObservation: {observation}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        try:
            # Clean the response by removing markdown formatting
            cleaned_response = response.choices[0].message.content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove '```json'
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # Remove '```'
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove '```'
            cleaned_response = cleaned_response.strip()
            
            evaluation = json.loads(cleaned_response)
            reason = evaluation.get("reason", "")
            is_complete = evaluation.get("is_complete", False)
            return is_complete, reason 
        except json.JSONDecodeError:
            logger.error("Failed to parse step evaluation")
            return False, "Failed to parse step evaluation"
    
    def evaluate_completion(self, observations, current_step):
        """Evaluate if the overall task is complete based on all observations"""
        messages = [
            {"role": "system", "content": self._get_completion_evaluation_prompt()},
            {"role": "user", "content": f"Current step: {current_step}\nObservations:\n{json.dumps(observations, indent=2)}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        try:
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation.get("should_terminate", False)
        except json.JSONDecodeError:
            logger.error("Failed to parse completion evaluation")
            return False
    
    def generate_final_response(self, observations):
        """Generate a final response based on all observations"""
        messages = [
            {"role": "system", "content": self._get_final_response_prompt()},
            {"role": "user", "content": f"Observations:\n{json.dumps(observations, indent=2)}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def reflect_on_execution(self, original_query, steps, final_response, observations=None):
        """Generate a reflection on the execution process"""
        logger.info("Starting execution reflection...")
        reflection_prompt = f"""
        Original query: {original_query}
        
        Steps executed:
        {steps}
        
        Observations during execution:
        {observations if observations else 'No observations recorded'}
        
        Final response:
        {final_response}
        
        Reflect on this execution. What went well? What could be improved? Are there any learnings for future similar queries?
        """
        
        messages = [
            {"role": "system", "content": "Generate a reflection on the execution of a task."},
            {"role": "user", "content": reflection_prompt}
        ]
        
        try:
            logger.debug("Calling LLM for reflection...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            reflection = response.choices[0].message.content
            logger.info("Reflection completed successfully")
            logger.debug(f"Reflection content: {reflection}")
            return reflection
        except Exception as e:
            logger.error(f"Reflection error: {str(e)}")
            return f"Reflection error: {str(e)}"

    def _get_decomposition_prompt(self):
        """Get the system prompt for task decomposition"""
        return PLANNER_PROMPTS["decomposition"]

    def _get_step_evaluation_prompt(self):
        return PLANNER_PROMPTS["step_evaluation"]

    def _get_completion_evaluation_prompt(self):
        return PLANNER_PROMPTS["completion_evaluation"]

    def _get_final_response_prompt(self):
        return PLANNER_PROMPTS["final_response"]
