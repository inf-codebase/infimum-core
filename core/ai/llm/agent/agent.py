import uuid
import openai
import json
import logging
from .planner import Planner
from .memory import Memory
from core.utils import auto_config
from .tools import ToolManager
from loguru import logger
from .prompts import AGENT_PROMPTS  # Import the centralized prompts

class Agent:
    def __init__(self, tools=None, model=auto_config.OPENAI_MODEL):
        """
        example: write down 4 strengths and 4 challenges of the Delta Wellington Ultra Short Treasury On-Chain Fund token ($ULTRA)

        Args:
            tools (_type_, optional): _description_. Defaults to None.
            model (_type_, optional): _description_. Defaults to auto_config.OPENAI_MODEL.
        """
        self.id = str(uuid.uuid4())
        self.model = model
        self.planner = Planner()
        self.memory = Memory()
        self.tool_manager = ToolManager(tools or [])
        self.client = openai.OpenAI(api_key=auto_config.OPENAI_API_KEY)
        self.max_iterations = 5  # Maximum number of ReAct iterations
        logger.info(f"Initialized Agent {self.id} with model {model}")

    def run(self, query):
        """Main method to process a user query using ReAct framework"""
        logger.info(f"Starting to process query: {query}")

        # Clear short-term memory and store the query in memory
        self.memory.short_term = []
        self.memory.add_to_short_term({"role": "user", "content": query})
        logger.debug("Stored query in short-term memory")

        # Use planner to get initial task decomposition
        logger.info("Decomposing task into steps...")
        plan = self.planner.decompose_task(query)
        steps = plan.get("steps", [])
        # Format steps as bullet points
        steps_formatted = "\n".join([f"    • {step}" for step in steps])
        logger.info(f"Task decomposed into {len(steps)} steps:\n{steps_formatted}")
        
        # Initialize ReAct loop
        iteration = 0
        final_response = ""
        observations = []
        current_step_index = 0

        while iteration < self.max_iterations and current_step_index < len(steps):
            logger.info(f"\n=== Iteration {iteration + 1} ===")
            current_step = steps[current_step_index]
            logger.info(f"Current step ({current_step_index + 1}/{len(steps)}): {current_step}")

            # Store current step in memory
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Current step: {current_step}"
            })

            # 1. Reasoning phase
            logger.info("REASONING:")
            # Get relevant context from memory
            context = self.memory.get_relevant_context(query, current_step)
            thought = self._reason(query=current_step, observations=observations, current_step=query, context=context)
            logger.info(f"THOUGH: {thought}")
            
            # Store thought in memory
            self.memory.add_to_short_term({
                "role": "assistant",
                "content": f"Thought: {thought}"
            })

            # 2. Acting phase
            logger.info("ACTION:")
            action = self._act(thought)
            logger.info(f"Action decided: {json.dumps(action, indent=2)}")
            
            # Store action in memory
            self.memory.add_to_short_term({
                "role": "assistant",
                "content": f"Action: {json.dumps(action, indent=2)}"
            })

            # 3. Observation phase
            logger.info("OBSERVATION:")
            observation = self._observe(action)
            logger.info(f"Observation: {observation}")
            observations.append(observation)
            
            # Store observation in memory
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Observation: {observation}"
            })

            # 4. Reflection phase - Check completion and termination
            logger.info("REFLECTION:")
            is_complete, reason = self._is_step_complete(observation, current_step)
            logger.info(f"Step complete: {is_complete} - {reason}")
            
            # Store reflection in memory
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Step completion status: {is_complete}"
            })

            if is_complete:
                # Check if we should terminate based on all observations
                if self._should_terminate(observations, current_step):
                    logger.info("Termination condition met, preparing final response")
                    final_response = self.planner.generate_final_response(observations)
                    break
                    
                current_step_index += 1
                iteration = 0  # Reset iteration counter for new step
                logger.info(f"Moving to next step: {current_step_index + 1}/{len(steps)}")
            else:
                iteration += 1  # Only increment if step is not complete

        # Add final response to memory
        logger.info("Storing final response in memory")
        
        # If final_response is empty, analyze observations to generate a response
        if not final_response and observations:
            logger.info("Final response is empty, analyzing observations to generate response")
            
            # Get the original query from memory
            original_query = next((msg["content"] for msg in self.memory.short_term if msg["role"] == "user"), "")
            
            # Prepare context for analysis
            analysis_context = {
                "query": original_query,
                "observations": observations,
                "steps": steps
            }
            
            # Call LLM to analyze and generate response from observations
            messages = [
                {"role": "system", "content": """You are an AI assistant that needs to generate a final response 
                by analyzing the available observations. Your task is to:
                1. Review all observations carefully
                2. Identify the most relevant information that answers the original query
                3. Synthesize the information into a coherent response
                4. If there are gaps in information, acknowledge them
                5. Format the response in a clear and organized manner"""},
                {"role": "user", "content": f"""Original query: {original_query}
                
                Please analyze these observations and generate a comprehensive response:
                {json.dumps(observations, indent=2)}"""}
            ]
            
            final_response = self._call_llm(messages)
            logger.info("Generated final response from observations analysis")
        
        self.memory.add_to_short_term({
            "role": "assistant",
            "content": final_response
        })

        # Use planner for reflection
        logger.info("Starting reflection phase...")
        reflection = self.planner.reflect_on_execution(
            query,
            steps,
            final_response,
            observations
        )
        logger.info(f"Reflection completed: {reflection}")

        # Store in long-term memory
        logger.info("Storing execution details in long-term memory")
        self.memory.add_to_long_term({
            "query": query,
            "response": final_response,
            "observations": observations,
            "reflection": reflection,
            "steps": steps,
            "total_iterations": iteration
        })

        logger.info("Execution completed successfully")
        return final_response

    def _reason(self, query, observations, current_step, context=None):
        """Generate thoughts about what to do next"""
        logger.debug("Preparing reasoning context...")
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": query},
            {"role": "system", "content": f"Current step: {current_step}"}
        ]

        # Add context from memory if available
        if context:
            messages.append({"role": "system", "content": f"Relevant context from memory:\n{context}"})

        # Add previous observations if any
        if observations:
            logger.debug(f"Adding {len(observations)} previous observations to context")
            obs_context = "\nPrevious observations:\n"
            for i, obs in enumerate(observations):
                obs_context += f"{i+1}. {obs}\n"
            messages.append({"role": "system", "content": obs_context})

        logger.debug("Calling LLM for reasoning...")
        response = self._call_llm(messages)
        return response

    def _act(self, thought):
        """Determine the next action based on the thought"""
        logger.debug("Preparing action context...")
        
        # Get relevant context from memory for action selection
        action_context = self.memory.get_relevant_context(thought, "action", context_type="action")
        
        messages = [
            {"role": "system", "content": self._get_action_prompt()},
            {"role": "user", "content": thought}
        ]
        
        # Add relevant context from memory if available
        if action_context:
            messages.append({"role": "system", "content": action_context})

        response = self._call_llm(messages)

        try:
            # Clean the response by removing markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove '```json'
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # Remove '```'
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove '```'
            cleaned_response = cleaned_response.strip()
            
            # Parse the action from the response
            action = json.loads(cleaned_response)
            logger.debug("Successfully parsed action from response")
            
            # Store the action and its context in memory
            self.memory.add_to_short_term({
                "role": "assistant",
                "content": f"Action selected: {json.dumps(action, indent=2)}",
                "metadata": {
                    "action_type": action.get("type", "unknown"),
                    "tool_name": action.get("tool_name", "none"),
                    "thought": thought
                }
            })
            
            return action
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse action as JSON: {str(e)}")
            logger.warning(f"Raw response: {response}")
            # Store the error in memory
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Action parsing error: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "raw_response": response
                }
            })
            return {
                "type": "final_answer",
                "content": response
            }

    def _observe(self, action):
        """Execute the action and observe the result"""
        logger.debug(f"Executing action: {json.dumps(action, indent=2)}")

        # Get relevant context from memory for observation
        observation_context = self.memory.get_relevant_context(action, "observation", context_type="observation")

        if action["type"] == "tool":
            logger.info(f"Executing tool: {action['tool_name']}")
            # Execute tool
            tool_result = self.tool_manager.execute_tool(
                action["tool_name"],
                action["tool_input"]
            )
            logger.info(f"Tool execution result: {tool_result}")
            
            # Store tool execution result in memory
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Tool execution result: {tool_result}",
                "metadata": {
                    "tool_name": action["tool_name"],
                    "tool_input": action["tool_input"],
                    "context": observation_context
                }
            })
            
            return f"Tool '{action['tool_name']}' result: {tool_result}"
        elif action["type"] == "final_answer":
            logger.info("Received final answer")
            # Store final answer in memory
            self.memory.add_to_short_term({
                "role": "assistant",
                "content": f"Final answer: {action['content']}",
                "metadata": {
                    "type": "final_answer",
                    "content": action["content"]
                }
            })
            return f"Final answer: {action['content']}"
        else:
            logger.warning(f"Unknown action type: {action['type']}")
            # Store unknown action type in memory
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Unknown action type: {action['type']}",
                "metadata": {
                    "error": "unknown_action_type",
                    "action": action
                }
            })
            return f"Unknown action type: {action['type']}"

    def _is_step_complete(self, observation, step):
        """Check if the current step is complete based on observation"""
        # Get relevant context from memory for step completion evaluation
        completion_context = self.memory.get_relevant_context(step, "completion", context_type="completion")
        
        # Store evaluation context in memory
        if completion_context:
            logger.debug(f"Completion context: {completion_context}")
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Step completion evaluation: {completion_context}",
                "metadata": {
                    "step": step,
                    "observation": observation
                }
            })
        
        is_complete, reason = self.planner.evaluate_step_completion(observation, step)
        
        # Store completion status in memory
        self.memory.add_to_short_term({
            "role": "system",
            "content": f"Step completion status: {is_complete} : {reason}",
            "metadata": {
                "step": step,
                "observation": observation,
                "is_complete": is_complete
            }
        })
        
        return is_complete, reason

    def _should_terminate(self, observations, current_step):
        """Check if we should terminate based on observations and current step"""
        # Get relevant context from memory for termination decision
        termination_context = self.memory.get_relevant_context(current_step, "termination", context_type="termination")
        
        # Store termination context in memory
        if termination_context:
            logger.debug(f"Termination context: {termination_context}")
            self.memory.add_to_short_term({
                "role": "system",
                "content": f"Termination evaluation: {termination_context}",
                "metadata": {
                    "step": current_step,
                    "observations": observations
                }
            })
        
        should_terminate = self.planner.evaluate_completion(observations, current_step)
        
        # Store termination decision in memory
        self.memory.add_to_short_term({
            "role": "system",
            "content": f"Termination decision: {should_terminate}",
            "metadata": {
                "step": current_step,
                "observations": observations,
                "should_terminate": should_terminate
            }
        })
        
        return should_terminate

    def _get_system_prompt(self):
        """Get the system prompt for reasoning"""
        return AGENT_PROMPTS["reasoning"]

    def _get_action_prompt(self):
        """Get the system prompt for action selection"""
        available_tools = self.tool_manager.get_available_tools()
        tools_str = ", ".join(available_tools)

        return AGENT_PROMPTS["action"].format(tools_str=tools_str)

    def _call_llm(self, messages):
        """Call the language model API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            result = response.choices[0].message.content
            return result
        except Exception as e:
            logger.error(f"Error calling language model: {str(e)}")
            return f"Error calling language model: {str(e)}"
