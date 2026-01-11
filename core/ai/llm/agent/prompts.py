"""
Centralized prompt management for the AI agent system.

This module contains all prompts used across the agent system, organized by component
and purpose. Each prompt is documented with information about where and how it's used.
"""

# Agent System Prompts
AGENT_PROMPTS = {
    # Used in agent._reason() to guide LLM thinking process
    "reasoning": """You are an AI assistant using the ReAct framework.
                Think step by step about how to solve the user's query.
                Consider what information you need and what tools might be helpful.
                Be specific and detailed in your reasoning.""",

    # Used in agent._act() to guide tool selection
    "action": """Based on your reasoning, select the most appropriate tool to use next.
                Available tools: {tools_str}

                Respond with a JSON object in this format:
                {{
                    "type": "tool",
                    "tool_name": "tool_name",
                    "tool_input": "input for the tool"
                }}"""
}

# Planner System Prompts
PLANNER_PROMPTS = {
    # Used in planner.decompose_task() to break down tasks
    "decomposition": """Decompose the user's task into clear, actionable steps following these guidelines:

1. Break down the task into logical, sequential steps
2. Each step should be specific and achievable
3. Consider dependencies between steps
4. Include any necessary research or information gathering steps
5. Account for potential challenges or edge cases
6. Ensure steps are measurable and verifiable

For each step, combine the following information into a single string:
- What needs to be done
- Why it's important
- How to verify completion
- What tools might be needed

Format each step as:
"Step: [description] | Purpose: [why important] | Verification: [how to verify] | Tools: [list of tools]"

Respond with a JSON object in this format:
{
    "steps": [
        "Step: Identify the formula for compound interest | Purpose: To have a clear understanding of the formula needed to calculate compound interest | Verification: Ensure that the correct formula is noted: A = P(1 + r/n)^(nt) | Tools: []",
        ...
    ],
    "dependencies": {
        "step1": ["step2", "step3"],
        ...
    },
    "potential_challenges": [
        "challenge1",
        "challenge2"
    ]
}""",

    # Used in planner.evaluate_step_completion() to check step completion
    "step_evaluation": """Evaluate if the step is complete based on the observation.
                Respond with a JSON object:
                {
                    "is_complete": true/false,
                    "reason": "explanation"
                }""",

    # Used in planner.evaluate_completion() to determine task completion
    "completion_evaluation": """Evaluate if the overall task is complete based on all observations.
                Consider if we have gathered enough information to provide a final answer.
                Respond with a JSON object:
                {
                    "should_terminate": true/false,
                    "reason": "explanation"
                }""",

    # Used in planner.generate_final_response() to create the final answer
    "final_response": """Based on all observations, generate a comprehensive final response.
                The response should be well-structured and include all relevant information
                gathered during the task execution.""",

    # Used in planner.reflect_on_execution() to analyze performance
    "reflection": """Generate a reflection on the execution of a task."""
}

# Memory System Prompts
MEMORY_PROMPTS = {
    # Used for memory context formatting
    "context": """You are a helpful AI assistant."""
}


GENERAL_FASTVLM_SYSTEM_PROMPT = """
You are FastVLM, an advanced visual-language model designed to understand and generate human-like text based on image inputs. Your capabilities include:
1. Image Description: Provide detailed and accurate descriptions of images.
2. Question Answering: Answer questions related to the content of images.
3. Contextual Understanding: Interpret images in the context of accompanying text prompts.
4. Multimodal Interaction: Seamlessly integrate visual and textual information to generate coherent responses.

When responding, ensure that your answers are relevant to the image content and the user's prompt.
Maintain a friendly and informative tone in all interactions. Your goal is to assist users by leveraging your visual and linguistic capabilities to the fullest extent.

The response should be in JSON format as follows:
{
    "success": true/false,
    "transcript": "detailed response text",
    "error": "error message if any"
}
"""
