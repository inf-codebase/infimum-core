"""
LangGraph workflow implementation for the AI Agent.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentFinish
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from .state import AgentState, update_state, should_terminate
from .types import AgentConfig, Task, TaskStatus, ToolResult
from ..memory import MemoryManager
from ..utils.logging import get_logger


logger = get_logger(__name__)


def create_agent_workflow(
    llm,
    tools: List[BaseTool],
    memory_manager: Optional[MemoryManager] = None,
    config: Optional[AgentConfig] = None,
    custom_workflows: Optional[Dict[str, Callable]] = None,
) -> StateGraph:
    """
    Create the main agent workflow using LangGraph.
    
    The workflow follows this pattern:
    1. Plan - Analyze query and create tasks
    2. Execute - Execute tasks using tools
    3. Reflect - Evaluate results and decide next action
    4. Respond - Generate final response
    """
    
    config = config or AgentConfig()
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. You have access to tools to help answer questions.
        
        Current conversation context:
        {memory_context}
        
        Use the following approach:
        1. Analyze the user's query carefully
        2. Break down complex questions into steps
        3. Use appropriate tools when needed
        4. Provide clear, helpful responses
        
        If you need to use tools, explain what you're doing and why.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Define workflow nodes
    def start_node(state: AgentState) -> AgentState:
        """Initialize the workflow."""
        logger.debug("Starting agent workflow")
        
        # Add system message
        system_msg = SystemMessage(content="Starting task execution...")
        messages = list(state["messages"])
        messages.append(system_msg)
        
        return update_state(
            state,
            messages=messages,
            next_action="plan",
        )
    
    def plan_node(state: AgentState) -> AgentState:
        """Plan the execution by analyzing the query and creating tasks."""
        logger.debug("Planning execution")
        
        try:
            query = state["user_query"]
            
            # Get memory context if available
            memory_context = ""
            if memory_manager:
                memory_context = memory_manager.get_context_for_prompt(query)
            
            # Simple task planning (can be enhanced with more sophisticated planning)
            planning_prompt = f"""
            Analyze this query and break it down into actionable tasks: "{query}"
            
            Context from memory:
            {memory_context}
            
            Create a list of tasks that need to be executed to answer this query effectively.
            Each task should be specific and actionable.
            """
            
            # For now, create a single main task
            # In a more sophisticated implementation, this would use the LLM to plan
            main_task = Task(
                id="main_task",
                description=f"Answer the user query: {query}",
                status=TaskStatus.PENDING,
            )
            
            return update_state(
                state,
                tasks=[main_task],
                next_action="execute",
            )
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return update_state(
                state,
                errors=state["errors"] + [f"Planning error: {str(e)}"],
                next_action="respond",
            )
    
    def execute_node(state: AgentState) -> AgentState:
        """Execute the current task using the agent."""
        logger.debug("Executing tasks")
        
        try:
            query = state["user_query"]
            messages = list(state["messages"])
            
            # Add user message if not already present
            if not any(isinstance(msg, HumanMessage) for msg in messages):
                messages.append(HumanMessage(content=query))
            
            # Get memory context
            memory_context = ""
            if memory_manager:
                memory_context = memory_manager.get_context_for_prompt(query)
            
            # Prepare agent input
            agent_input = {
                "input": query,
                "memory_context": memory_context,
                "chat_history": messages,
                "intermediate_steps": [],
            }
            
            # Execute the agent
            result = agent.invoke(agent_input)
            
            # Debug: log the result structure
            logger.info(f"Agent result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"Agent result keys: {list(result.keys())}")
                logger.info(f"Agent result values: {result}")
            else:
                logger.info(f"Agent result (non-dict): {result}")
            
            # Process the result
            if isinstance(result, list):
                # Agent returned a list of actions to execute
                logger.info("Agent returned actions to execute")
                tool_results = []
                
                for action in result:
                    try:
                        # Execute the tool
                        tool_name = action.tool
                        tool_input = action.tool_input
                        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                        
                        # Find the tool and execute it
                        tool = None
                        for t in tools:
                            if t.name == tool_name:
                                tool = t
                                break
                        
                        if tool:
                            # Execute the tool
                            tool_result = tool.invoke(tool_input)
                            logger.info(f"Tool {tool_name} result: {tool_result}")
                            
                            # Create ToolResult object
                            tool_result_obj = ToolResult(
                                tool_call_id=getattr(action, 'tool_call_id', f"call_{datetime.now().timestamp()}"),
                                tool_name=tool_name,
                                result=tool_result,
                                success=True,
                            )
                            tool_results.append(tool_result_obj)
                        else:
                            logger.error(f"Tool {tool_name} not found")
                            
                    except Exception as e:
                        logger.error(f"Error executing tool {action.tool}: {e}")
                        tool_result_obj = ToolResult(
                            tool_call_id=getattr(action, 'tool_call_id', f"call_{datetime.now().timestamp()}"),
                            tool_name=action.tool,
                            result=f"Error: {str(e)}",
                            success=False,
                            error_message=str(e)
                        )
                        tool_results.append(tool_result_obj)
                
                # Continue with next iteration to get the final response
                # Add tool results to intermediate_steps for the next call
                new_intermediate_steps = []
                for action, tool_result_obj in zip(result, tool_results):
                    new_intermediate_steps.append((action, tool_result_obj.result))
                
                # Call the agent again with the tool results
                updated_input = agent_input.copy()
                updated_input["intermediate_steps"] = new_intermediate_steps
                
                logger.info("Re-invoking agent with tool results")
                final_result = agent.invoke(updated_input)
                logger.info(f"Final agent result: {final_result}")
                
                # Extract the output from the final result
                output_text = None
                if isinstance(final_result, dict):
                    if "output" in final_result:
                        output_text = final_result["output"]
                    elif "return_values" in final_result and isinstance(final_result["return_values"], dict):
                        output_text = final_result["return_values"].get("output")
                elif hasattr(final_result, 'return_values') and hasattr(final_result.return_values, 'get'):
                    output_text = final_result.return_values.get("output")
                elif hasattr(final_result, 'output'):
                    output_text = final_result.output
                
                if output_text:
                    # Agent provided a final answer after tool execution
                    ai_message = AIMessage(content=output_text)
                    messages.append(ai_message)
                    
                    return update_state(
                        state,
                        messages=messages,
                        final_response=output_text,
                        tool_results=state["tool_results"] + tool_results,
                        next_action="respond",
                    )
                else:
                    return update_state(
                        state,
                        errors=state["errors"] + [f"Agent did not provide final response after tool execution: {final_result}"],
                        tool_results=state["tool_results"] + tool_results,
                        next_action="respond",
                    )
                
            elif isinstance(result, dict) and "output" in result:
                # Agent provided a final answer
                ai_message = AIMessage(content=result["output"])
                messages.append(ai_message)
                
                return update_state(
                    state,
                    messages=messages,
                    final_response=result["output"],
                    next_action="respond",
                )
            
            elif isinstance(result, AgentFinish):
                # Agent returned an AgentFinish object
                output_text = result.return_values.get("output")
                if output_text:
                    logger.info(f"Agent finished with output: {output_text[:100]}...")
                    ai_message = AIMessage(content=output_text)
                    messages.append(ai_message)
                    
                    return update_state(
                        state,
                        messages=messages,
                        final_response=output_text,
                        next_action="respond",
                    )
                else:
                    logger.warning(f"AgentFinish object has no output: {result.return_values}")
                    return update_state(
                        state,
                        errors=state["errors"] + [f"AgentFinish object missing output: {result.return_values}"],
                        next_action="respond",
                    )
                    
            elif hasattr(result, 'return_values') and hasattr(result.return_values, 'get'):
                # Agent returned some other object with return_values
                output_text = result.return_values.get("output")
                if output_text:
                    logger.info(f"Agent finished with output from return_values: {output_text[:100]}...")
                    ai_message = AIMessage(content=output_text)
                    messages.append(ai_message)
                    
                    return update_state(
                        state,
                        messages=messages,
                        final_response=output_text,
                        next_action="respond",
                    )
                else:
                    logger.warning(f"Return values object has no output: {result}")
                    return update_state(
                        state,
                        errors=state["errors"] + [f"Return values object missing output: {result}"],
                        next_action="respond",
                    )
            
            elif isinstance(result, dict) and "intermediate_steps" in result:
                # Agent used tools
                tool_results = []
                
                for action, observation in result["intermediate_steps"]:
                    tool_result = ToolResult(
                        tool_call_id=f"call_{datetime.now().timestamp()}",
                        tool_name=action.tool,
                        result=observation,
                        success=True,
                    )
                    tool_results.append(tool_result)
                
                # Add tool results to messages
                for tool_result in tool_results:
                    messages.append(AIMessage(content=f"Used tool {tool_result.tool_name}: {tool_result.result}"))
                
                return update_state(
                    state,
                    messages=messages,
                    tool_results=state["tool_results"] + tool_results,
                    next_action="reflect",
                )
            
            else:
                # Unexpected result format - provide detailed info
                result_info = f"Type: {type(result)}, Content: {result}"
                if isinstance(result, dict):
                    result_info += f", Keys: {list(result.keys())}"
                
                logger.warning(f"Unexpected agent result format: {result_info}")
                return update_state(
                    state,
                    errors=state["errors"] + [f"Unexpected agent result format: {result_info}"],
                    next_action="respond",
                )
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Execution failed: {e}")
            logger.error(f"Full error details: {error_details}")
            return update_state(
                state,
                errors=state["errors"] + [f"Execution error: {str(e)} - Details: {error_details}"],
                retry_count=state["retry_count"] + 1,
                next_action="reflect",
            )
    
    def reflect_node(state: AgentState) -> AgentState:
        """Reflect on the results and decide the next action."""
        logger.debug("Reflecting on results")
        
        try:
            # Check if we have errors and should retry
            if state["errors"] and state["retry_count"] < state["max_retries"]:
                logger.info(f"Retrying due to errors (attempt {state['retry_count'] + 1})")
                return update_state(
                    state,
                    next_action="execute",
                )
            
            # Check if we have a final response
            if state["final_response"]:
                return update_state(
                    state,
                    next_action="respond",
                )
            
            # Check if we have tool results but no final response
            if state["tool_results"] and not state["final_response"]:
                # Generate response based on tool results
                tool_summary = "\n".join([
                    f"- {tr.tool_name}: {tr.result}" 
                    for tr in state["tool_results"]
                ])
                
                reflection_prompt = f"""
                Based on the following tool results, provide a comprehensive answer to the user's query: "{state['user_query']}"
                
                Tool Results:
                {tool_summary}
                
                Please synthesize this information into a clear, helpful response.
                """
                
                try:
                    response = llm.invoke([HumanMessage(content=reflection_prompt)])
                    
                    return update_state(
                        state,
                        final_response=response.content,
                        next_action="respond",
                    )
                    
                except Exception as e:
                    logger.error(f"Response generation failed: {e}")
                    return update_state(
                        state,
                        final_response=f"I encountered an issue generating a response: {str(e)}",
                        next_action="respond",
                    )
            
            # Default: move to respond
            return update_state(
                state,
                next_action="respond",
            )
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return update_state(
                state,
                errors=state["errors"] + [f"Reflection error: {str(e)}"],
                next_action="respond",
            )
    
    def respond_node(state: AgentState) -> AgentState:
        """Generate the final response."""
        logger.debug("Generating final response")
        
        try:
            # If we don't have a final response, generate a fallback
            if not state["final_response"]:
                if state["errors"]:
                    error_summary = "; ".join(state["errors"])
                    final_response = f"I encountered some difficulties: {error_summary}. Please try rephrasing your question or provide more specific details."
                else:
                    final_response = "I wasn't able to provide a complete answer to your query. Please try rephrasing or provide more details."
                
                return update_state(
                    state,
                    final_response=final_response,
                    should_continue=False,
                    termination_reason="fallback_response",
                )
            
            # We have a final response
            return update_state(
                state,
                should_continue=False,
                termination_reason="task_completed",
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return update_state(
                state,
                final_response=f"I encountered an error while generating a response: {str(e)}",
                should_continue=False,
                termination_reason="error",
            )
    
    def should_continue_decision(state: AgentState) -> str:
        """Decide whether to continue or end the workflow."""
        if should_terminate(state):
            return END
        
        # Route based on next action
        next_action = state.get("next_action", "end")
        
        if next_action in ["plan", "execute", "reflect", "respond"]:
            return next_action
        
        return END
    
    # Add nodes to the workflow
    workflow.add_node("start", start_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("respond", respond_node)
    
    # Add edges
    workflow.set_entry_point("start")
    workflow.add_edge("start", "plan")
    workflow.add_conditional_edges(
        "plan",
        should_continue_decision,
        {
            "execute": "execute",
            "respond": "respond",
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "execute",
        should_continue_decision,
        {
            "reflect": "reflect",
            "respond": "respond",
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "reflect",
        should_continue_decision,
        {
            "execute": "execute",
            "respond": "respond",
            END: END,
        }
    )
    workflow.add_edge("respond", END)
    
    # Add custom workflows if provided
    if custom_workflows:
        for name, workflow_func in custom_workflows.items():
            workflow.add_node(name, workflow_func)
    
    return workflow