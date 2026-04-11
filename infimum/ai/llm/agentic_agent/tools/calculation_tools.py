"""
Calculation and mathematical tools using decorator approach.
"""

import math
import ast
import operator
from typing import Any, Dict, Union

from langchain_core.tools import tool


# Safe operators and functions for evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'sqrt': math.sqrt,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'pi': math.pi,
    'e': math.e,
}


def _safe_eval(expression: str) -> Union[int, float]:
    """Safely evaluate mathematical expressions."""
    try:
        # Parse the expression into AST
        node = ast.parse(expression, mode='eval')
        return _eval_node(node.body)
    except Exception as e:
        raise Exception(f"Invalid expression: {str(e)}")


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate AST nodes."""
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise Exception(f"Unsupported operator: {type(node.op).__name__}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise Exception(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(operand)
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in SAFE_FUNCTIONS:
            raise Exception(f"Unsupported function: {func_name}")
        args = [_eval_node(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        else:
            raise Exception(f"Unsupported name: {node.id}")
    else:
        raise Exception(f"Unsupported node type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations and evaluate expressions safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "sqrt(16)")
    
    Returns:
        str: The calculated result or error message
    """
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Parse and evaluate safely
        result = _safe_eval(expression)
        
        return f"Result: {result} (type: {type(result).__name__})"
        
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def compound_interest(
    principal: float, 
    rate: float, 
    time: float, 
    compounds_per_year: int = 12
) -> str:
    """Calculate compound interest for investments.
    
    Args:
        principal: Initial principal balance (must be positive)
        rate: Annual interest rate as percentage (e.g., 5.0 for 5%)
        time: Time period in years (must be positive)
        compounds_per_year: Compounding frequency per year (default: 12 for monthly)
    
    Returns:
        str: Formatted compound interest calculation results
    """
    try:
        if principal <= 0:
            return "Error: Principal must be positive"
        if rate <= 0 or rate > 100:
            return "Error: Rate must be between 0 and 100"
        if time <= 0:
            return "Error: Time must be positive"
        if compounds_per_year <= 0:
            return "Error: Compounds per year must be positive"
        
        annual_rate = rate / 100  # Convert percentage to decimal
        
        # Calculate final amount: A = P(1 + r/n)^(nt)
        final_amount = principal * (
            1 + (annual_rate / compounds_per_year)
        ) ** (compounds_per_year * time)
        
        interest_earned = final_amount - principal
        
        # Calculate effective annual rate
        effective_rate = ((1 + annual_rate / compounds_per_year) ** compounds_per_year - 1) * 100
        
        return f"""Compound Interest Calculation:
- Principal: ${round(principal, 2):,.2f}
- Annual Rate: {rate}%
- Time: {time} years
- Compounding: {compounds_per_year} times per year
- Final Amount: ${round(final_amount, 2):,.2f}
- Interest Earned: ${round(interest_earned, 2):,.2f}
- Effective Annual Rate: {round(effective_rate, 4)}%
- Total Return: {round((interest_earned / principal) * 100, 2)}%"""
        
    except Exception as e:
        return f"Compound interest calculation error: {str(e)}"


# Legacy classes for backward compatibility (using tool instances)
class CalculatorTool:
    def __init__(self):
        self.tool = calculator
        self.name = calculator.name
        self.description = calculator.description

class CompoundInterestTool:
    def __init__(self):
        self.tool = compound_interest
        self.name = compound_interest.name  
        self.description = compound_interest.description