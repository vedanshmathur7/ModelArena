"""
Tool use utilities for ModelArena.

Provides two tools:
  1. get_current_time: Returns the current date and time.
  2. calculate: Safely evaluates simple mathematical expressions.

Exposes a unified handler for local and hosted model tool-calling integration.
"""

from __future__ import annotations

import ast
import datetime
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core Tool Actions
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Get the current date and time in a human-readable format."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y %I:%M:%S %p")


def calculate(expression: str) -> str:
    """
    Safely evaluate a basic mathematical expression.
    Allowed characters: digits, operators (+, -, *, /, (, )), and spaces.
    """
    # Clean expression
    expression = expression.strip()
    
    # Filter allowed characters to prevent code injection
    if not re.match(r"^[0-9+\-*/().\s]*$", expression):
        return "Error: Invalid characters in expression."
    
    # Ensure there are numbers
    if not re.search(r"\d+", expression):
        return "Error: Expression must contain numbers."

    try:
        # Parse expression to AST to ensure safety
        node = ast.parse(expression, mode="eval")
        
        # Verify AST contains only safe operations
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,  # Python 3.8+
            ast.Num,       # Deprecated but kept for older Python versions
            ast.operator,
            ast.unaryop
        )
        
        for subnode in ast.walk(node):
            if not isinstance(subnode, allowed_nodes):
                return f"Error: Operations of type '{type(subnode).__name__}' are not supported."
        
        # Compile and evaluate in an isolated environment
        code = compile(node, "<string>", "eval")
        result = eval(code, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool Schemas for API / Frontier Models
# ---------------------------------------------------------------------------

def get_tools_definition() -> List[Dict[str, Any]]:
    """Return list of tool definitions in OpenAI/Groq function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current day, date, and local time. Use this when the user asks about the time, date, today, or current day.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations. Supports operators: +, -, *, /, (, ). Use this to get precise results for math equations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate, e.g., '128 * 4.5' or '(34 + 19) / 2'.",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "brave_search",
                "description": "Search the web for current or historical information. Use this only when the query requires searching for external or up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current or historical information. Use this only when the query requires searching for external or up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool by name with arguments and return output string."""
    logger.info("Executing tool: %s with args: %s", name, arguments)
    if name == "get_current_time":
        return get_current_time()
    elif name == "calculate":
        expression = arguments.get("expression", "")
        return calculate(expression)
    elif name in ("brave_search", "web_search"):
        query = arguments.get("query", "")
        logger.info("Mock search executed for query: %s", query)
        return f"Search results for '{query}': No search connection is active. Please answer this query directly using your existing knowledge, or explain that you do not have search capability."
    else:
        return f"Error: Unknown tool '{name}'."


# ---------------------------------------------------------------------------
# Semantic Prompt Injection Middleware (OSS Model Interceptor)
# ---------------------------------------------------------------------------

def check_and_inject_tools(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Middleware that intercepts user queries asking for time or math.
    Executes the tool immediately and prepends the output to the query.
    Allows smaller local models running on CPU to use tools transparently.
    """
    if not messages:
        return messages

    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        return messages

    content = last_msg.get("content", "")
    content_lower = content.lower()

    # 1. Match time/date queries
    time_regexes = [
        r"\bwhat\s+time\s+is\s+it\b",
        r"\bwhat\s+is\s+the\s+time\b",
        r"\bcurrent\s+time\b",
        r"\bcurrent\s+date\b",
        r"\bwhat\s+date\s+is\s+it\b",
        r"\bwhat\s+is\s+today'?s\s+date\b",
        r"\bwhat\s+is\s+the\s+date\b",
        r"\bwhat\s+day\s+is\s+it\s+today\b",
        r"\btoday'?s\s+day\b",
    ]
    is_time_req = any(re.search(pat, content_lower) for pat in time_regexes)

    # 2. Match math queries
    math_regexes = [
        r"\bcalculate\s+([0-9+\-*/().\s]+)",
        r"\bwhat\s+is\s+([0-9+\-*/().\s]+)\?",
        r"\bwhat\s+is\s+([0-9+\-*/().\s]+)$",
        r"\bcompute\s+([0-9+\-*/().\s]+)",
    ]

    math_expr = None
    for pat in math_regexes:
        match = re.search(pat, content_lower)
        if match:
            expr = match.group(1).strip()
            # Remove trailing question marks or punctuation
            expr = expr.rstrip("?").strip()
            # Verify it has digits and at least one arithmetic sign
            if re.search(r"\d+", expr) and any(op in expr for op in ["+", "-", "*", "/"]):
                math_expr = expr
                break

    # If it is a raw math expression without keywords (e.g., "128 * 4")
    if not math_expr:
        if (
            re.match(r"^[0-9+\-*/().\s]+$", content_lower)
            and any(op in content_lower for op in ["+", "-", "*", "/"])
            and re.search(r"\d+", content_lower)
        ):
            math_expr = content.strip()

    if is_time_req:
        time_result = get_current_time()
        injected = f"[System Tool Output - get_current_time(): {time_result}]\n\n{content}"
        new_messages = list(messages)
        new_messages[-1] = {"role": "user", "content": injected}
        logger.info("Injected time tool output into OSS context: %s", time_result)
        return new_messages

    elif math_expr:
        math_result = calculate(math_expr)
        injected = f"[System Tool Output - calculate('{math_expr}'): {math_result}]\n\n{content}"
        new_messages = list(messages)
        new_messages[-1] = {"role": "user", "content": injected}
        logger.info("Injected math tool output into OSS context: %s -> %s", math_expr, math_result)
        return new_messages

    return messages
