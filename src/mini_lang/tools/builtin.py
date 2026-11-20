"""Built-in tools: a safe arithmetic calculator and a mock search."""

from __future__ import annotations

import ast
import operator

from .base import FnTool, ToolRegistry

# --- safe arithmetic evaluator (no eval()) --------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


async def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression safely (no names, no calls)."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


# A tiny canned knowledge base for the mock search tool.
_FACTS = {
    "refund": "Refunds are processed within 5 business days to the original payment method.",
    "shipping": "Standard shipping takes 3-5 business days; express takes 1-2.",
    "default": "No specific record found; please consult the documentation.",
}


async def mock_search(query: str) -> str:
    q = query.lower()
    for key, value in _FACTS.items():
        if key in q:
            return value
    return _FACTS["default"]


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        FnTool("calculator", "Evaluate an arithmetic expression.", calculator)
    )
    registry.register(
        FnTool("search", "Look up a fact by keyword.", mock_search)
    )
    return registry
