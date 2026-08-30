"""Specialist output contracts.

Today a specialist returns a plain string (``BaseAgent.analyze -> str``) and the
Executive receives prose with no way to tell a sourced fact from a model's own
arithmetic. This package holds the typed contract that replaces that boundary.

Nothing here is wired into the production path yet — see
``result_contract`` for the schema, parser, and compatibility renderer.
"""
