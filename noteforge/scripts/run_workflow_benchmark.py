"""Compatibility alias for the published inference runtime."""
from noteforge.scripts.runtime import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith('_')]
