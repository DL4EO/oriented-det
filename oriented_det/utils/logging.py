"""Logging and tracing utilities for oriented-det framework.

This module provides configurable logging for debugging and performance analysis.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional, Callable, Any
import functools


class TraceLogger:
    """Configurable logger for tracing function execution."""
    
    _enabled: bool = False
    _indent_level: int = 0
    _indent_str: str = "  "
    
    @classmethod
    def enable(cls) -> None:
        """Enable tracing/logging."""
        cls._enabled = True
    
    @classmethod
    def disable(cls) -> None:
        """Disable tracing/logging."""
        cls._enabled = False
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracing is enabled."""
        return cls._enabled
    
    @classmethod
    def trace(cls, message: str, *args, **kwargs) -> None:
        """Print a trace message if logging is enabled."""
        if cls._enabled:
            indent = cls._indent_str * cls._indent_level
            formatted_msg = message.format(*args, **kwargs) if args or kwargs else message
            print(f"{indent}{formatted_msg}")
    
    @classmethod
    @contextmanager
    def trace_block(cls, message: str, *args, **kwargs):
        """Context manager for tracing a code block with timing."""
        if cls._enabled:
            formatted_msg = message.format(*args, **kwargs) if args or kwargs else message
            cls.trace(f"→ {formatted_msg}")
            cls._indent_level += 1
            start_time = time.time()
            try:
                yield
            finally:
                elapsed = time.time() - start_time
                cls._indent_level -= 1
                cls.trace(f"✅ {formatted_msg} ({elapsed:.3f}s)")
        else:
            yield
    
    @classmethod
    def trace_timing(cls, message: str, elapsed: float, *args, **kwargs) -> None:
        """Trace a message with timing information."""
        if cls._enabled:
            formatted_msg = message.format(*args, **kwargs) if args or kwargs else message
            cls.trace(f"{formatted_msg} ({elapsed:.3f}s)")
    
    @classmethod
    def trace_value(cls, name: str, value: Any, unit: Optional[str] = None) -> None:
        """Trace a value with optional unit."""
        if cls._enabled:
            if unit:
                cls.trace(f"{name}: {value} {unit}")
            else:
                cls.trace(f"{name}: {value}")


# Global instance
logger = TraceLogger()


def trace_function(func_name: Optional[str] = None):
    """Decorator to trace function execution.
    
    Args:
        func_name: Optional custom name for the function in logs.
                   If None, uses the actual function name.
    
    Example:
        @trace_function()
        def my_function(x, y):
            return x + y
        
        @trace_function("custom_name")
        def another_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger.is_enabled():
                # Format arguments for display
                args_str = ", ".join([str(arg)[:50] for arg in args[:3]])
                if len(args) > 3:
                    args_str += f", ... (+{len(args)-3} more)"
                kwargs_str = ", ".join([f"{k}={str(v)[:30]}" for k, v in list(kwargs.items())[:2]])
                if len(kwargs) > 2:
                    kwargs_str += f", ... (+{len(kwargs)-2} more)"
                
                call_str = f"{name}({args_str}"
                if kwargs_str:
                    call_str += f", {kwargs_str}"
                call_str += ")"
                
                logger.trace(f"→ {call_str}")
                logger._indent_level += 1
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    logger._indent_level -= 1
                    logger.trace(f"✅ {name} returned ({elapsed:.3f}s)")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger._indent_level -= 1
                    logger.trace(f"❌ {name} raised {type(e).__name__}: {e} ({elapsed:.3f}s)")
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenience functions
def enable_tracing() -> None:
    """Enable tracing globally."""
    logger.enable()


def disable_tracing() -> None:
    """Disable tracing globally."""
    logger.disable()


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return logger.is_enabled()


__all__ = [
    "TraceLogger",
    "logger",
    "trace_function",
    "enable_tracing",
    "disable_tracing",
    "is_tracing_enabled",
]

