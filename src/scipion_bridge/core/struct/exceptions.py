"""Exceptions for the struct module."""

from __future__ import annotations


class UninitializedFieldError(AttributeError, ValueError):
    """Raised when attempting to access a field or element that has not been initialized or is null.

    Inherits from both AttributeError and ValueError so callers expecting either standard Python
    attribute lookup failure or Arrow null-value errors catch this consistently.
    """

    pass

