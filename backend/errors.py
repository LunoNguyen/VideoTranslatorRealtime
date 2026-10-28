"""Errors shared across pipeline stages."""
from __future__ import annotations


class ModelMissingError(RuntimeError):
    """Model is not on disk and could not be downloaded."""
