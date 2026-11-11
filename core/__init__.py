"""Публичный интерфейс генерации коротких видео по песням."""

from .models import PipelineConfig, PipelineResult
from .pipeline import run_pipeline

__all__ = ["run_pipeline", "PipelineConfig", "PipelineResult"]
