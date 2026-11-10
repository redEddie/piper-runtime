"""Inference only: imports neither JAX, MuJoCo, CUDA nor a robot SDK."""
from .policy import Policy

__all__ = ["Policy"]
