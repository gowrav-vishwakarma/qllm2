"""V6: Phase-First Architecture with Multi-Timescale SSM, Working Memory, and External Memory Layers."""
from .model import PhaseFieldLM, create_model
from .backbone import PhaseFieldBackbone
from .config import V6Config, get_config
