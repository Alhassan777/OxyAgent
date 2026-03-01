"""Multi-agent lifeguard system."""

from .detection_agent import DetectionAgent
from .path_agent import PathAgent
from .decision_agent import DecisionAgent
from .ems_agent import EMSAgent

__all__ = ["DetectionAgent", "PathAgent", "DecisionAgent", "EMSAgent"]
