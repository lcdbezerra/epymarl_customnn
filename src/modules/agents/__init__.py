from functools import partial

from .custom_agent import CustomAgent
from .ns_agent import NSAgent

REGISTRY = {}
REGISTRY["custom"] = CustomAgent
REGISTRY["custom_ns"] = partial(NSAgent, agent=CustomAgent)