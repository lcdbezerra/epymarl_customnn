from functools import partial

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .custom_agent import CustomAgent
from .ns_agent import NSAgent

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = partial(NSAgent, agent=RNNAgent)
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["custom"] = CustomAgent
REGISTRY["custom_ns"] = partial(NSAgent, agent=CustomAgent)