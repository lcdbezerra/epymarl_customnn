from controllers.basic_controller import BasicMAC
import torch as th


class NonSharedMAC(BasicMAC):
    def _flatten_hidden(self, hidden_states):
        return hidden_states

    def _unflatten_hidden(self, hidden_states):
        return hidden_states
