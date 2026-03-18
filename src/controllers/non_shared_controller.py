from controllers.basic_controller import BasicMAC
import torch as th


class NonSharedMAC(BasicMAC):
    def expand_hidden_states(self, hidden_states, batch_size, n_agents=None):
        if isinstance(hidden_states, list):
            hidden_states = [
                tuple([x.expand(batch_size, -1, -1) for x in h])
                for h in hidden_states
            ]
        elif isinstance(hidden_states, th.Tensor):
            hidden_states = hidden_states.expand(batch_size, -1, -1)
        else:
            raise ValueError(f"Unexpected hidden states type: {type(hidden_states)}")
        return hidden_states
