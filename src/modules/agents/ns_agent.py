import torch.nn as nn
import torch as th

from modules.agents.custom_agent import CustomAgent

class NSAgent(nn.Module):
    """
    General non-shared parameter agent wrapper.
    Creates a separate instance of the base agent for each agent in the environment.
    The base agent class is specified via args.agent or args.base_agent.
    """
    def __init__(self, input_shape, args, agent=CustomAgent):
        super(NSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agent_class = agent
        self.agents = th.nn.ModuleList(
            [self.agent_class(input_shape, args) for _ in range(self.n_agents)]
        )

    def init_hidden(self):
        """
        Initialise hidden state for all (non-shared) agents.

        Must work for:
        - `RNNAgent.init_hidden` which returns a Tensor of shape (1, hidden_dim)
        - `CustomAgent.init_hidden` which returns a list of tuples of Tensors
          (one entry per layer, each Tensor shaped (1, hidden_dim_layer))

        For the non-shared MAC we want a per‑environment hidden state that
        already has an agent dimension:
        - RNNAgent case:      (n_agents, hidden_dim)
        - CustomAgent case:   list[layer][state_part] -> (n_agents, hidden_dim_layer)
        """
        if self.agent_class == CustomAgent:
            return self._init_hidden_custom_agent()
        else:
            return th.cat([a.init_hidden() for a in self.agents], dim=0)

    def _init_hidden_custom_agent(self):
        """
        Initialize hidden state for CustomAgent-style agents.
        Returns a list of tuples of tensors with agent dimension added.
        """
        all_hiddens = [agent.init_hidden() for agent in self.agents]
        n_layers = len(all_hiddens[0])

        combined_hidden = []
        base_hidden = all_hiddens[0]
        for layer_idx in range(n_layers):
            # Each layer entry is a tuple, e.g. (h,) for GRUCell or (h, c) for LSTMCell
            layer_state_parts = []
            for idx in range(len(base_hidden[layer_idx])):
                # Collect this part across agents and concatenate along agent dimension
                part_tensors = [
                    agent_hidden[layer_idx][idx]  # shape: (1, hidden_dim_layer)
                    for agent_hidden in all_hiddens
                ]
                layer_state_parts.append(th.cat(part_tensors, dim=0))  # (n_agents, hidden_dim_layer)

            combined_hidden.append(tuple(layer_state_parts))
        return combined_hidden

    def forward(self, inputs, hidden_state):
        """
        Forward pass for non-shared agents.

        Supports:
        - RNNAgent: tensor hidden state of shape (batch, n_agents, hidden_dim)
        - CustomAgent: list[layer][state_part] with tensors of shape
                       (batch, n_agents, hidden_dim_layer)
        """
        inputs = inputs.view(-1, self.n_agents, self.input_shape)

        hiddens = []
        qs = []
        for i in range(self.n_agents):
            h_in = self._get_hidden_for_agent(i, hidden_state)
            q, h = self.agents[i](inputs[:, i], h_in)
            qs.append(q)
            hiddens.append(h)

        q_out = th.cat([q.unsqueeze(1) for q in qs], dim=1).view(-1, qs[0].size(-1))
        h_out = self._aggregate_hidden_states(hiddens)

        return q_out, h_out

    def _get_hidden_for_agent(self, agent_idx, hidden_state):
        """
        Extract hidden state for a specific agent.
        Supports both CustomAgent-style (list of tuples) and RNNAgent-style (tensor) hidden states.
        """
        if isinstance(hidden_state, list):
            return [
                tuple(part[:, agent_idx] for part in layer_tuple)
                for layer_tuple in hidden_state
            ]
        else:
            return hidden_state[:, agent_idx]

    def _aggregate_hidden_states(self, hiddens):
        """
        Aggregate hidden states from all agents.
        Supports tensor (RNNAgent-style) and list-of-tuples (CustomAgent-style) hidden states.
        """
        if isinstance(hiddens[0], th.Tensor):
            return th.cat([h.unsqueeze(1) for h in hiddens], dim=1)
        else:
            return self._aggregate_custom_hidden_states(hiddens)

    def _aggregate_custom_hidden_states(self, hiddens):
        """
        Aggregate CustomAgent-style hidden states (list of tuples).
        Returns: list[layer_idx] of tuples of tensors with shape (batch, n_agents, hidden_dim_layer)
        """
        return [
            tuple(
                th.stack([h[layer_idx][part_idx] for h in hiddens], dim=1)
                for part_idx in range(len(hiddens[0][layer_idx]))
            )
            for layer_idx in range(len(hiddens[0]))
        ]

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
