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
        single_episode = self._is_single_episode(inputs)
        inputs_reshaped = self._prepare_inputs(inputs, single_episode)

        hiddens = []
        qs = []

        # Per-agent forward
        for i in range(self.n_agents):
            agent_input = self._get_agent_input(inputs_reshaped, i, single_episode)
            h_in = self._get_hidden_for_agent(i, hidden_state)
            q, h = self.agents[i](agent_input, h_in)
            qs.append(q)
            hiddens.append(h)

        q_out = self._aggregate_q_values(qs, single_episode)
        h_out = self._aggregate_hidden_states(hiddens, single_episode)

        return q_out, h_out

    def _is_single_episode(self, inputs):
        """
        Determine if inputs correspond to a single episode batch.
        Returns True if inputs.size(0) == n_agents, False otherwise.
        """
        return inputs.size(0) == self.n_agents

    def _prepare_inputs(self, inputs, single_episode):
        """
        Prepare inputs for per-agent processing.
        For single episode: returns inputs as-is (shape: n_agents, input_dim).
        For batched: reshapes to (batch, n_agents, input_dim).
        """
        if single_episode:
            # inputs: (n_agents, input_dim)
            return inputs
        else:
            # inputs: (batch * n_agents, input_dim) -> (batch, n_agents, input_dim)
            return inputs.view(-1, self.n_agents, self.input_shape)

    def _get_agent_input(self, inputs, agent_idx, single_episode):
        """
        Extract input for a specific agent.
        For single episode: returns (1, input_dim) by unsqueezing agent dimension.
        For batched: returns (batch, input_dim) by indexing agent dimension.
        """
        if single_episode:
            # inputs: (n_agents, input_dim)
            return inputs[agent_idx].unsqueeze(0)
        else:
            # inputs: (batch, n_agents, input_dim)
            return inputs[:, agent_idx]

    def _get_hidden_for_agent(self, agent_idx, hidden_state):
        """
        Extract hidden state for a specific agent.
        Supports both CustomAgent-style (list of tuples) and RNNAgent-style (tensor) hidden states.
        """
        if isinstance(hidden_state, list):
            # CustomAgent-style hidden state: list[layer][state_part] with shape (batch, n_agents, hidden_dim_layer)
            return self._get_custom_hidden_for_agent(agent_idx, hidden_state)
        else:
            # RNNAgent-style tensor hidden state: (batch, n_agents, hidden_dim)
            return hidden_state[:, agent_idx]

    def _get_custom_hidden_for_agent(self, agent_idx, hidden_state):
        """
        Extract CustomAgent-style hidden state for a specific agent.
        hidden_state: list[layer_idx] of tuples of tensors with shape (batch, n_agents, hidden_dim_layer)
        Returns: list[layer_idx] of tuples of tensors with shape (batch, hidden_dim_layer)
        """
        per_agent_hidden = []
        for layer_tuple in hidden_state:
            layer_state_parts = tuple(part[:, agent_idx] for part in layer_tuple)
            per_agent_hidden.append(layer_state_parts)
        return per_agent_hidden

    def _aggregate_q_values(self, qs, single_episode):
        """
        Aggregate Q-values from all agents into a single tensor.
        For single episode: returns (n_agents, n_actions).
        For batched: returns (batch * n_agents, n_actions).
        """
        if single_episode:
            # qs: list of (1, n_actions) -> (n_agents, n_actions)
            return th.cat(qs, dim=0)
        else:
            # qs: list of (batch, n_actions) -> (batch, n_agents, n_actions) -> (batch*n_agents, n_actions)
            return th.cat([q.unsqueeze(1) for q in qs], dim=1).view(-1, qs[0].size(-1))

    def _aggregate_hidden_states(self, hiddens, single_episode):
        """
        Aggregate hidden states from all agents.
        Supports tensor (RNNAgent-style) and list-of-tuples (CustomAgent-style) hidden states.
        """
        if isinstance(hiddens[0], th.Tensor):
            # RNNAgent-style tensor hidden state
            return self._aggregate_tensor_hidden_states(hiddens, single_episode)
        else:
            # CustomAgent-style list-of-tuples hidden state
            return self._aggregate_custom_hidden_states(hiddens)

    def _aggregate_tensor_hidden_states(self, hiddens, single_episode):
        """
        Aggregate tensor-based hidden states (RNNAgent-style).
        For single episode: returns (1, n_agents, hidden_dim).
        For batched: returns (batch, n_agents, hidden_dim).
        """
        if single_episode:
            # hiddens: list of (1, hidden_dim) -> (1, n_agents, hidden_dim)
            return th.cat(hiddens, dim=0).unsqueeze(0)
        else:
            # hiddens: list of (batch, hidden_dim) -> (batch, n_agents, hidden_dim)
            return th.cat([h.unsqueeze(1) for h in hiddens], dim=1)

    def _aggregate_custom_hidden_states(self, hiddens):
        """
        Aggregate CustomAgent-style hidden states (list of tuples).
        Returns: list[layer_idx] of tuples of tensors with shape (batch, n_agents, hidden_dim_layer)
        """
        combined_hidden = [
            tuple([
                th.stack([h[layer_idx][part_idx] for h in hiddens], dim=1)
                for part_idx in range(len(hiddens[0][layer_idx]))
            ])
            for layer_idx in range(len(hiddens[0]))
        ]
        return combined_hidden

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
