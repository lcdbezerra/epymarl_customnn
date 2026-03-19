import torch.nn as nn
import torch as th

from modules.agents.custom_agent import CustomAgent

class NSAgent(nn.Module):
    """
    Non-shared parameter agent wrapper.
    Creates a separate instance of the base agent for each agent in the environment.
    """
    def __init__(self, input_shape, args, agent=CustomAgent):
        super(NSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList(
            [agent(input_shape, args) for _ in range(self.n_agents)]
        )

    def init_hidden(self):
        """Initialise hidden state for all (non-shared) agents.

        Returns list[layer][state_part] with tensors of shape (n_agents, hidden_dim_layer).
        """
        all_hiddens = [agent.init_hidden() for agent in self.agents]
        base_hidden = all_hiddens[0]
        return [
            tuple(
                th.cat([ah[layer_idx][idx] for ah in all_hiddens], dim=0)
                for idx in range(len(base_hidden[layer_idx]))
            )
            for layer_idx in range(len(base_hidden))
        ]

    def forward(self, inputs, hidden_state):
        """Forward pass for non-shared agents.

        hidden_state: list[layer][state_part] with tensors of shape
                      (batch, n_agents, hidden_dim_layer)
        """
        inputs = inputs.view(-1, self.n_agents, self.input_shape)

        hiddens = []
        qs = []
        for i in range(self.n_agents):
            h_in = [
                tuple(part[:, i] for part in layer_tuple)
                for layer_tuple in hidden_state
            ]
            q, h = self.agents[i](inputs[:, i], h_in)
            qs.append(q)
            hiddens.append(h)

        q_out = th.cat([q.unsqueeze(1) for q in qs], dim=1).view(-1, qs[0].size(-1))
        h_out = [
            tuple(
                th.stack([h[layer_idx][part_idx] for h in hiddens], dim=1)
                for part_idx in range(len(hiddens[0][layer_idx]))
            )
            for layer_idx in range(len(hiddens[0]))
        ]

        return q_out, h_out

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
