from controllers.basic_controller import BasicMAC
from controllers.non_shared_controller import NonSharedMAC
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs

def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


# This multi-agent controller shares parameters between agents
class MADDPGMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        # super().__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        # Prevent initialization of action_selector from BasicMAC
        self.action_selector = None

        self._hidden_states_flatten = None
        self._batch_size = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        # Only select actions for the selected batch elements in bs
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def target_actions(self, ep_batch, t_ep):
        agent_outputs = self.forward(ep_batch, t_ep)
        return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self._hidden_states_flatten = self.agent(agent_inputs, self._hidden_states_flatten)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions==0] = -1e10
        return agent_outs

    def init_hidden_one_agent(self, batch_size):
        self._batch_size = batch_size
        expanded = self.expand_hidden_states(self.agent.init_hidden(), batch_size, n_agents=1)
        self._hidden_states_flatten = self._flatten_hidden(expanded)


class NonSharedMADDPGMAC(NonSharedMAC):
    """MADDPG controller for non-shared-parameter agents."""

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def target_actions(self, ep_batch, t_ep):
        agent_outputs = self.forward(ep_batch, t_ep)
        return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self._hidden_states = self.agent(agent_inputs, self._hidden_states)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs
