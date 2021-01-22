import torch.nn as nn
import math
import functools
import torch as ch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .torch_utils import *

'''
Neural network models for estimating value and policy functions
Contains:
- Initialization utilities
- Value Network(s)
- Policy Network(s)
- Retrieval Function
'''

########################
### INITIALIZATION UTILITY FUNCTIONS:
# initialize_weights
########################

HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Tanh
STD = 2**0.5

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")
            

########################
### INITIALIZATION UTILITY FUNCTIONS:
# Generic Value network, Value network MLP
########################

class ValueDenseNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init=None, hidden_sizes=(64, 64), activation=None):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = ACTIVATION()
        self.affine_layers = nn.ModuleList()

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            if init is not None:
                initialize_weights(l, init)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        if init is not None:
            initialize_weights(self.final, init, scale=1.0)

    def initialize(self, init="orthogonal"):
        for l in self.affine_layers:
            initialize_weights(l, init)
        initialize_weights(self.final, init, scale=1.0)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        value = self.final(x)
        return value

    def get_value(self, x):
        return self(x)

    def reset(self):
        return

    # MLP does not maintain history.
    def pause_history(self):
        return

    def continue_history(self):
        return


def pack_history(features, not_dones):
    # Features has dimension (N, state_dim), where N contains a few episodes
    # not_dones splits these episodes (0 in not_dones is end of an episode)
    nnz = ch.nonzero(1.0 - not_dones, as_tuple=False).view(-1).cpu().numpy()
    # nnz has the position where not_dones = 0 (end of episode)
    all_pieces = []
    lengths = []
    start = 0
    for i in nnz:
        end = i + 1
        all_pieces.append(features[start:end, :])
        lengths.append(end - start)
        start = end
    # The last episode is missing, unless the previous episode end at the last element.
    if end != features.size(0):
        all_pieces.append(features[end:, :])
        lengths.append(features.size(0) - end)
    # print(lengths)
    padded = pad_sequence(all_pieces, batch_first=True)
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    return packed

def unpack_history(padded_pieces, lengths):
    # padded pieces in shape (batch, time, hidden)
    # lengths in shape (batch,)
    all_pieces = []
    for i, l in enumerate(lengths.cpu().numpy()):
        # For each batch element in padded_pieces, get the first l elements.
        all_pieces.append(padded_pieces[i, 0:l, :])
    # return shape (N, hidden)
    return ch.cat(all_pieces, dim=0)


class ValueLSTMNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, init=None, hidden_sizes=(64, 64), activation=None):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        print('Using LSTM for value function!!')
        super().__init__()
        self.hidden_sizes = hidden_sizes

        self.embedding_layer = nn.Linear(state_dim, self.hidden_sizes[0])
        initialize_weights(self.embedding_layer, init, scale=0.01)

        self.lstm = nn.LSTM(input_size=self.hidden_sizes[0], hidden_size=self.hidden_sizes[1], num_layers=1, batch_first=True)

        self.final = nn.Linear(self.hidden_sizes[-1], 1)
        if init is not None:
            initialize_weights(self.final, init, scale=1.0)

        # LSTM hidden states. Only used in inference mode when a batch size of 1 is used.
        self.hidden = [ch.zeros(1, 1, self.hidden_sizes[1]),
            ch.zeros(1, 1, self.hidden_sizes[1])]
        self.paused = False


    def initialize(self, init="orthogonal"):
        for l in self.affine_layers:
            initialize_weights(l, init)
        initialize_weights(self.final, init, scale=1.0)


    def forward(self, states, not_dones=None):
        if not_dones is not None:  # we get a full batch of states, we split them into episodes based on not_dones
            assert states.size(0) == 1 and states.size(1) != 1 and states.ndim == 3  # input dimension must be in shape (1, N, state_dim)
            # New shape: (N, state_dim)
            states = states.squeeze(0)
            features = self.embedding_layer(states)
            # New shape: (N, )
            not_dones = not_dones.squeeze(0)
            # Pack states into episodes according to not_dones
            packed_features = pack_history(features, not_dones)
            # Run LSTM
            outputs, _ = self.lstm(packed_features)
            # pad output results
            padded, lengths = pad_packed_sequence(outputs, batch_first=True)
            # concate output to a single array (N, hidden_dim)
            hidden = unpack_history(padded, lengths)
            """
            hidden = F.relu(features)
            """
            # final output, apply linear transformation on hidden output.
            value = self.final(hidden)
            """
            print(states.size(), not_dones.size())
            print(padded.size())
            print(hidden.size())
            print(lengths)
            print(value.size())
            import traceback; traceback.print_stack()
            input()
            """
            # add back the extra dimension. Shape (1, N, 1)
            return value.unsqueeze(0)
        elif states.ndim == 2 and states.size(0) == 1:
            # We get a state with batch shape 1. This is only possible in inferece and attack mode.
            # embedding has shape (1, 1, hidden_dim)
            embedding = self.embedding_layer(states).unsqueeze(1)
            # Use saved hidden states
            _, hidden = self.lstm(embedding, self.hidden)
            # hidden dimension: (1, 1, hidden_size)
            output = self.final(hidden[0])
            # save hidden state.
            if not self.paused:
                self.hidden[0] = hidden[0]
                self.hidden[1] = hidden[1]

            # squeeze the time dimension, return shape (1, action_dim)
            value = self.final(hidden[0]).squeeze(1)
            return value
        else:
            raise NotImplementedError
            # state: (N, time, state_dim)
            embeddings = self.embedding_layer(states)
            # Run LSTM, output (N, time, hidden_dim)
            # outputs = F.relu(embeddings)
            outputs, _ = self.lstm(embeddings)
            # final output (N, time, 1)
            value = self.final(outputs)
            # add back the extra dimension. Shape (1, N, 1)
            return value

    def multi_forward(self, x, hidden):
        embeddings = self.embedding_layer(x)
        # print('embeddings', embeddings.size())
        # Run LSTM with packed sequence
        outputs, hidden = self.lstm(embeddings, hidden)
        # desired outputs dimension: (batch, time_step, hidden_size)
        # print('outputs', outputs.size())
        """
        outputs = F.relu(embeddings)
        """
        # print('unpacked_outputs', outputs.size())
        # value has size (batch, time_step, action_dim)
        value = self.final(outputs)
        # print('value', value.size())

        return value, hidden

    def get_value(self, *args):
        return self(*args)

    # Reset LSTM hidden states.
    def reset(self):
        # LSTM hidden states.
        self.hidden = [ch.zeros(1, 1, self.hidden_sizes[1]),
            ch.zeros(1, 1, self.hidden_sizes[1])]

    def pause_history(self):
        self.paused = True

    def continue_history(self):
        self.paused = False

########################
### POLICY NETWORKS
# Discrete and Continuous Policy Examples
########################

'''
A policy network can be any class which is initialized 
with a state_dim and action_dim, as well as optional named arguments.
Must provide:
- A __call__ override (or forward, for nn.Module): 
    * returns a tensor parameterizing a distribution, given a 
    BATCH_SIZE x state_dim tensor representing shape
- A function calc_kl(p, q): 
    * takes in two batches tensors which parameterize probability 
    distributions (of the same form as the output from __call__), 
    and returns the KL(p||q) tensor of length BATCH_SIZE
- A function entropies(p):
    * takes in a batch of tensors parameterizing distributions in 
    the same way and returns the entropy of each element in the 
    batch as a tensor
- A function sample(p): 
    * takes in a batch of tensors parameterizing distributions in
    the same way as above and returns a batch of actions to be 
    performed
- A function get_likelihoods(p, actions):
    * takes in a batch of parameterizing tensors (as above) and an 
    equal-length batch of actions, and returns a batch of probabilities
    indicating how likely each action was according to p.
'''

class DiscPolicy(nn.Module):
    '''
    A discrete policy using a fully connected neural network.
    The parameterizing tensor is a categorical distribution over actions
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES, time_in_state=False, share_weights=False):
        '''
        Initializes the network with the state dimensionality and # actions
        Inputs:
        - state_dim, dimensionality of the state vector
        - action_dim, # of possible discrete actions
        - hidden_sizes, an iterable of length #layers,
            hidden_sizes[i] = number of neurons in layer i
        - time_in_state, a boolean indicating whether the time is 
            encoded in the state vector
        '''
        super().__init__()
        self.activation = ACTIVATION()
        self.time_in_state = time_in_state

        self.discrete = True
        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final = nn.Linear(prev_size, action_dim)

        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

    def forward(self, x):
        '''
        Outputs the categorical distribution (via softmax)
        by feeding the state through the neural network
        '''
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        
        probs = F.softmax(self.final(x))
        return probs

    def calc_kl(self, p, q, get_mean=True): # TODO: does not return a list
        '''
        Calculates E KL(p||q):
        E[sum p(x) log(p(x)/q(x))]
        Inputs:
        - p, first probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        - q, second probability distribution (NUM_SAMPLES, NUM_ACTIONS)
        Returns:
        - Empirical KL from p to q
        '''
        p, q = p.squeeze(), q.squeeze()
        assert shape_equal_cmp(p, q)
        kl = (p * (ch.log(p) - ch.log(q))).sum(-1)
        return kl

    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * ch.log(p)).sum(dim=1)
        return entropies

    def get_loglikelihood(self, p, actions):
        '''
        Inputs:
        - p, batch of probability tensors
        - actions, the actions taken
        '''
        try:
            dist = ch.distributions.categorical.Categorical(p)
            return dist.log_prob(actions)
        except Exception as e:
            raise ValueError("Numerical error")
    
    def sample(self, probs):
        '''
        given probs, return: actions sampled from P(.|s_i), and their
        probabilities
        - s: (batch_size, state_dim)
        Returns actions:
        - actions: shape (batch_size,)
        '''
        dist = ch.distributions.categorical.Categorical(probs)
        actions = dist.sample()
        return actions.long()

    def get_value(self, x):
        # If the time is in the state, discard it
        assert self.share_weights, "Must be sharing weights to use get_value"
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)


class CtsPolicy(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False, activation=None, use_merged_bias=False):
        super().__init__()
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = ACTIVATION()
        print('Using activation function', self.activation)
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state
        self.use_merged_bias = use_merged_bias

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            if use_merged_bias:
                # Use an extra dimension for weight perturbation, simulating bias.
                lin = nn.Linear(prev_size + 1, i, bias=False)
            else:
                lin = nn.Linear(prev_size, i, bias=True)
            initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        if use_merged_bias:
            self.final_mean = nn.Linear(prev_size + 1, action_dim, bias=False)
        else:
            self.final_mean = nn.Linear(prev_size, action_dim, bias=True)
        initialize_weights(self.final_mean, init, scale=0.01)
        
        # For the case where we want to share parameters 
        # between the policy and value networks
        self.share_weights = share_weights
        if share_weights:
            assert not use_merged_bias
            if time_in_state:
                self.final_value = nn.Linear(prev_size + 1, 1)
            else:
                self.final_value = nn.Linear(prev_size, 1)

            initialize_weights(self.final_value, init, scale=1.0)

        stdev_init = ch.zeros(action_dim)
        self.log_stdev = ch.nn.Parameter(stdev_init)

    def forward(self, x):
        # If the time is in the state, discard it
        if self.time_in_state:
            x = x[:,:-1]
        for affine in self.affine_layers:
            if self.use_merged_bias:
                # Generate an extra "one" for each element, which acts as a bias.
                bias_padding = ch.ones(x.size(0),1)
                x = ch.cat((x, bias_padding), dim=1)
            else:
                pass
            x = self.activation(affine(x))
        
        if self.use_merged_bias:
            bias_padding = ch.ones(x.size(0),1)
            x = ch.cat((x, bias_padding), dim=1)
        means = self.final_mean(x)
        std = ch.exp(self.log_stdev)

        return means, std 

    def get_value(self, x):
        assert self.share_weights, "Must be sharing weights to use get_value"

        # If the time is in the state, discard it
        t = None
        if self.time_in_state:
            t = x[...,-1:]
            x = x[...,:-1]

        for affine in self.affine_layers:
            x = self.activation(affine(x))

        if self.time_in_state:
            return self.final_value(ch.cat((x, t), -1))
        else:
            return self.final_value(x)

    def sample(self, p):
        '''
        Given prob dist (mean, var), return: actions sampled from p_i, and their
        probabilities. p is tuple (means, var). means shape 
        (batch_size, action_space), var (action_space,), here are batch_size many
        prboability distributions you're sampling from

        Returns tuple (actions, probs):
        - actions: shape (batch_size, action_dim)
        - probs: shape (batch_size, action_dim)
        '''
        means, std = p
        return (means + ch.randn_like(means)*std).detach()

    def get_loglikelihood(self, p, actions):
        try:    
            mean, std = p
            nll =  0.5 * ((actions - mean) / std).pow(2).sum(-1) \
                   + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] \
                   + self.log_stdev.sum(-1)
            return -nll
        except Exception as e:
            raise ValueError("Numerical error")

    def calc_kl(self, p, q):
        '''
        Get the expected KL distance between two sets of gaussians over states -
        gaussians p and q where p and q are each tuples (mean, var)
        - In other words calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))]
        - From https://stats.stackexchange.com/a/60699
        '''
        p_mean, p_std = p
        q_mean, q_std = q
        p_var, q_var = p_std.pow(2), q_std.pow(2)
        assert shape_equal([-1, self.action_dim], p_mean, q_mean)
        assert shape_equal([self.action_dim], p_var, q_var)

        d = q_mean.shape[1]
        diff = q_mean - p_mean

        log_quot_frac = ch.log(q_var).sum() - ch.log(p_var).sum()
        tr = (p_var / q_var).sum()
        quadratic = ((diff / q_var) * diff).sum(dim=1)

        kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
        assert kl_sum.shape == (p_mean.shape[0],)
        return kl_sum

    def entropies(self, p):
        '''
        Get entropies over the probability distributions given by p
        p_i = (mean, var), p mean is shape (batch_size, action_space),
        p var is shape (action_space,)
        '''
        _, std = p
        detp = determinant(std)
        d = std.shape[0]
        entropies = ch.log(detp) + .5 * (d * (1. + math.log(2 * math.pi)))
        return entropies
    
    def reset(self):
        return

    def pause_history(self):
        return

    def continue_history(self):
        return


class CtsLSTMPolicy(CtsPolicy):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, init, hidden_sizes=HIDDEN_SIZES,
                 time_in_state=False, share_weights=False, activation=None, use_merged_bias=False):
        print('Using LSTM policy!!')
        assert share_weights is False
        assert use_merged_bias is False
        assert time_in_state is False
        super().__init__(state_dim, action_dim, init, hidden_sizes, time_in_state, share_weights, activation, use_merged_bias)
        self.hidden_sizes = hidden_sizes
        self.action_dim = action_dim
        self.discrete = False
        self.time_in_state = time_in_state
        self.use_merged_bias = use_merged_bias
        self.share_weights = share_weights
        self.paused = False

        self.embedding_layer = nn.Linear(state_dim, self.hidden_sizes[0])
        initialize_weights(self.embedding_layer, init, scale=0.01)

        self.lstm = nn.LSTM(input_size=self.hidden_sizes[0], hidden_size=self.hidden_sizes[1], num_layers=1, batch_first=True)

        self.output_layer = nn.Linear(self.hidden_sizes[-1], action_dim)
        initialize_weights(self.output_layer, init, scale=1.0)

        stdev_init = ch.zeros(action_dim)
        self.log_stdev = ch.nn.Parameter(stdev_init)

        # LSTM hidden states.
        self.hidden = [ch.zeros(1, 1, self.hidden_sizes[1]),
            ch.zeros(1, 1, self.hidden_sizes[1])]

    def forward(self, x, not_dones=None):
        if isinstance(x, ch.Tensor) and x.size(0) != 1:
            # We are given a batch of states. We need not_dones to split them into episodes.
            assert not_dones is not None
            # input dimension must be in shape (N, state_dim)
            # not_dones has shape: (N, )
            # features shape: (N, hidden_dim)
            features = self.embedding_layer(x)
            # Pack states into episodes according to not_dones
            packed_features = pack_history(features, not_dones)
            # Run LSTM
            outputs, _ = self.lstm(packed_features)
            # pad output results
            padded, lengths = pad_packed_sequence(outputs, batch_first=True)
            # concate output to a single array (N, hidden_dim)
            hidden = unpack_history(padded, lengths)
            """
            hidden = F.relu(features)
            """
            # final output, apply linear transformation on hidden output.
            means = self.output_layer(hidden)
            std = ch.exp(self.log_stdev)
            return means, std

        if isinstance(x, ch.Tensor) and x.ndim == 2:  # inference mode, state input one by one. No time dimension.
            assert not_dones is None
            # it must have batch size 1.
            assert x.size(0) == 1
            # input x dimension: (1, time_slice, state_dim)
            # We use torch.nn.utils.rnn.pack_padded_sequence() as input.
            embedding = self.embedding_layer(x).unsqueeze(0)
            # embedding dimension: (batch, time_slice, hidden_dim)
            _, hidden = self.lstm(embedding, self.hidden)
            # _, hidden = self.lstm(embedding)
            """
            hidden = F.relu(embedding)
            hidden = [hidden, hidden]
            """
            # hidden dimension: (1, 1, hidden_size)
            output = self.output_layer(hidden[0])
            # save hidden state.
            if not self.paused:
                self.hidden[0] = hidden[0]
                self.hidden[1] = hidden[1]

            means = output.squeeze(0)  # remove the extra dimension.
            std = ch.exp(self.log_stdev)

            return means, std
        else:  # with time dimension, used for training LSTM.
            raise ValueError(f'Unsupported input {x} to LSTM policy')

    def multi_forward(self, x, hidden=None):
        embeddings = self.embedding_layer(x)
        # print('embeddings', embeddings.size())
        # Run LSTM with packed sequence
        outputs, hidden = self.lstm(embeddings, hidden)
        # desired outputs dimension: (batch, time_step, hidden_size)
        # print('outputs', outputs.size())
        """
        outputs = F.relu(embeddings)
        """
        # print('unpacked_outputs', outputs.size())
        # means has size (batch, time_step, action_dim)
        means = self.output_layer(outputs)
        # print('means', means.size())

        # std is still time and history independent.
        std = ch.exp(self.log_stdev)
        return means, std, hidden

    # Reset LSTM hidden states.
    def reset(self):
        # LSTM hidden states.
        self.hidden = [ch.zeros(1, 1, self.hidden_sizes[1]),
            ch.zeros(1, 1, self.hidden_sizes[1])]

    def pause_history(self):
        self.paused = True

    def continue_history(self):
        self.paused = False


class CtsPolicyLarger(CtsPolicy):

    def __init__(self, state_dim, action_dim, init, 
                 time_in_state=False, share_weights=False, activation=None, use_merged_bias=False):
        super().__init__(state_dim, action_dim, init, hidden_sizes=[400,300], time_in_state=False,
                share_weights=False, activation='relu', use_merged_bias=False)

    def forward(self, x):
        mean, std = super().forward(x)
        return ch.tanh(mean), std


class CtsPolicySAC(CtsPolicy):

    def __init__(self, state_dim, action_dim, init, 
                 time_in_state=False, share_weights=False, activation=None, use_merged_bias=False):
        super().__init__(state_dim, action_dim, init, hidden_sizes=[256,256], time_in_state=False,
                share_weights=False, activation='relu', use_merged_bias=False)

    def forward(self, x):
        mean, std = super().forward(x)
        return ch.tanh(mean), std

## Retrieving networks
# Make sure to add newly created networks to these dictionaries!

POLICY_NETS = {
    "DiscPolicy": DiscPolicy,
    "CtsPolicy": CtsPolicy,
    "CtsPolicyLarger": CtsPolicyLarger,
    "CtsPolicySAC": CtsPolicySAC,
}

VALUE_NETS = {
    "ValueNet": ValueDenseNet,
}

def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewCls


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "leaky0.05": partialclass(nn.LeakyReLU, negative_slope=0.05),
    "leaky0.1": partialclass(nn.LeakyReLU, negative_slope=0.1),
    "hardtanh": nn.Hardtanh,
}

def activation_with_name(name):
    return ACTIVATIONS[name]

def policy_net_with_name(name):
    return POLICY_NETS[name]

def value_net_with_name(name):
    return VALUE_NETS[name]
