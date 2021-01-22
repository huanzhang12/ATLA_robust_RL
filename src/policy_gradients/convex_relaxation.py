import random, sys, time, multiprocessing
from auto_LiRPA import BoundedModule, BoundedTensor, BoundedParameter
from auto_LiRPA.perturbations import *

from policy_gradients.models import activation_with_name


forward_one = True

## Step 1: Initial original model as usual, see model details in models/sample_models.py
class RelaxedCtsPolicyForState(nn.Module):
    def __init__(self, state_dim=11, action_dim=3, init=None, hidden_sizes=[64, 64],
                 time_in_state=False, share_weights=False, activation='tanh', policy_model=None):
        super().__init__()
        assert time_in_state is False
        assert share_weights is False
        assert init is None
        if isinstance(activation, str):
            self.activation = activation_with_name(activation)()
        else:
            # Default to tanh.
            self.activation = nn.Tanh()
        self.action_dim = action_dim

        if policy_model is None:
            # Create our own layers.
            self.affine_layers = nn.ModuleList()
            prev_size = state_dim
            for i in hidden_sizes:
                lin = nn.Linear(prev_size, i, bias=False)
                self.affine_layers.append(lin)
                prev_size = i

            self.final_mean = nn.Linear(prev_size, action_dim, bias=False)
            
            stdev_init = torch.zeros(action_dim)
            # FIXME: name of this variable must contain "weight" due to a bug in auto_LiRPA.
            if not forward_one:
                self.log_weight = torch.nn.Parameter(stdev_init)
        else:
            print("Create Relaxed model without duplicating parameters...")
            # Copy parameters from an existing model, do not create new parameters!
            self.affine_layers = policy_model.affine_layers
            # Copy the final mean vector.
            self.final_mean = policy_model.final_mean
            if not forward_one:
                # Copy the log of variance.
                self.log_weight = policy_model.log_stdev

    '''
    Compute the L2 distance of mean vectors, to bound KL divergence.
    '''
    if forward_one:
        def forward(self, x):
            for affine in self.affine_layers:
                x = self.activation(affine(x))
            means = self.final_mean(x)
            return means
    else:
        def forward(self, x, old_means):
            for affine in self.affine_layers:
                # Generate an extra "one" for each element, which acts as a bias.
                x = self.activation(affine(x))
            means = self.final_mean(x)
            diff = means - old_means
            stdev = torch.exp(self.log_weight)
            return ((diff * diff) / stdev).sum(axis=-1, keepdim=True)

    def get_means(self, x):
        for affine in self.affine_layers:
            x = affine(x)
            x = self.activation(x)
        means = self.final_mean(x)
        return means

def intermediate_to_kl(lb, ub, means, stdev=None):
    lb = lb - means
    ub = ub - means
    u = torch.max(lb.abs(), ub.abs())
    if stdev is None:
        return (u * u).sum(axis=-1, keepdim=True)
    else:
        return ((u * u) / (stdev * stdev)).sum(axis=-1, keepdim=True)

if forward_one:
    def get_kl_bound(model, x, means, eps, beta=None, stdev=None, use_full_backward=False):
        # Set each layer's perturbation eps and log_stdev's perturbation.
        x = BoundedTensor(x, ptb=PerturbationLpNorm(norm=np.inf, eps=eps)).requires_grad_(False)
        if forward_one:
            inputs = (x, )
        else:
            inputs = (x, means)
        if use_full_backward:
            # Full backward method, tightest bound.
            ilb, iub = model.compute_bounds(inputs, IBP=False, C=None, method="backward", bound_lower=True, bound_upper=True)
            # Fake beta, avoid backward below.
            beta = 1.0
        else:
            # IBP Pass.
            ilb, iub = model.compute_bounds(inputs, IBP=True, C=None, method=None, bound_lower=True, bound_upper=True)
        if beta is None or (1 - beta) > 1e-20:
            # CROWN Pass.
            clb, cub = model.compute_bounds(x=None, IBP=False, C=None, method='backward', bound_lower=True, bound_upper=True)
        if beta is None:
            # Bound final output neuron.
            ikl = intermediate_to_kl(ilb, iub, means, stdev=stdev)
            ckl = intermediate_to_kl(clb, cub, means, stdev=stdev)
            return ikl, ckl
        else:
            # Beta schedule is from 0 to 1.
            if 1 - beta < 1e-20:
                lb = ilb
                ub = iub
            else:
                lb = beta * ilb + (1 - beta) * clb
                ub = beta * iub + (1 - beta) * cub
            kl = intermediate_to_kl(lb, ub, means, stdev=stdev)
            return kl
else:
    def get_kl_bound(model, x, means, eps):
        # Set each layer's perturbation eps and log_stdev's perturbation.
        x = BoundedTensor(x, ptb=PerturbationLpNorm(norm=np.inf, eps=eps))
        if forward_one:
            inputs = (x, )
        else:
            inputs = (x, means)
        # IBP Pass.
        _, iub = model.compute_bounds(inputs, IBP=True, C=None, method=None, bound_lower=False, bound_upper=True)
        # CROWN Pass.
        _, cub = model.compute_bounds(x=None, IBP=False, C=None, method='backward', bound_lower=False, bound_upper=True)
        # iub = cub
        return iub, cub

def compute_perturbations(model, x, means, perturbations):
    use_ibp = True
    method = 'backward'
    x = BoundedTensor(x, ptb=PerturbationLpNorm(norm=np.inf, eps=0))
    inputs = (x, means)
    for p in perturbations:
        x.ptb.eps = p
        lb, ub = model.compute_bounds(inputs, IBP=use_ibp, C=None, method=method, bound_lower=True, bound_upper=True)
        lb = lb.detach().cpu().numpy().squeeze()
        ub = ub.detach().cpu().numpy().squeeze()
        print("eps={:.4f}, lb={}, ub={}".format(p, lb, ub))
    x.ptb.eps = 0.0
    lb, ub = model.compute_bounds(inputs, IBP=use_ibp, C=None, method=method, bound_lower=True, bound_upper=True)
    lb = lb.detach().cpu().numpy().squeeze()
    ub = ub.detach().cpu().numpy().squeeze()
    print("eps=0.0000, lb={}, ub={}".format(lb, ub))


def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    input_size = 17
    action_size = 6

    ## Step 1: Initial original model as usual; note that this model has BoundedParameter as its weight parameters
    model_ori = RelaxedCtsPolicyForState(state_dim=input_size, action_dim=action_size)
    state_dict = torch.load('test_policy_net.model')
    if not forward_one:
        state_dict['log_weight'] = state_dict['log_stdev']
    del state_dict['log_stdev']
    # model_ori.load_state_dict(state_dict)

    ## Step 2: Prepare dataset as usual
    dummy_input1 = torch.randn(1, input_size)
    dummy_input2 = torch.randn(1, action_size)
    if forward_one:
        inputs = (dummy_input1, )
    else:
        inputs = (dummy_input1, dummy_input2)
    model_ori(*inputs)
    # inputs = (dummy_input1, )
    # dummy_input2 = model_ori.get_means(dummy_input1)

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundedModule(model_ori, inputs)

    all_states = x = torch.randn(2, input_size)
    means = model_ori.get_means(x).detach()
    if forward_one:
        print('prediction', model_ori(x).sum())
    else:
        print('prediction', model_ori(x, means).sum())
    action_means = means


    perturbations = np.arange(0.0, 0.1, 0.01)
    compute_perturbations(model, x, means, perturbations)


    if forward_one:
        # pred = model_ori(all_states)
        # pred = ((pred - means) ** 2).mean()
        ikl, ckl = get_kl_bound(model, all_states, action_means, 0.1)
        ikl, ckl = get_kl_bound(model, all_states, action_means, 0.0)
        print('ikl', ikl.mean().item())
        print('ckl', ckl.mean().item())
        pred = (0.5 * ikl + 0.5 * ckl).mean()
        pred.backward()
        print('pred', pred.item())
    else:
        iub, cub = get_kl_bound(model, all_states, action_means, 0.1)
        # iub, cub = get_kl_bound(model, all_states, action_means, 0)
        # iub, cub = model_ori(all_states, action_means).mean()
        print('iub', iub.mean().item())
        print('cub', cub.mean().item())
        kl = (0.5 * iub + 0.5 * cub).mean()
        kl.backward()
        print('kl', kl.item())
    for p in model.parameters():
        if p.grad is not None:
            print(p.size(), p.grad.abs().sum().item())
            # print(p.size(), p.grad)
        else:
            print(p.size(), p.grad)


if __name__ == "__main__":
    main()
