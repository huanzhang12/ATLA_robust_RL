import torch as ch
import numpy as np
from .torch_utils import *
from torch.nn.utils import parameters_to_vector as flatten
from torch.nn.utils import vector_to_parameters as assign
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from .steps import value_loss_returns, value_loss_gae, adv_normalize


def log_weight_updates(agent, old_params, new_params, table='weight_updates'):
    row = {}
    for name in old_params.keys():
        new_w = new_params[name]
        old_w = old_params[name]
        diff_w = new_w - old_w
        l1 = ch.norm(new_w.view(-1), p=1).item()
        l2 = ch.norm(new_w.view(-1), p=2).item()
        linf = ch.norm(new_w.view(-1), p=np.inf).item()
        l1_delta = ch.norm(diff_w.view(-1), p=1).item()
        l2_delta = ch.norm(diff_w.view(-1), p=2).item()
        linf_delta = ch.norm(diff_w.view(-1), p=np.inf).item()
        print('layer {}:\tlinf={:.5g} l2={:.5g} l1={:.5g}\tdelta_linf={:.5g} delta_l2={:.5g} delta_l1={:.5g}'.format(
            name, linf, l2, l1, linf_delta, l2_delta, l1_delta))
        name += '.'
        row[name + "l1"] = l1
        row[name + "l2"] = l2
        row[name + "linf"] = linf
        row[name + "delta_l1"] = l1_delta
        row[name + "delta_l2"] = l2_delta
        row[name + "delta_linf"] = linf_delta
    agent.store.log_table_and_tb(table, row)


#####
# Understanding TRPO approximations for KL constraint
#####

def paper_constraints_logging(agent, saps, old_pds, table):
    '''Computes average, max KL and max clipping ratio'''
    # New mean and variance.
    new_pds = agent.policy_model(saps.states)
    # Get the likelyhood of old actions under the new Gaussian distribution.
    new_log_ps = agent.policy_model.get_loglikelihood(new_pds,
                                                    saps.actions)

    # Likelyhood of the old actions, under new and old action distributions.
    ratios = ch.exp(new_log_ps - saps.action_log_probs)
    max_rat = ratios.max()

    kls = agent.policy_model.calc_kl(old_pds, new_pds)
    avg_kl = kls.mean()
    max_kl = kls.max()

    row = {
        'avg_kl':avg_kl,
        'max_kl':max_kl,
        'max_ratio':max_rat,
        'opt_step':agent.n_steps,
    }
    print(f'Step {agent.n_steps}, avg_kl {avg_kl:.5f}, max_kl {max_kl:.5f}, max_ratio {max_rat:.5f}')

    for k in row:
        if k != 'opt_step':
            row[k] = float(row[k])

    agent.store.log_table_and_tb(table, row)

##
# Treating value learning as a supervised learning problem:
# How well do we do?
##
def log_value_losses(agent, saps, label_prefix, table='value_data'):
    '''
    Computes the validation loss of the value function modeling it 
    as a supervised learning of returns. Calculates the loss using 
    all three admissible loss functions (returns, consistency, mixed).
    Inputs: None
    Outputs: None, logs to the store 
    '''
    with ch.no_grad():
        # Compute validation loss
        new_values = agent.val_model(saps.states).squeeze(-1)
        args = [new_values, saps.returns, saps.advantages, saps.not_dones,
                agent.params, saps.values]
        returns_loss, returns_mre, returns_msre = value_loss_returns(*args, re=True)
        gae_loss, gae_mre, gae_msre = value_loss_gae(*args, re=True)

        agent.store.log_table_and_tb(table, {
            ('%s_returns_loss' % label_prefix): returns_loss,
            ('%s_gae_loss' % label_prefix): gae_loss,
        })
