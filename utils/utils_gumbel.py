
import numpy as np
import torch

def sample_gumbel(mu_size):
    """Sample a Gumbel(mu)."""
    m = torch.distributions.gumbel.Gumbel(torch.zeros(mu_size), torch.ones(mu_size))
    return m.sample()
    # return -np.log(np.random.exponential()) + mu


def sample_truncated_gumbel(mu, b):
    #Sample a Gumbel(mu) truncated to be less than b.
    m = torch.distributions.exponential.Exponential(torch.ones_like(mu))
    return -torch.log(m.sample() + torch.exp(-b + mu)) + mu

"""
def sample_truncated_gumbel(mu, b):
    # Sample a Gumbel(mu) truncated to be less than b.
    return -np.log((np.random.exponential()) + np.exp(-b + mu)) + mu

"""
def sample_gumbel_argmax(logits):
    """Sample from a softmax distribution over logits.

    TODO: check this is correct.

    Args:
    logits: A flat numpy array of logits.

    Returns:
    A sample from softmax(logits).
    """
    #phi_x_g = -np.log(np.random.exponential(size=logits.shape)) + logits
    #argmax = np.argmax(phi_x_g)

    m = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits),torch.ones_like(logits))
    phi_x_g = m.sample() + logits
    _,argmax = torch.max(phi_x_g,-1)

    return phi_x_g, argmax


"""
def logsumexp(logits):
    c = torch.max(logits)
    return torch.log(torch.sum(torch.exp(logits - c))) + c

"""
def logsumexp(logits):
    c = np.max(logits)
    return np.log(np.sum(np.exp(logits - c))) + c


def log_softmax(logits, axis=1):
    """Normalize logits per row so that they are logprobs.
  
      TODO: check this is correct. 
    """
    maxes = np.max(logits, axis=axis, keepdims=True)
    offset_logits = logits - maxes
    log_zs = np.log(np.sum(np.exp(offset_logits), axis=axis, keepdims=True))
    return offset_logits - log_zs

def use_gpu(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
