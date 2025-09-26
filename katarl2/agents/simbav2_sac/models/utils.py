import torch
import torch.nn.functional as F
from typing import Literal

EPS = 1e-8


def l2normalize(x, dim):
    l2norm = torch.linalg.norm(x, ord=2, dim=dim, keepdims=True)
    x = x / torch.maximum(l2norm, torch.tensor(EPS))  # prevent zero vector
    return x


def soft_ce(pred, target, cfg, input_mode: Literal['logits', 'log_prob'] = 'logits'):
    """Computes the cross entropy loss between predictions and soft targets."""
    if input_mode == 'logits':
        pred = F.log_softmax(pred, dim=-1)
    else:
        pred = pred
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(
        1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset
    )
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(
        cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype
    )
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)

import torch
import torch.nn.functional as F

def categorical_td_loss(
    pred_log_probs: torch.Tensor,     # Shape: (batch_size, num_bins)
    target_log_probs: torch.Tensor,   # Shape: (batch_size, num_bins)
    reward: torch.Tensor,             # Shape: (batch_size,)
    done: torch.Tensor,               # Shape: (batch_size,)
    actor_entropy: torch.Tensor,      # Shape: (batch_size,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    """
    Calculates the categorical temporal difference loss for a distributional critic,
    incorporating the entropy term from Soft Actor-Critic (SAC).

    Args:
        pred_log_probs: Log probabilities of the Q-value distribution from the online critic.
        target_log_probs: Log probabilities from the target critic for the next state.
        reward: The reward received for the current transition.
        done: A boolean tensor indicating if the episode has terminated.
        actor_entropy: The entropy term (-alpha * log_pi) for the SAC update.
        gamma: The discount factor.
        num_bins: The number of atoms in the discrete value distribution.
        min_v: The minimum value of the distribution's support.
        max_v: The maximum value of the distribution's support.

    Returns:
        The calculated cross-entropy loss as a scalar tensor.
    """
    # Ensure inputs are on the same device and reshape for broadcasting
    device = pred_log_probs.device
    reward = reward.view(-1, 1)
    # Use float() for done tensor to multiply with other floats
    done = done.float().view(-1, 1)
    actor_entropy = actor_entropy.view(-1, 1)

    # === Step 1: Compute the projected target distribution's support ===
    # This is the core of the distributional soft Bellman update.
    # We don't update a single Q-value, but every atom (bin_value) of the support.

    # The fixed locations of the bins (atoms) for the value distribution
    bin_values = torch.linspace(min_v, max_v, num_bins, device=device).view(1, -1)

    # Apply the soft Bellman operator to each atom of the target distribution's support.
    # This computes where each atom z_j should be moved to: Tz_j = r + gamma * (z_j - entropy)
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)

    # Clip the results to be within the predefined value range [min_v, max_v]
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # === Step 2: Project the target distribution onto the original support ===
    # The new atom locations (target_bin_values) are continuous and don't align with
    # our fixed bins. We need to distribute their probability mass to the nearest fixed bins.

    delta_z = (max_v - min_v) / (num_bins - 1)
    # Calculate the fractional index of each new atom location
    b = (target_bin_values - min_v) / delta_z
    
    # Determine the lower and upper bin indices for projection
    # .long() is required for using these as indices
    l = b.floor().long()
    u = b.ceil().long()

    # Handle the edge case where a projected atom lands exactly on a bin
    # In this case, l == u, and we want all probability mass on that one bin.
    l[(u > 0) & (l == u)] -= 1
    u[(l < (num_bins - 1)) & (l == u)] += 1
    
    # Get the probabilities from the target network's log-probs
    _target_probs = torch.exp(target_log_probs) # Shape: (batch_size, num_bins)

    # Initialize the final target probability distribution with zeros
    target_probs = torch.zeros_like(pred_log_probs)

    # Calculate the probability mass to be distributed to the lower and upper bins
    # This is the linear interpolation step.
    mass_l = _target_probs * (u.float() - b)
    mass_u = _target_probs * (b - l.float())

    # Use scatter_add_ to efficiently project the probability masses.
    # This is a vectorized way of adding `mass_l` to indices `l` and `mass_u` to indices `u`.
    target_probs.scatter_add_(dim=1, index=l, src=mass_l)
    target_probs.scatter_add_(dim=1, index=u, src=mass_u)

    # === Step 3: Calculate the cross-entropy loss ===
    # The `target_probs` are treated as the ground-truth label.
    # We use stop_gradient (or .detach() in PyTorch) because we don't backpropagate through the target.
    loss = -torch.mean(torch.sum(target_probs.detach() * pred_log_probs, dim=1))

    return loss
