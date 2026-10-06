"""General utility functions used by the model and training script."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def str2bool(string):
    """ Convert string to corresponding boolean.
        -  string : str
    """
    if string in ["True","true","1"]:
        return True
    elif string in ["False","false","0"]:
        return False
    else :
        raise ValueError(
            f"expected a boolean value (true/false/1/0), got {string!r}")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
common_layers
'''

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def safe_l2_normalize(x, dim=1, min_norm=1e-3):
    """FP16-safe normalization with zero gradient for near-zero vectors.

    Casting the forward computation to FP32 alone is insufficient: its gradient
    can still overflow when it is cast back to an FP16 activation.  Near-zero
    embeddings therefore use a constant zero branch.
    """
    x32 = x.float()
    norm = torch.linalg.vector_norm(x32, dim=dim, keepdim=True)
    normalized = x32 / norm.clamp_min(min_norm)
    return torch.where(norm >= min_norm, normalized,
                       torch.zeros_like(normalized))

def tensor_version_or_none(tensor):
    """Return the in-place version counter, or None for inference tensors."""
    try:
        return tensor._version
    except RuntimeError:
        # Tensors created inside torch.inference_mode() do not track versions.
        # Such masks are validated on every call rather than cached unsafely.
        return None

def info_nce_loss(z_a, z_b, temperature=0.07):
    """InfoNCE contrastive loss for maximizing cross-view consistency.

    Treats embeddings of the same sample from different views as positive pairs,
    and embeddings from different samples as negative pairs.

    Args:
        z_a: [batch, dim] float tensor, embeddings from view A.
        z_b: [batch, dim] float tensor, embeddings from view B.
        temperature: scaling factor controlling the concentration of the
            similarity distribution. Smaller values make the distribution
            sharper (more discriminative). Typical range: 0.05 ~ 0.5.

    Returns:
        Scalar loss value.
    """
    if z_a.shape[0] == 0:
        return z_a.new_zeros(())
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Keep the complete contrastive calculation in FP32. safe_l2_normalize also
    # prevents the FP32 gradient from overflowing when cast back to FP16.
    with torch.amp.autocast('cuda', enabled=False):
        z_a = safe_l2_normalize(z_a, dim=1, min_norm=1e-3)
        z_b = safe_l2_normalize(z_b, dim=1, min_norm=1e-3)
        logits = torch.mm(z_a, z_b.T) / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)


'''
CLCL model
'''
