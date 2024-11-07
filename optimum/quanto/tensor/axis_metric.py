import torch

__all__ = [
    "variance_axis_metric",
    "std_axis_metric", 
    "mean_abs_axis_metric",
    "l2_norm_axis_metric",
    "sparsity_axis_metric",
    "kurtosis_axis_metric",
    "entropy_axis_metric",
    "peak_to_peak_axis_metric"
]

def variance_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute variance along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        # For 3D tensors, compute variance along batch dimension first
        weight = torch.var(weight, dim=0, unbiased=False)
    variance = torch.var(weight, dim=1 if axis == 0 else -1, unbiased=False)
    return torch.mean(variance)

def std_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute standard deviation along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension  
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        weight = torch.std(weight, dim=0, unbiased=False)
    std_dev = torch.std(weight, dim=1 if axis == 0 else -1, unbiased=False)
    return torch.mean(std_dev)

def mean_abs_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute mean absolute value along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        weight = torch.mean(torch.abs(weight), dim=0)
    mean_abs = torch.mean(torch.abs(weight), dim=1 if axis == 0 else -1)
    return torch.mean(mean_abs)

def l2_norm_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute L2 norm along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        weight = torch.norm(weight, p=2, dim=0)
    l2_norm = torch.norm(weight, p=2, dim=1 if axis == 0 else -1)
    return torch.mean(l2_norm)

def sparsity_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute sparsity (fraction of zeros) along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        weight = torch.mean((weight == 0).float(), dim=0)
    sparsity = torch.mean((weight == 0).float(), dim=1 if axis == 0 else -1)
    return torch.mean(sparsity)

def kurtosis_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute kurtosis along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        # Compute kurtosis for each batch then average
        mean = torch.mean(weight, dim=0, keepdim=True)
        std = torch.std(weight, dim=0, keepdim=True, unbiased=False)
        z = (weight - mean) / (std + 1e-10)
        weight = torch.mean(z.pow(4), dim=0) - 3
    axis_dim = 1 if axis == 0 else -1
    mean = torch.mean(weight, dim=axis_dim, keepdim=True)
    std = torch.std(weight, dim=axis_dim, keepdim=True, unbiased=False)
    z = (weight - mean) / (std + 1e-10)
    kurtosis = torch.mean(z.pow(4), dim=axis_dim) - 3
    return torch.mean(kurtosis)

def entropy_axis_metric(weight: torch.Tensor, axis: int = 0, eps: float = 1e-10) -> torch.Tensor:
    """Compute entropy of absolute values along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        abs_weight = torch.abs(weight)
        prob = abs_weight / (torch.sum(abs_weight, dim=0, keepdim=True) + eps)
        weight = -torch.sum(prob * torch.log(prob + eps), dim=0)
    axis_dim = 1 if axis == 0 else -1
    abs_weight = torch.abs(weight)
    prob = abs_weight / (torch.sum(abs_weight, dim=axis_dim, keepdim=True) + eps)
    entropy = -torch.sum(prob * torch.log(prob + eps), dim=axis_dim)
    return torch.mean(entropy)

def peak_to_peak_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute range (max - min) along specified axis and average across dimensions
    For a tensor of shape [bs, 4096, 11080]:
    - axis=0 computes along 4096 dimension
    - axis=-1 computes along 11080 dimension
    Returns a scalar value.
    """
    if len(weight.shape) == 3:
        weight = torch.amax(weight, dim=0) - torch.amin(weight, dim=0)
    p2p = torch.amax(weight, dim=1 if axis == 0 else -1) - torch.amin(weight, dim=1 if axis == 0 else -1)
    return torch.mean(p2p)
