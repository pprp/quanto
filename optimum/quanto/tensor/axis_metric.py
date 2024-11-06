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
    """Compute variance along specified axis"""
    return torch.var(weight, dim=axis)

def std_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute standard deviation along specified axis"""
    return torch.std(weight, dim=axis)

def mean_abs_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute mean absolute value along specified axis"""
    return torch.mean(torch.abs(weight), dim=axis)

def l2_norm_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute L2 norm along specified axis"""
    return torch.norm(weight, p=2, dim=axis)

def sparsity_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute sparsity (fraction of zeros) along specified axis"""
    return torch.mean((weight == 0).float(), dim=axis)

def kurtosis_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute kurtosis along specified axis"""
    mean = torch.mean(weight, dim=axis, keepdim=True)
    std = torch.std(weight, dim=axis, keepdim=True)
    z = (weight - mean) / std
    return torch.mean(z.pow(4), dim=axis) - 3

def entropy_axis_metric(weight: torch.Tensor, axis: int = 0, eps: float = 1e-10) -> torch.Tensor:
    """Compute entropy of absolute values along specified axis"""
    abs_weight = torch.abs(weight)
    # Normalize to get probability distribution
    prob = abs_weight / (torch.sum(abs_weight, dim=axis, keepdim=True) + eps)
    return -torch.sum(prob * torch.log(prob + eps), dim=axis)

def peak_to_peak_axis_metric(weight: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Compute range (max - min) along specified axis"""
    return torch.amax(weight, dim=axis) - torch.amin(weight, dim=axis)
