from typing import Dict, Tuple, Optional, Union
import torch
import re

from ..qtype import qtype
from ..grouped import group
from .optimizer import Optimizer

class AdaptiveAxisOptimizer(Optimizer):
    """
    Optimizer that adaptively selects quantization axis based on layer name
    """
    def __init__(
        self,
        base_optimizer: Optimizer,
        axis_mapping: Dict[str, int],
        pattern_mapping: Dict[str, int] = None
    ):
        """
        Args:
            base_optimizer: Base optimizer (e.g. MaxOptimizer)
            axis_mapping: Direct mapping from layer names to axes
            pattern_mapping: Mapping from regex patterns to axes
        """
        self.base_optimizer = base_optimizer
        self.axis_mapping = axis_mapping
        self.pattern_mapping = pattern_mapping or {}
        
    def _get_axis_for_layer(self, layer_name: str) -> int:
        """Determine quantization axis based on layer name"""
        # 1. Direct match
        if layer_name in self.axis_mapping:
            return self.axis_mapping[layer_name]
            
        # 2. Pattern match
        for pattern, axis in self.pattern_mapping.items():
            if re.match(pattern, layer_name):
                return axis
                
        # 3. Default to per-channel
        return 0
    
    def __call__(
        self,
        base: torch.Tensor,
        qtype: qtype,
        axis: int,
        group_size: Optional[int] = None,
        layer_name: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Override call method to select appropriate axis based on layer name
        """
        if layer_name is not None:
            axis = self._get_axis_for_layer(layer_name)
            
        if axis not in [0, -1]:
            raise ValueError("axis parameter must be 0 (first axis) or -1 (last axis)")
        if group_size is not None:
            base = group(base, axis, group_size)
        if axis is not None and base.shape[axis] == 1:
            axis = None
            
        return self.base_optimizer(base, qtype, axis, group_size)
