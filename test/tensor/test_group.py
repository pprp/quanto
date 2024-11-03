from optimum.quanto.tensor.grouped import group, ungroup

import torch

def test_group_ungroup():
    # Create a random tensor 4096x4096
    tensor = torch.randn(4096, 4096)
    
    # Test grouping along axis 0 with group size 128
    grouped_0 = group(tensor, axis=0, group_size=128)
    ungrouped_0 = ungroup(grouped_0, axis=0, orig_shape=tensor.shape)
    print("Tensor matches ungrouped_0:", torch.allclose(tensor, ungrouped_0))
    print("grouped_0 shape:", grouped_0.shape, "(4096*4096/128, 128)")
    
    # Test grouping along axis -1 with group size 128 
    grouped_1 = group(tensor, axis=-1, group_size=128)
    ungrouped_1 = ungroup(grouped_1, axis=-1, orig_shape=tensor.shape)
    print("Tensor matches ungrouped_1:", torch.allclose(tensor, ungrouped_1))
    print("grouped_1 shape:", grouped_1.shape, "(128, 4096*4096/128)")


test_group_ungroup()