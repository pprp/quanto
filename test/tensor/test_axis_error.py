from optimum.quanto import (
    WeightQBitsTensor,
    qint4,
    MaxOptimizer
)
import torch
import scipy
import numpy as np

def clip_tensor_to_multiple(tensor, multiple):
    shape = tensor.shape
    new_shape = tuple((dim // multiple) * multiple for dim in shape)
    slices = tuple(slice(0, new_dim) for new_dim in new_shape)
    return tensor[slices]

def perform_quant_cal_error(ori_weight_tensor, metric='range_var', qtype=qint4, optimizer=MaxOptimizer(), device='cuda'):
    # breakpoint()
    ori_weight_tensor = ori_weight_tensor[0, :1024, :1024]
    # clip_tensor_to_multiple(ori_weight_tensor, 128)
    # breakpoint()
    ori_weight = ori_weight_tensor.numpy()
    # if metric =='kurtosis':
    #     if np.mean(scipy.stats.kurtosis(ori_weight, axis=1)) > np.mean(scipy.stats.kurtosis(ori_weight, axis=0)) :
    #         axis = -1
    #     else:
    #         axis = 0
    
    axis = -1
    print(f"axis: {axis}")

    ori_weight_tensor.to(device)
    scale, shift = optimizer(ori_weight_tensor, qtype=qtype, axis=axis, group_size=128)
    q_weight_tensor = WeightQBitsTensor.quantize(ori_weight_tensor, qtype, axis, 128, scale, shift, optimized=True)
    q_weight_tensor = q_weight_tensor.flatten()
    ori_weight_tensor = ori_weight_tensor.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(q_weight_tensor, ori_weight_tensor, dim = 0)
    l2_loss = torch.nn.functional.mse_loss(q_weight_tensor, ori_weight_tensor)
    return cos_sim, l2_loss, axis


if __name__ == '__main__':
    import numpy as np
    data = np.load('single_transformer_blocks.5.attn.to_v.npz')

    tensor_3d = torch.from_numpy(data['arr_0'])
    
    cos_sim, l2_loss, axis = perform_quant_cal_error(tensor_3d, metric='kurtosis')
    print(f"cos_sim: {cos_sim}, l2_loss: {l2_loss}, axis: {axis}")