import unittest
import torch
from optimum.quanto.tensor import axis_metric


def test_variance_axis_metric(tensor_3d):
    print("Testing variance_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.variance_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.variance_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_std_axis_metric(tensor_3d):
    print("Testing std_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.std_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.std_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_mean_abs_axis_metric(tensor_3d):
    print("Testing mean_abs_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.mean_abs_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.mean_abs_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_l2_norm_axis_metric(tensor_3d):
    print("Testing l2_norm_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.l2_norm_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.l2_norm_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_sparsity_axis_metric(tensor_3d):
    print("Testing sparsity_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.sparsity_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.sparsity_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_kurtosis_axis_metric(tensor_3d):
    print("Testing kurtosis_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.kurtosis_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.kurtosis_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_entropy_axis_metric(tensor_3d):
    print("Testing entropy_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.entropy_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.entropy_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

def test_peak_to_peak_axis_metric(tensor_3d):
    print("Testing peak_to_peak_axis_metric")
    print("Before process shape:", tensor_3d.shape)
    result_axis0 = axis_metric.peak_to_peak_axis_metric(tensor_3d, axis=0)
    result_axis_neg1 = axis_metric.peak_to_peak_axis_metric(tensor_3d, axis=-1)
    print("Result axis=0:", result_axis0)
    print("Result axis=-1:", result_axis_neg1)

if __name__ == '__main__':
    # Load numpy array from npz file and convert to PyTorch tensor
    import numpy as np
    data = np.load('single_transformer_blocks.5.attn.to_v.npz')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    tensor_3d = torch.from_numpy(data['arr_0'])
    
    # Plot the distribution with x-axis as the value and y-axis as the frequency
    plt.figure()
    plt.hist(tensor_3d.flatten().numpy(), bins=100, color='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Tensor Values')
    plt.savefig('tensor_distribution.png')

    # Plot the 3D tensor
    depth, out_channels, in_channels = tensor_3d.shape
    _x = np.arange(in_channels)
    _y = np.arange(out_channels)
    _X, _Y = np.meshgrid(_x, _y)

    # Add new figure and subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Remove breakpoint
    ttt = tensor_3d[0, :, :].numpy()
    # 3D surface
    surf = ax.plot_surface(_X, _Y, np.abs(tensor_3d[0, :, :].numpy()), cmap='coolwarm')

    # Set axis labels
    ax.set_xlabel('Input Channels')
    ax.set_ylabel('Output Channels')
    ax.set_zlabel('Absolute Activations')

    ax.set_zlim([0, np.max(np.abs(tensor_3d.numpy()))])

    # Add color bar
    fig.colorbar(surf)
    plt.savefig('tensor_3d.png')
    plt.close()
    
    # plot along two axes with variance metric
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _tmp = tensor_3d[0]
    variance_axis0 = torch.var(_tmp, dim=0).numpy()  # Compute variance along axis 0
    bar_width = 0.5 if np.max(variance_axis0) < 0.5 else 0.7  # Set bar width based on variance value
    ax.bar(np.arange(variance_axis0.shape[0]), variance_axis0, width=bar_width)
    ax.set_ylim([0, 0.5])  # Set max y-axis
    plt.savefig('tensor_3d_variance_axis0.png')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _tmp = tensor_3d[0]
    variance_axis_neg1 = torch.var(_tmp, dim=1).numpy()  # Compute variance along axis -1
    bar_width = 0.5 if np.max(variance_axis_neg1) < 0.5 else 0.7  # Set bar width based on variance value
    ax.bar(np.arange(variance_axis_neg1.shape[0]), variance_axis_neg1, width=bar_width)
    ax.set_ylim([0, 0.5])  # Set max y-axis
    plt.savefig('tensor_3d_variance_axis_neg1.png')
    plt.close()
    
    # plot along two axes with mean metric
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _tmp = tensor_3d[0]
    mean_axis0 = torch.mean(_tmp, dim=0).numpy()  # Compute mean along axis 0
    bar_width = 0.5 if np.max(mean_axis0) < 0.5 else 0.7  # Set bar width based on mean value
    ax.bar(np.arange(mean_axis0.shape[0]), mean_axis0, width=bar_width)
    ax.set_ylim([0, np.max(mean_axis0)])  # Set max y-axis
    plt.savefig('tensor_3d_mean_axis0.png')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _tmp = tensor_3d[0]
    mean_axis_neg1 = torch.mean(_tmp, dim=1).numpy()  # Compute mean along axis -1
    bar_width = 0.5 if np.max(mean_axis_neg1) < 0.5 else 0.7  # Set bar width based on mean value
    ax.bar(np.arange(mean_axis_neg1.shape[0]), mean_axis_neg1, width=bar_width)
    ax.set_ylim([0, np.max(mean_axis_neg1)])  # Set max y-axis
    plt.savefig('tensor_3d_mean_axis_neg1.png')
    plt.close()
    
    
    
    
    # Run all test functions with deepcopy to avoid modifying original tensor
    # from copy import deepcopy
    # breakpoint()
    # test_variance_axis_metric(tensor_3d)
    # test_std_axis_metric(deepcopy(tensor_3d))
    # test_mean_abs_axis_metric(deepcopy(tensor_3d))
    # test_l2_norm_axis_metric(deepcopy(tensor_3d))
    # test_sparsity_axis_metric(deepcopy(tensor_3d))
    # test_kurtosis_axis_metric(deepcopy(tensor_3d))
    # test_entropy_axis_metric(deepcopy(tensor_3d))
    # test_peak_to_peak_axis_metric(deepcopy(tensor_3d))
    # test_kurtosis_axis_metric(tensor_3d)
