import unittest
import torch
from optimum.quanto.tensor import axis_metric

class TestAxisMetric(unittest.TestCase):
    def setUp(self):
        # Create test tensor
        self.tensor_3d = torch.randn(1, 4096, 11081)
        print("tensor_3d shape:", self.tensor_3d.shape)  # [1, 4096, 11081]

    def test_variance_axis_metric(self):
        print("Testing variance_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.variance_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.variance_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_std_axis_metric(self):
        print("Testing std_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.std_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.std_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_mean_abs_axis_metric(self):
        print("Testing mean_abs_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.mean_abs_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.mean_abs_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_l2_norm_axis_metric(self):
        print("Testing l2_norm_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.l2_norm_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.l2_norm_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_sparsity_axis_metric(self):
        print("Testing sparsity_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.sparsity_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.sparsity_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_kurtosis_axis_metric(self):
        print("Testing kurtosis_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.kurtosis_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.kurtosis_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_entropy_axis_metric(self):
        print("Testing entropy_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.entropy_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.entropy_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

    def test_peak_to_peak_axis_metric(self):
        print("Testing peak_to_peak_axis_metric")
        print("Before process shape:", self.tensor_3d.shape)
        result_axis0 = axis_metric.peak_to_peak_axis_metric(self.tensor_3d, axis=0)
        result_axis_neg1 = axis_metric.peak_to_peak_axis_metric(self.tensor_3d, axis=-1)
        print("Result axis=0:", result_axis0)
        print("Result axis=-1:", result_axis_neg1)

if __name__ == '__main__':
    unittest.main()
