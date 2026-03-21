import unittest

import torch

from airllm.device_utils import get_xpu_device, is_xpu_available


@unittest.skipUnless(is_xpu_available(), "XPU not available — skipping Intel XPU tests")
class TestXpuTensorOps(unittest.TestCase):
    """
    Validates basic PyTorch tensor operations on Intel XPU when an
    XPU-enabled PyTorch runtime is installed on Linux.
    """

    def setUp(self):
        self.device = get_xpu_device(0)

    def test_device_string(self):
        self.assertEqual(str(self.device), "xpu:0")

    def test_tensor_creation_on_device(self):
        t = torch.randn(64, 64).to(self.device)
        self.assertEqual(str(t.device), "xpu:0")

    def test_matmul(self):
        a = torch.randn(256, 256).to(self.device)
        b = torch.randn(256, 256).to(self.device)
        c = torch.matmul(a, b)
        self.assertEqual(c.shape, torch.Size([256, 256]))
        self.assertEqual(str(c.device), "xpu:0")

    def test_float16_matmul(self):
        a = torch.randn(128, 128, dtype=torch.float16).to(self.device)
        b = torch.randn(128, 128, dtype=torch.float16).to(self.device)
        c = torch.matmul(a, b)
        self.assertEqual(c.dtype, torch.float16)

    def test_layer_cycle(self):
        weight = torch.randn(128, 128, dtype=torch.float16)
        x = torch.randn(1, 128, dtype=torch.float16)

        weight_xpu = weight.to(self.device)
        x_xpu = x.to(self.device)
        out = x_xpu @ weight_xpu.T

        self.assertEqual(str(out.device), "xpu:0")
        self.assertEqual(out.shape, torch.Size([1, 128]))

        del weight_xpu
        result = out.cpu()
        self.assertEqual(result.device.type, "cpu")