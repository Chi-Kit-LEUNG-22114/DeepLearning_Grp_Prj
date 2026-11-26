from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brand_momentum_transfer.models.common import ModalityAdapters


class AdapterTests(unittest.TestCase):
    def test_single_channel_keeps_signal_and_input_gradients(self):
        adapter = ModalityAdapters({"text": [0, 1]}, 1, dropout=0).eval()
        with torch.no_grad():
            adapter.adapters["text"][0].weight.fill_(1)
            adapter.adapters["text"][0].bias.zero_()
        values = torch.tensor([[[0., 0.], [1., 2.]]], requires_grad=True)
        result = adapter(values)
        self.assertNotEqual(result[0, 0, 0].item(), result[0, 1, 0].item())
        result.square().sum().backward()
        self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.adapters["text"][0].weight.grad.abs().sum().item(), 0)

    def test_adapter_state_reload_preserves_outputs(self):
        torch.manual_seed(42)
        first = ModalityAdapters({"text": [0, 1], "image": [2]}, 1, dropout=0).eval()
        second = ModalityAdapters({"text": [0, 1], "image": [2]}, 1, dropout=0).eval()
        second.load_state_dict(first.state_dict())
        values = torch.randn(2, 4, 3)
        torch.testing.assert_close(first(values), second(values))

    def test_invalid_width_is_rejected(self):
        with self.assertRaises(ValueError):
            ModalityAdapters({"text": [0]}, 0)


if __name__ == "__main__":
    unittest.main()
