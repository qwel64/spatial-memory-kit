"""Behavioral controls for the released model, independent of training metrics."""
from pathlib import Path
import unittest
import tempfile
import numpy as np
import torch
from spatial_memory_kit.model_io import load_model
from spatial_memory_kit.replay import infer, load_trace, observations, replay
from spatial_memory_kit.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(0)
        cls.model, _ = load_model(ROOT, "coupled_b7_s33101")
        cls.trace, _ = load_trace(ROOT, "coupled_b7_s33101", "a_two")

    def test_future_targets_and_display_labels_do_not_enter_belief(self):
        before = infer(self.model, self.trace)["h"]
        altered = {k: v.copy() for k, v in self.trace.items()}
        for key in ("truth_image", "truth_image_h5", "display_cells", "display_heading", "display_targets"):
            altered[key][...] = 123
        altered["body"][49:] = 123
        np.testing.assert_array_equal(before, infer(self.model, altered)["h"])

    def test_prefix_has_no_future_observations(self):
        image, body, action = observations(self.trace)
        with torch.no_grad():
            full, _ = self.model.beliefs(image, body, action)
            for length in (1, 8, 25, 49):
                prefix, _ = self.model.beliefs(image[:, :length], body[:, :length], action[:, :length])
                torch.testing.assert_close(prefix, full[:, :length], rtol=0, atol=1e-5)

    def test_discarded_episode_start_prior_does_not_change_first_belief(self):
        image, body, action = observations(self.trace)
        self.assertEqual(int(torch.count_nonzero(action[:, :1])), 0)
        with torch.no_grad():
            coupled, _ = self.model.beliefs(image[:, :1], body[:, :1], action[:, :1])
            self.model.observation_update = "separate"
            try:
                separate, _ = self.model.beliefs(image[:, :1], body[:, :1], action[:, :1])
            finally:
                self.model.observation_update = "coupled"
        torch.testing.assert_close(coupled, separate, rtol=0, atol=0)

    def test_window_retains_real_preceding_action_and_has_no_older_memory(self):
        image, body, action = observations(self.trace)
        reset = infer(self.model, self.trace, "reset8")["h"][-1]
        with torch.no_grad():
            direct, _ = self.model.beliefs(image[:, -8:], body[:, -8:], action[:, -8:])
        np.testing.assert_allclose(reset, direct[0, -1].numpy(), rtol=0, atol=1e-5)

    def test_output_cannot_replace_an_existing_npz(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "saved.npz"
            path.write_bytes(b"keep existing file")
            with self.assertRaises(FileExistsError):
                replay(ROOT, output=path)
            with self.assertRaises(ValueError):
                replay(ROOT, output=path.with_suffix(""))
            self.assertEqual(path.read_bytes(), b"keep existing file")


if __name__ == "__main__":
    unittest.main()
