import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

from PIL import Image

try:
    import torch as TEST_TORCH
except ImportError:
    TEST_TORCH = None

from engine.runtime_profile import PROFILES, resolve_runtime_profile
from models.judge_model import QwenJudgeModel
from models.vision_model import Qwen3VLVisionModel


class RuntimeProfileTests(unittest.TestCase):
    def test_prefill_cleanup_runs_once_and_preserves_forward_output(self):
        runtime = QwenJudgeModel.__new__(QwenJudgeModel)
        runtime.hardware_profile = PROFILES["paper-8gb"]
        runtime.model = Mock()
        cuda = Mock()
        cuda.is_available.return_value = True
        fake_torch = SimpleNamespace(cuda=cuda)
        with patch.object(runtime, "_cuda_memory", return_value={"allocated_gb": 3.2}):
            handle, measurement = runtime._install_prefill_cleanup(fake_torch, True)
            callback = runtime.model.register_forward_hook.call_args.args[0]
            output = {"past_key_values": object(), "logits": object()}
            retained = dict(output)
            self.assertIsNone(callback(runtime.model, (), output))
            self.assertIsNone(callback(runtime.model, (), output))
            self.assertEqual(output, retained)
        cuda.empty_cache.assert_called_once()
        cuda.synchronize.assert_called_once()
        self.assertTrue(measurement["succeeded"])
        self.assertIs(handle, runtime.model.register_forward_hook.return_value)

    def test_cache_free_fallback_does_not_install_prefill_cleanup(self):
        runtime = QwenJudgeModel.__new__(QwenJudgeModel)
        runtime.hardware_profile = PROFILES["paper-8gb"]
        runtime.model = Mock()
        handle, measurement = runtime._install_prefill_cleanup(SimpleNamespace(cuda=Mock()), False)
        self.assertIsNone(handle)
        self.assertFalse(measurement["performed"])
        runtime.model.register_forward_hook.assert_not_called()

    def test_auto_profile_is_conservative_and_deterministic(self):
        self.assertEqual(resolve_runtime_profile("auto", 7.9).name, "8gb")
        self.assertEqual(resolve_runtime_profile("auto", 11.9).name, "12gb")
        self.assertEqual(resolve_runtime_profile("auto", 15.5).name, "16gb")

    def test_profiles_raise_detail_monotonically(self):
        self.assertLess(PROFILES["8gb"].agent1_max_pixels, PROFILES["12gb"].agent1_max_pixels)
        self.assertLess(PROFILES["12gb"].agent1_max_pixels, PROFILES["16gb"].agent1_max_pixels)

    def test_paper_profile_locks_conservative_inputs(self):
        paper = resolve_runtime_profile("paper-8gb", 16.0)
        base = resolve_runtime_profile("8gb", 8.0)
        self.assertEqual(paper.agent1_max_pixels, base.agent1_max_pixels)
        self.assertEqual(paper.judge_max_pixels, base.judge_max_pixels)
        self.assertEqual(paper.judge_output_tokens, 1024)
        self.assertEqual(paper.judge_max_seconds, 120.0)
        self.assertEqual(paper.oom_retry_pixels, ())

    def test_fit_never_upscales(self):
        class Image:
            size = (640, 480)

        image = Image()
        prepared, changed = Qwen3VLVisionModel._fit_image_to_pixel_budget(
            image, PROFILES["16gb"].agent1_max_pixels
        )
        self.assertIs(prepared, image)
        self.assertFalse(changed)

    @unittest.skipIf(TEST_TORCH is None, "PyTorch is unavailable")
    def test_8gb_judge_starts_cached_then_uses_bounded_fallbacks(self):
        runtime = QwenJudgeModel.__new__(QwenJudgeModel)
        runtime.hardware_profile = PROFILES["8gb"]
        runtime._last_generation_diagnostics = {}
        calls = []

        def generate_once(image, _prompt, _tokens, *, use_cache):
            calls.append((image.size, use_cache))
            if len(calls) <= 2:
                raise TEST_TORCH.OutOfMemoryError("simulated")
            return "{}", 0.1, {
                "image_size": list(image.size),
                "generated_tokens": 1,
                "max_new_tokens": 384,
            }

        source = Image.new("RGB", (2000, 1000), "white")
        with (
            patch.object(runtime, "_generate_once", side_effect=generate_once),
            patch.object(runtime, "_release_generation_memory"),
            patch.object(runtime, "_cuda_memory", return_value={}),
        ):
            answer, _ = runtime.generate(source, "Review", 384)

        self.assertEqual(answer, "{}")
        self.assertTrue(calls[0][1])
        self.assertFalse(calls[1][1])
        self.assertEqual(calls[1][0], calls[0][0])
        self.assertLess(calls[2][0][0] * calls[2][0][1], calls[0][0][0] * calls[0][0][1])
        self.assertTrue(runtime._last_generation_diagnostics["oom_recovery_used"])
        self.assertTrue(runtime._last_generation_diagnostics["resolution_reduced"])


if __name__ == "__main__":
    unittest.main()
