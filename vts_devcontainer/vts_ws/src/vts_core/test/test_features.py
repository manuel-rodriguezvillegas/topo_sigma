"""Feature extraction contracts without downloading pretrained weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from vts_core.features import (
    DINOV3_MODEL,
    DINOV3_REVISION,
    FrozenDinoV3Extractor,
    build_extractor,
    generalized_mean,
    resolve_torch_device,
)


class DeviceSelectionTest(unittest.TestCase):
    @patch("vts_core.features.torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, _cuda_available: object) -> None:
        self.assertEqual(resolve_torch_device("auto").type, "cuda")

    @patch("vts_core.features.torch.cuda.is_available", return_value=False)
    @patch("vts_core.features.torch.backends.mps.is_available", return_value=True)
    def test_auto_uses_mps_without_cuda(
        self, _mps_available: object, _cuda_available: object
    ) -> None:
        self.assertEqual(resolve_torch_device("auto").type, "mps")

    @patch("vts_core.features.torch.cuda.is_available", return_value=False)
    @patch("vts_core.features.torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(
        self, _mps_available: object, _cuda_available: object
    ) -> None:
        self.assertEqual(resolve_torch_device("auto").type, "cpu")

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported DINO device"):
            resolve_torch_device("tpu")

    @patch("vts_core.features.torch.cuda.is_available", return_value=False)
    def test_unavailable_explicit_cuda_is_rejected(
        self, _cuda_available: object
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            resolve_torch_device("cuda")

    def test_generalized_mean_preserves_signed_features(self) -> None:
        features = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
        pooled = generalized_mean(features, power=3.0)
        torch.testing.assert_close(pooled, torch.tensor([[-1.0, 1.0]]))

    def test_generalized_mean_rejects_empty_patch_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            generalized_mean(torch.empty(1, 0, 4))

    def test_anyloc_requires_explicit_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit"):
            build_extractor(backend="anyloc_gem")

    def test_unknown_feature_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown feature backend"):
            build_extractor(backend="unknown")


class DinoV3Test(unittest.TestCase):
    def test_default_uses_pinned_processor_cls_and_checkpoint_identity(self) -> None:
        processor = Mock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
        processor.to_dict.return_value = {"size": {"height": 224, "width": 224}}
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config.hidden_size = 384
        cls = torch.zeros(1, 384)
        cls[0, :2] = torch.tensor([3.0, 4.0])
        model.return_value = SimpleNamespace(pooler_output=cls)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "transformers.AutoImageProcessor.from_pretrained", return_value=processor
            ) as load_processor,
            patch("transformers.AutoModel.from_pretrained", return_value=model) as load_model,
            patch("huggingface_hub.hf_hub_download") as download,
        ):
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"test-checkpoint")
            download.return_value = str(checkpoint)
            extractor = build_extractor(device="cpu", cache_dir=directory)
            bgr = np.full((4, 4, 3), [10, 20, 30], dtype=np.uint8)
            vector = extractor.extract(bgr)
            np.testing.assert_allclose(vector[:2], [0.6, 0.8])
            self.assertEqual(vector.shape, (384,))
            self.assertEqual(vector.dtype, np.float32)
            self.assertEqual(processor.call_args.kwargs["images"].getpixel((0, 0)), (30, 20, 10))
            for loader in (load_processor, load_model):
                loader.assert_called_once_with(DINOV3_MODEL, revision=DINOV3_REVISION)
            self.assertEqual(extractor.model_identity["revision"], DINOV3_REVISION)
            self.assertEqual(len(extractor.model_identity["checkpoint_sha256"]), 64)
            np.testing.assert_array_equal(extractor.extract(bgr), vector)
            model.assert_called_once()
            self.assertEqual(extractor.hits, 1)

    def test_access_error_explains_authentication(self) -> None:
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=OSError):
            with self.assertRaisesRegex(RuntimeError, "hf auth login"):
                FrozenDinoV3Extractor(device="cpu")

    def test_dinov3_cannot_silently_replace_anyloc_value_facets(self) -> None:
        with self.assertRaisesRegex(ValueError, "DINOv2 backbone"):
            build_extractor(backend="anyloc_gem", layer=11)


if __name__ == "__main__":
    unittest.main()
