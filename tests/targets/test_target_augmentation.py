from __future__ import annotations

import unittest

import numpy as np

from scripts.targets.augmentation import AugmentationConfig, augment


class TargetAugmentationTest(unittest.TestCase):
    def test_augmentation_is_reproducible_and_preserves_shape(self) -> None:
        image = np.full((120, 80), 255, dtype=np.uint8)
        image[30:90, 20:60] = 0
        config = AugmentationConfig(noise_stddev=(0.0, 0.0))

        first, first_meta = augment(image, seed=42, config=config)
        second, second_meta = augment(image, seed=42, config=config)

        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_meta, second_meta)
        self.assertFalse(np.array_equal(first, image))

    def test_noise_can_be_disabled_exactly(self) -> None:
        image = np.full((50, 50), 128, dtype=np.uint8)
        config = AugmentationConfig(
            perspective_px=(0.0, 0.0),
            blur_sigma_px=(0.0, 0.0),
            illumination_strength=(0.0, 0.0),
            noise_stddev=(0.0, 0.0),
        )

        result, metadata = augment(image, seed=1, config=config)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(metadata["sampled"]["noise_stddev"], 0.0)


if __name__ == "__main__":
    unittest.main()
