from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.targets.augmentation.low_resolution import (
    LowResolutionConfig,
    simulate_low_resolution_a4,
)
from scripts.targets.generation.renderer import render
from scripts.targets.generation.shapes import ShapeSpec
from scripts.targets.presets.numbered import PRESETS as NUMBERED_PRESETS
from scripts.targets.presets.overlap import PRESETS
from src.vision.lowres_preprocess import (
    WORK_HEIGHT_PX,
    WORK_WIDTH_PX,
    prepare_low_resolution_a4,
)
from src.vision.overlap.lowres import reconstruct_low_resolution_squares


class LowResolutionOverlapTest(unittest.TestCase):
    def test_simulation_is_reproducible(self) -> None:
        source, _ = render(PRESETS["01"])
        config = LowResolutionConfig(
            native_width_px=82,
            phase_x_px=0.25,
            phase_y_px=-0.25,
        )

        first, first_metadata = simulate_low_resolution_a4(source, config, seed=42)
        second, second_metadata = simulate_low_resolution_a4(source, config, seed=42)

        self.assertEqual(first.shape, (WORK_HEIGHT_PX, WORK_WIDTH_PX))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_metadata, second_metadata)

    def test_black_probability_uses_page_references(self) -> None:
        image = np.full((WORK_HEIGHT_PX, WORK_WIDTH_PX), 180, dtype=np.uint8)
        border = 40
        image[:border, :] = 20
        image[-border:, :] = 20
        image[:, :border] = 20
        image[:, -border:] = 20
        cv2.rectangle(image, (150, 220), (270, 340), 25, -1)

        prepared = prepare_low_resolution_a4([image, image, image])

        self.assertAlmostEqual(prepared.black_level, 20.0, delta=1.0)
        self.assertAlmostEqual(prepared.white_level, 180.0, delta=1.0)
        self.assertGreater(float(prepared.black_probability[250, 200]), 0.9)
        self.assertLess(float(prepared.black_probability[100, 200]), 0.1)

    def test_four_visible_edges_are_reconstructed_at_82_pixel_a4_width(self) -> None:
        source, _ = render(PRESETS["03"])
        frame, _ = simulate_low_resolution_a4(
            source,
            LowResolutionConfig(native_width_px=82),
            seed=1,
        )

        result = reconstruct_low_resolution_squares([frame])
        actual = sorted(square.side_mm for square in result.squares)
        expected = sorted(square.size_mm for square in PRESETS["03"])

        self.assertEqual(len(actual), len(expected))
        for measured, target in zip(actual, expected):
            self.assertAlmostEqual(measured, target, delta=3.0)

    def test_digit_lines_do_not_create_an_extra_square(self) -> None:
        source, _ = render(NUMBERED_PRESETS["03"])
        frames = [
            simulate_low_resolution_a4(
                source,
                LowResolutionConfig(native_width_px=130, phase_x_px=phase),
                seed=index,
            )[0]
            for index, phase in enumerate((-0.4, 0.0, 0.4))
        ]

        result = reconstruct_low_resolution_squares(frames)

        self.assertEqual(len(result.squares), 2)

    def test_opposite_short_fragments_are_joined_before_square_generation(self) -> None:
        squares = (
            ShapeSpec("square", 76.67, 138.75, 67.69, -15.47, 0),
            ShapeSpec("square", 137.69, 160.87, 61.02, 14.62, 1),
        )
        source, _ = render(squares)
        frames = [
            simulate_low_resolution_a4(
                source,
                LowResolutionConfig(
                    native_width_px=82,
                    phase_x_px=phase,
                    phase_y_px=-phase,
                ),
                seed=index + 20,
            )[0]
            for index, phase in enumerate((-0.35, 0.0, 0.35))
        ]

        result = reconstruct_low_resolution_squares(frames)

        self.assertEqual(len(result.squares), 2)
        self.assertEqual(sorted(round(item.side_mm) for item in result.squares), [61, 68])


if __name__ == "__main__":
    unittest.main()
