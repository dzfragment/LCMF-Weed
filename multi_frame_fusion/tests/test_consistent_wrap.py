"""Regression tests for circular-displacement unwrapping in ConsistentAligner."""

import unittest

from fft_align.consistent import ConsistentAligner


class ConsistentAlignerWrapTests(unittest.TestCase):
    def solve(self, c10, c21, c20, image_hw=(1024, 1024)):
        return ConsistentAligner._solve_from_candidates(
            c10, c21, c20, image_hw=image_hw)

    def test_no_wrap_keeps_direct_candidate(self):
        result = self.solve(
            [(-100, -200, 0.9)],
            [(-80, -180, 0.8)],
            [(-180, -380, 0.85)],
        )
        self.assertEqual(result[:4], (-100, -200, -180, -380))
        self.assertEqual(result[5], "top1")

    def test_vertical_negative_shift_beyond_half_is_unwrapped(self):
        result = self.solve(
            [(-12, -264, 0.9)],
            [(-16, -260, 0.85)],
            [(-28, 500, 0.95)],
        )
        self.assertEqual(result[:4], (-12, -264, -28, -524))
        self.assertAlmostEqual(result[4], 0.0)
        self.assertEqual(result[5], "top1_unwrapped")

    def test_vertical_700px_total_shift_is_unwrapped(self):
        result = self.solve(
            [(-20, -348, 0.9)],
            [(-16, -352, 0.88)],
            [(-36, 324, 0.92)],
        )
        self.assertEqual(result[:4], (-20, -348, -36, -700))
        self.assertEqual(result[5], "top1_unwrapped")

    def test_axes_use_independent_periods_for_non_square_image(self):
        # W=1280 and H=720.  The direct peak wraps independently on both axes.
        result = self.solve(
            [(-350, -210, 0.9)],
            [(-350, -210, 0.85)],
            [(580, 300, 0.95)],
            image_hw=(720, 1280),
        )
        self.assertEqual(result[:4], (-350, -210, -700, -420))
        self.assertEqual(result[5], "top1_unwrapped")

    def test_unbalanced_steps_do_not_enable_wrap(self):
        result = self.solve(
            [(-20, -500, 0.95)],
            [(-16, -40, 0.90)],
            [(-36, 484, 0.93)],
        )
        self.assertNotIn("unwrapped", result[5])
        self.assertGreater(result[3], 0)

    def test_opposite_direction_steps_do_not_enable_wrap(self):
        result = self.solve(
            [(-20, -400, 0.95)],
            [(-16, 940, 0.90)],
            [(-36, -484, 0.93)],
        )
        self.assertNotIn("unwrapped", result[5])
        self.assertGreater(result[3], -512)

    def test_incoherent_2d_steps_do_not_enable_wrap(self):
        result = self.solve(
            [(-300, -300, 0.95)],
            [(300, -300, 0.90)],
            [(0, 424, 0.93)],
        )
        self.assertNotIn("unwrapped", result[5])
        self.assertGreater(result[3], -512)

    def test_secondary_candidate_cannot_enable_wrap_inside_half_range(self):
        # Top1 chain is safely inside half-width. A secondary c10 happens to
        # cross the boundary, but must not switch the sequence to modulo mode.
        result = self.solve(
            [(-100, -100, 0.95), (-400, -400, 0.90)],
            [(-100, -100, 0.90)],
            [(-200, -200, 0.93), (524, 524, 0.92)],
        )
        self.assertEqual(result[:4], (-100, -100, -200, -200))
        self.assertEqual(result[5], "top1")


if __name__ == "__main__":
    unittest.main()
