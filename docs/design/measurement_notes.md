# Measurement Notes

- The A4 target is a known rectangle: 210 mm by 297 mm.
- The black border gives robust plane corners for perspective correction.
- The bottom-center marker defines the target point used by the problem statement for distance `D`.
- After camera calibration, `solvePnP` can estimate the target plane pose from the four A4 corners.
- Shape size `x` should be measured on the rectified A4 plane, where pixel-to-mm scale is fixed by the target paper dimensions.
