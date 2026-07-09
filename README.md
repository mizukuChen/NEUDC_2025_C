# NUEDC 2025 C - Monocular Target Measurement

Raspberry Pi project for the 2025 NUEDC C problem, "基于单目视觉的目标物测量装置".

## Directory Layout

- `src/`: Raspberry Pi runtime code.
- `src/vision/`: image processing, camera calibration, pose estimation, and target measurement modules.
- `src/hardware/`: button, current-sense ADC, and power measurement drivers.
- `src/display/`: screen output adapters.
- `config/`: local runtime configuration examples.
- `scripts/targets/`: scripts that generate A4 debugging targets.
- `assets/targets/generated/`: generated PDF/PNG target sheets.
- `docs/problem/`: original contest problem statement.
- `docs/design/`: design notes and measurement theory.
- `runtime/`: local captures, calibration files, and logs created while running on Raspberry Pi.
- `tests/`: unit and image-processing regression tests.

## Quick Start

```powershell
python -m pip install -r requirements-dev.txt
python scripts/targets/generate_basic_targets.py
python scripts/targets/generate_advanced_targets.py
python scripts/targets/merge_target_pdfs.py
```

On Raspberry Pi:

```bash
python3 -m pip install -r requirements-rpi.txt
python3 src/camera_test.py
```

## Implementation Roadmap

1. Camera capture and calibration.
2. A4 black border and marker detection.
3. Monocular pose estimation for distance `D`.
4. Perspective correction of the target plane.
5. Shape and square-size measurement for `x`.
6. Button, display, current, and power integration.
