# Architecture

The project is organized around the contest device:

1. `src/vision` captures one image, finds the A4 target plane, estimates pose, and measures the black geometry on the rectified paper plane.
2. `src/hardware` owns Raspberry Pi GPIO, key input, current sensing, and power calculation.
3. `src/display` owns the result presentation layer.
4. `src/main.py` will connect the one-key measurement workflow.

Generated target sheets are kept under `assets/targets/generated` so they can be printed for local debugging without mixing them with runtime captures.
