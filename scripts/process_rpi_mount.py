from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.measure_basic import draw_basic_measurement_debug, measure_basic_shape
from src.vision.rectify import rectify_a4
from src.vision.target_plane import draw_target_plane_debug, find_target_plane


DEFAULT_RPI_MOUNT = Path(r"\\wsl.localhost\Ubuntu\home\mizuku\nailong")
DEFAULT_REMOTE_CAPTURE_DIRS = (
    Path("runtime/captures/rpi"),
    Path("captures"),
    Path("."),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ProcessResult:
    source: Path
    local: Path
    shape_type: str
    x_mm: float
    debug_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import images from the Raspberry Pi mount path and run basic measurement.",
    )
    parser.add_argument("--mount", type=Path, default=DEFAULT_RPI_MOUNT, help="Mounted Raspberry Pi project/root path.")
    parser.add_argument(
        "--remote-dir",
        type=Path,
        default=None,
        help="Image directory under --mount. If omitted, common capture folders are searched.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=ROOT / "runtime" / "captures" / "rpi",
        help="Local directory where imported images are copied.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=ROOT / "runtime" / "debug",
        help="Directory for debug output.",
    )
    parser.add_argument("--latest", action="store_true", help="Process only the newest image.")
    parser.add_argument("--no-copy", action="store_true", help="Process images in-place on the mount.")
    return parser.parse_args()


def resolve_remote_dir(mount: Path, remote_dir: Path | None) -> Path:
    if remote_dir is not None:
        candidate = remote_dir if remote_dir.is_absolute() else mount / remote_dir
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"remote image directory does not exist: {candidate}")

    for relative in DEFAULT_REMOTE_CAPTURE_DIRS:
        candidate = mount / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no capture directory found under mount: {mount}")


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.stat().st_mtime,
    )


def copy_image(source: Path, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def process_image(image_path: Path, debug_root: Path) -> ProcessResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")

    plane = find_target_plane(image)
    rectified = rectify_a4(image, plane.corners)
    measurement = measure_basic_shape(rectified)

    debug_dir = debug_root / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "01_plane_corners.jpg"), draw_target_plane_debug(image, plane))
    cv2.imwrite(str(debug_dir / "02_rectified.jpg"), rectified)
    cv2.imwrite(str(debug_dir / "03_measurement.jpg"), draw_basic_measurement_debug(rectified, measurement))

    return ProcessResult(
        source=image_path,
        local=image_path,
        shape_type=measurement.shape_type,
        x_mm=measurement.x_mm,
        debug_dir=debug_dir,
    )


def main() -> int:
    args = parse_args()
    if not args.mount.exists():
        print(f"ERROR: mount path is not available: {args.mount}", file=sys.stderr)
        return 2

    remote_dir = resolve_remote_dir(args.mount, args.remote_dir)
    images = iter_images(remote_dir)
    if not images:
        print(f"ERROR: no images found in: {remote_dir}", file=sys.stderr)
        return 3
    if args.latest:
        images = [images[-1]]

    failures = 0
    for source in images:
        try:
            image_path = source if args.no_copy else copy_image(source, args.local_dir)
            result = process_image(image_path, args.debug_dir)
            print(
                f"PASS {source.name:32s} shape={result.shape_type:8s} "
                f"x={result.x_mm:6.1f} mm debug={result.debug_dir}"
            )
        except Exception as exc:
            failures += 1
            print(f"FAIL {source.name:32s} error={exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
