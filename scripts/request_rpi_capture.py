from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_RPI_MOUNT = Path(r"\\wsl.localhost\Ubuntu\home\mizuku\nailong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request a capture from the Raspberry Pi preview process.")
    parser.add_argument(
        "name",
        nargs="?",
        default="",
        help="Optional image file name. Omit it to use an automatic timestamp name.",
    )
    parser.add_argument("--mount", type=Path, default=DEFAULT_RPI_MOUNT)
    parser.add_argument(
        "--request-file",
        type=Path,
        default=Path("runtime/capture_request.txt"),
        help="Request file path under --mount, or an absolute path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_file = args.request_file if args.request_file.is_absolute() else args.mount / args.request_file
    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text(args.name.strip(), encoding="utf-8")
    print(f"capture requested: {args.name or '<automatic timestamp>'}")
    print(f"request file: {request_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
