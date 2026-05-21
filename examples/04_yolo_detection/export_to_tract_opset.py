#!/usr/bin/env python3
"""Convert a YOLOv8 (or any) ONNX export to an opset that tract-onnx accepts.

The Ultralytics releases on GitHub ship `yolov8n.onnx` at opset 9, which
emits the deprecated `Upsample` op. tract-onnx 0.21 doesn't implement
`Upsample` and fails to load the graph. Bumping the opset to >= 10 makes
the converter rewrite every `Upsample` as `Resize`, which tract handles
fine.

This is generic ONNX surgery — no torch / no ultralytics / no GPU
required. The only dependency is the `onnx` PyPI package (~7 MB).

Usage:
    python3 export_to_tract_opset.py input.onnx [output.onnx]

Defaults to writing alongside the input with an `_opset10` suffix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import onnx
    from onnx import version_converter
except ImportError:
    sys.exit(
        "onnx package not found. Install with:\n"
        "    python3 -m venv /tmp/onnx-venv\n"
        "    /tmp/onnx-venv/bin/pip install onnx\n"
        "Then re-run with /tmp/onnx-venv/bin/python."
    )


def detect_bad_ops(model: onnx.ModelProto) -> set[str]:
    """Return the set of op_types tract-onnx does not implement (as of 0.21)."""
    return {
        n.op_type
        for n in model.graph.node
        if n.op_type in {"Upsample"}
    }


def target_opset_for(bad_ops: set[str]) -> int:
    """The minimum opset that rewrites every known bad op."""
    if "Upsample" in bad_ops:
        return 10
    return max(opset.version for opset in [])  # placeholder


def convert(input_path: Path, output_path: Path, target_opset: int | None) -> None:
    model = onnx.load(str(input_path))

    current = next(
        (op.version for op in model.opset_import if op.domain in ("", "ai.onnx")),
        None,
    )
    bad = detect_bad_ops(model)
    print(f"input opset: {current}")
    print(f"unsupported ops detected: {sorted(bad) or 'none'}")

    if target_opset is None:
        target_opset = target_opset_for(bad) if bad else (current or 10)

    if current is not None and current >= target_opset and not bad:
        print(f"nothing to do — already at opset {current}, no bad ops")
        onnx.save(model, str(output_path))
        return

    print(f"converting to opset {target_opset}...")
    converted = version_converter.convert_version(model, target_opset)

    remaining_bad = detect_bad_ops(converted)
    if remaining_bad:
        sys.exit(
            f"FAILED: bad ops still present after conversion: {sorted(remaining_bad)}.\n"
            "Try a higher --opset."
        )

    onnx.checker.check_model(converted)
    onnx.save(converted, str(output_path))

    out_ops = {n.op_type for n in converted.graph.node}
    print(f"wrote {output_path}")
    print(f"output opset: {target_opset}")
    print(f"ops present: {sorted(out_ops)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="path to source .onnx file")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="path for converted .onnx (default: alongside input with _opset10 suffix)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=None,
        help="target opset version (auto-picked from detected bad ops if omitted)",
    )
    args = p.parse_args()

    if args.output is None:
        target = args.opset or 10
        args.output = args.input.with_name(f"{args.input.stem}_opset{target}.onnx")

    convert(args.input, args.output, args.opset)


if __name__ == "__main__":
    main()
