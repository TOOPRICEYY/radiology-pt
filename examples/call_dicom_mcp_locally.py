#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call dicom_mcp MCP tools/resources as regular Python functions."
    )
    parser.add_argument(
        "--root",
        default="dicom",
        help="DICOM root directory (default: ./dicom)",
    )
    parser.add_argument(
        "--series-id",
        default=None,
        help="Series ID to inspect. Defaults to first discovered series.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for extraction (inclusive, zero-based).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1,
        help="End index for extraction (inclusive, zero-based).",
    )
    parser.add_argument(
        "--save-png",
        default=None,
        help="Optional path to save the first extracted PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        # Importing server.py requires the MCP SDK because decorators are applied at import time.
        from dicom_mcp import server as dicom_server
    except ImportError as exc:
        raise SystemExit(
            "This example imports `dicom_mcp.server`, which requires the MCP SDK.\n"
            "Install project deps first: `pip install -e .`"
        ) from exc

    print("# set_dicom_root (tool)")
    _json_print(dicom_server.set_dicom_root(args.root))

    print("\n# get_dicom_root (tool)")
    _json_print(dicom_server.get_dicom_root())

    print("\n# list_dicom_series (tool)")
    series_list = dicom_server.list_dicom_series()
    print(f"series_count={len(series_list)}")
    if not series_list:
        raise SystemExit("No series found under the configured root.")
    _json_print(series_list[:2])

    series_id = args.series_id or series_list[0]["series_id"]
    print(f"\nUsing series_id={series_id}")

    print("\n# get_dicom_series_details (tool)")
    details = dicom_server.get_dicom_series_details(series_id)
    print(
        f"instance_count={details['instance_count']}, "
        f"rows={details.get('rows')}, cols={details.get('columns')}"
    )
    _json_print(details["instances"][:3])

    print("\n# list_series_resource (resource)")
    series_resource_json = dicom_server.list_series_resource()
    print(series_resource_json[:300] + ("..." if len(series_resource_json) > 300 else ""))

    print("\n# range_metadata_resource (resource)")
    range_resource_json = dicom_server.range_metadata_resource(
        series_id, args.start, args.end
    )
    print(range_resource_json[:400] + ("..." if len(range_resource_json) > 400 else ""))

    print("\n# annotate_dicom_slice (tool)")
    annotate_result = dicom_server.annotate_dicom_slice(
        series_id="SER00007",
        index=args.start+9,
        shapes=[
            {
                "type": "bbox",
                "x": 16,
                "y": 16,
                "width": 60,
                "height": 40,
                "color": "yellow",
            },
            {
                "type": "circle",
                "cx": 64,
                "cy": 64,
                "radius": 20,
                "color": [0, 255, 0],
            },
            {
                "type": "ellipse",
                "cx": 90,
                "cy": 40,
                "rx": 18,
                "ry": 10,
                "color": "#00AAFF",
            },
        ],
        include_png_base64=True,
        include_pixels=False,
        include_dicom_bytes_base64=False,
    )
    print(
        f"annotated_extracted_count={annotate_result['extracted_count']}, "
        f"applied_annotations={annotate_result['instances'][0].get('render', {}).get('applied_annotations')}"
    )

    if args.save_png:
        first_png_b64 = annotate_result["instances"][0].get("png_base64")
        if first_png_b64:
            out_path = Path(args.save_png).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(base64.b64decode(first_png_b64))
            print(f"Saved annotated PNG to {out_path}")


if __name__ == "__main__":
    main()
