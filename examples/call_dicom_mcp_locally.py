#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from typing import Any


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call dicom_mcp through a local MCP stdio client."
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


def _snippet(text: str, limit: int) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _parse_tool_result(result: Any) -> Any:
    if getattr(result, "isError", False):
        text = _tool_result_text(result)
        raise RuntimeError(text or "MCP tool call failed")

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    text = _tool_result_text(result)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


def _tool_result_text(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _read_resource_text(result: Any) -> str:
    for item in getattr(result, "contents", []):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            return text
    raise RuntimeError("Resource response did not contain text content")


async def _async_main(args: argparse.Namespace) -> None:
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    try:
        import anyio
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as exc:
        raise SystemExit(
            "This example requires the MCP SDK and its dependencies.\n"
            "Install project deps first: `pip install -e .`"
        ) from exc

    repo_root = Path(__file__).resolve().parents[1]
    root_dir = str(Path(args.root).expanduser().resolve())

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "dicom_mcp.server", "--root", root_dir],
        cwd=str(repo_root),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print(
                f"# initialized MCP server: {init.serverInfo.name} "
                f"(version={init.serverInfo.version})"
            )

            tools = await session.list_tools()
            print(f"\n# list_tools (MCP)\ntool_count={len(tools.tools)}")
            _json_print([tool.name for tool in tools.tools])

            resources = await session.list_resources()
            print(f"\n# list_resources (MCP)\nresource_count={len(resources.resources)}")
            _json_print([str(resource.uri) for resource in resources.resources])

            templates = await session.list_resource_templates()
            print(
                f"\n# list_resource_templates (MCP)\n"
                f"template_count={len(templates.resourceTemplates)}"
            )
            _json_print(
                [
                    str(template.uriTemplate)
                    for template in templates.resourceTemplates
                ]
            )

            print("\n# set_dicom_root (tool)")
            set_root_result = await session.call_tool(
                "set_dicom_root", {"root_dir": root_dir}
            )
            _json_print(_parse_tool_result(set_root_result))

            print("\n# get_dicom_root (tool)")
            get_root_result = await session.call_tool("get_dicom_root")
            _json_print(_parse_tool_result(get_root_result))

            print("\n# read_resource dicom://config")
            config_result = await session.read_resource("dicom://config")
            print(_snippet(_read_resource_text(config_result), 300))

            print("\n# list_dicom_series (tool)")
            series_list_result = await session.call_tool("list_dicom_series")
            series_list = _parse_tool_result(series_list_result)
            if not isinstance(series_list, list):
                raise RuntimeError("Expected list_dicom_series to return a list")
            print(f"series_count={len(series_list)}")
            if not series_list:
                raise SystemExit("No series found under the configured root.")
            _json_print(series_list[:2])

            series_id = args.series_id or series_list[0]["series_id"]
            print(f"\nUsing series_id={series_id}")

            print("\n# get_dicom_series_details (tool)")
            details_result = await session.call_tool(
                "get_dicom_series_details", {"series_id": series_id}
            )
            details = _parse_tool_result(details_result)
            if not isinstance(details, dict):
                raise RuntimeError("Expected get_dicom_series_details to return an object")
            print(
                f"instance_count={details['instance_count']}, "
                f"rows={details.get('rows')}, cols={details.get('columns')}"
            )
            _json_print(details["instances"][:3])

            max_index = max(int(details["instance_count"]) - 1, 0)
            start = min(max(args.start, 0), max_index)
            end = min(max(args.end, start), max_index)
            if start != args.start or end != args.end:
                print(f"Adjusted range to start={start}, end={end} for available slices")

            print("\n# read_resource dicom://series")
            series_resource = await session.read_resource("dicom://series")
            print(_snippet(_read_resource_text(series_resource), 300))

            print("\n# read_resource dicom://series/{series_id}/range/{start}/{end}")
            range_uri = f"dicom://series/{series_id}/range/{start}/{end}"
            range_resource = await session.read_resource(range_uri)
            print(_snippet(_read_resource_text(range_resource), 400))

            print("\n# annotate_dicom_slice (tool)")
            annotate_result_raw = await session.call_tool(
                "annotate_dicom_slice",
                {
                    "series_id": series_id,
                    "index": start,
                    "shapes": [
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
                    "include_png_base64": True,
                    "include_pixels": False,
                    "include_dicom_bytes_base64": False,
                },
            )
            annotate_result = _parse_tool_result(annotate_result_raw)
            if not isinstance(annotate_result, dict):
                raise RuntimeError("Expected annotate_dicom_slice to return an object")
            print(
                f"annotated_extracted_count={annotate_result['extracted_count']}, "
                f"applied_annotations="
                f"{annotate_result['instances'][0].get('render', {}).get('applied_annotations')}"
            )

            if args.save_png:
                first_png_b64 = annotate_result["instances"][0].get("png_base64")
                if first_png_b64:
                    out_path = Path(args.save_png).expanduser().resolve()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(base64.b64decode(first_png_b64))
                    print(f"Saved annotated PNG to {out_path}")


def main() -> None:
    args = _parse_args()
    try:
        import anyio
    except ImportError as exc:
        raise SystemExit(
            "This example requires `anyio` (installed with `mcp[cli]`).\n"
            "Install project deps first: `pip install -e .`"
        ) from exc
    anyio.run(_async_main, args)


if __name__ == "__main__":
    main()
