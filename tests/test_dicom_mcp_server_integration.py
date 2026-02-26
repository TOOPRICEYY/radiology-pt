from __future__ import annotations

import base64
import functools
import json
import sys
import unittest
from pathlib import Path
from typing import Any

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:  # pragma: no cover - test environment dependency issue
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DICOM_ROOT = (REPO_ROOT / "dicom").resolve()
TEST_DESCRIPTIONS_ROOT = REPO_ROOT / "tests" / "test_descriptions"
SERIES_ID = "SER00012"
RANGE_START = 0
RANGE_END = 10

EXPECTED_TOOL_NAMES = {
    "set_dicom_root",
    "get_dicom_root",
    "refresh_dicom_index",
    "list_dicom_series",
    "get_dicom_series_details",
    "extract_dicom_range",
    "extract_dicom_ranges",
    "annotate_dicom_slice",
    "get_pixel_stats",
}

EXPECTED_RESOURCE_URIS = {
    "dicom://config",
    "dicom://series",
}

EXPECTED_RESOURCE_TEMPLATES = {
    "dicom://series/{series_id}",
    "dicom://series/{series_id}/range/{start}/{end}",
    "dicom://series/{series_id}/ranges/{ranges_spec}",
    "dicom://series/{series_id}/range/{start}/{end}/crop/{x}/{y}/{width}/{height}",
}


def mcp_integration_test(fn):
    @functools.wraps(fn)
    def wrapper(self):
        if ClientSession is None or StdioServerParameters is None or stdio_client is None:
            self.skipTest("MCP SDK not installed. Install project deps with `pip install -e .`")
        if not DICOM_ROOT.exists():
            self.skipTest(f"DICOM root does not exist: {DICOM_ROOT}")

        try:
            import anyio
        except ImportError:
            self.skipTest("anyio not installed. Install project deps with `pip install -e .`")

        async def runner() -> None:
            server = StdioServerParameters(
                command=sys.executable,
                args=["-m", "dicom_mcp.server", "--root", str(DICOM_ROOT)],
                cwd=str(REPO_ROOT),
            )
            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self.session = session
                    try:
                        await fn(self)
                    finally:
                        if hasattr(self, "session"):
                            delattr(self, "session")

        anyio.run(runner)

    return wrapper


class DicomMcpServerIntegrationTests(unittest.TestCase):

    def _text_parts(self, content_items: list[Any]) -> list[str]:
        parts: list[str] = []
        for item in content_items:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return parts

    def _parse_tool_result(self, result: Any) -> Any:
        if getattr(result, "isError", False):
            message = "\n".join(self._text_parts(getattr(result, "content", [])))
            self.fail(f"MCP tool call failed: {message or 'unknown error'}")

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
                return structured["result"]
            return structured

        texts = self._text_parts(getattr(result, "content", []))
        if not texts:
            return None
        parsed: list[Any] = []
        for text in texts:
            try:
                parsed.append(json.loads(text))
            except json.JSONDecodeError:
                parsed.append(text)
        return parsed[0] if len(parsed) == 1 else parsed

    async def _call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = await self.session.call_tool(name, arguments or {})
        return self._parse_tool_result(result)

    async def _read_resource_json(self, uri: str) -> Any:
        result = await self.session.read_resource(uri)
        parts = self._text_parts(getattr(result, "contents", []))
        self.assertTrue(parts, f"Expected text content from resource {uri}")
        self.assertEqual(len(parts), 1, f"Expected single text content item from {uri}")
        return json.loads(parts[0])

    def _test_description_dir(self) -> Path:
        test_name = self.id().rsplit(".", 1)[-1]
        path = TEST_DESCRIPTIONS_ROOT / test_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_png_artifact(self, png_b64: str, filename: str) -> Path:
        out_path = self._test_description_dir() / filename
        out_path.write_bytes(base64.b64decode(png_b64))
        return out_path

    @mcp_integration_test
    async def test_01_capability_discovery_lists_tools_resources_and_templates(self) -> None:
        tools = await self.session.list_tools()
        resources = await self.session.list_resources()
        templates = await self.session.list_resource_templates()

        self.assertEqual({tool.name for tool in tools.tools}, EXPECTED_TOOL_NAMES)
        self.assertEqual({str(item.uri) for item in resources.resources}, EXPECTED_RESOURCE_URIS)
        self.assertEqual(
            {str(item.uriTemplate) for item in templates.resourceTemplates},
            EXPECTED_RESOURCE_TEMPLATES,
        )

    @mcp_integration_test
    async def test_02_root_tools_and_refresh_index_use_local_directory(self) -> None:
        set_result = await self._call_tool("set_dicom_root", {"root_dir": str(DICOM_ROOT)})
        get_result = await self._call_tool("get_dicom_root")
        refresh_result = await self._call_tool("refresh_dicom_index")

        self.assertEqual(set_result["root_dir"], str(DICOM_ROOT))
        self.assertEqual(get_result["root_dir"], str(DICOM_ROOT))
        self.assertTrue(get_result["exists"])
        self.assertTrue(get_result["is_dir"])
        self.assertEqual(refresh_result["root_dir"], str(DICOM_ROOT))
        self.assertGreaterEqual(refresh_result["series_count"], 19)

    @mcp_integration_test
    async def test_03_list_dicom_series_includes_ser00012_summary(self) -> None:
        series_list = await self._call_tool("list_dicom_series", {"refresh": True})
        self.assertIsInstance(series_list, list)
        self.assertGreaterEqual(len(series_list), 19)

        series = next((item for item in series_list if item["series_id"] == SERIES_ID), None)
        self.assertIsNotNone(series, f"{SERIES_ID} not found in list_dicom_series")
        assert series is not None  # keep type checker happy
        self.assertEqual(series["rows"], 320)
        self.assertEqual(series["columns"], 320)
        self.assertEqual(series["instance_count"], 30)

    @mcp_integration_test
    async def test_04_get_dicom_series_details_for_ser00012_covers_indices_0_to_10(self) -> None:
        details = await self._call_tool("get_dicom_series_details", {"series_id": SERIES_ID})

        self.assertEqual(details["series_id"], SERIES_ID)
        self.assertEqual(details["rows"], 320)
        self.assertEqual(details["columns"], 320)
        self.assertEqual(details["instance_count"], 30)
        self.assertEqual(len(details["instances"]), 30)

        first_eleven = details["instances"][RANGE_START : RANGE_END + 1]
        self.assertEqual([item["index"] for item in first_eleven], list(range(0, 11)))
        self.assertTrue(all(item["rows"] == 320 for item in first_eleven))
        self.assertTrue(all(item["columns"] == 320 for item in first_eleven))

    @mcp_integration_test
    async def test_05_resources_config_series_and_series_details_return_expected_json(self) -> None:
        config = await self._read_resource_json("dicom://config")
        series_listing = await self._read_resource_json("dicom://series")
        series_details = await self._read_resource_json(f"dicom://series/{SERIES_ID}")

        self.assertEqual(config["server"], "dicom-mcp")
        self.assertEqual(config["root_dir"], str(DICOM_ROOT))
        self.assertTrue(config["root_exists"])
        self.assertIn("extract_dicom_range", config["usage"]["tools"])

        self.assertIsInstance(series_listing, list)
        self.assertTrue(any(item["series_id"] == SERIES_ID for item in series_listing))

        self.assertEqual(series_details["series_id"], SERIES_ID)
        self.assertEqual(series_details["instance_count"], 30)
        self.assertEqual(series_details["rows"], 320)
        self.assertEqual(series_details["columns"], 320)

    @mcp_integration_test
    async def test_06_range_metadata_resource_for_ser00012_0_to_10(self) -> None:
        data = await self._read_resource_json(
            f"dicom://series/{SERIES_ID}/range/{RANGE_START}/{RANGE_END}"
        )

        self.assertEqual(data["series_id"], SERIES_ID)
        self.assertEqual(data["requested_range"], {"start": RANGE_START, "end": RANGE_END})
        self.assertEqual(data["resolved_indices"], list(range(0, 11)))
        self.assertEqual(len(data["instances"]), 11)
        self.assertEqual(data["instances"][0]["index"], 0)
        self.assertEqual(data["instances"][-1]["index"], 10)
        self.assertNotIn("png_base64", data["instances"][0])

    @mcp_integration_test
    async def test_07_multi_range_and_cropped_resources_for_ser00012(self) -> None:
        multi = await self._read_resource_json(f"dicom://series/{SERIES_ID}/ranges/0-4,8-10")
        cropped = await self._read_resource_json(
            f"dicom://series/{SERIES_ID}/range/0/2/crop/10/20/30/40"
        )

        self.assertEqual(multi["series_id"], SERIES_ID)
        self.assertEqual(multi["resolved_indices"], [0, 1, 2, 3, 4, 8, 9, 10])
        self.assertEqual(multi["extracted_count"], 8)
        self.assertIsNone(multi["crop"])

        self.assertEqual(cropped["series_id"], SERIES_ID)
        self.assertEqual(cropped["resolved_indices"], [0, 1, 2])
        self.assertEqual(cropped["crop"], {"x": 10, "y": 20, "width": 30, "height": 40})
        render = cropped["instances"][0]["render"]
        self.assertEqual(render["original_size"], {"width": 320, "height": 320})
        self.assertEqual(render["output_size"], {"width": 30, "height": 40})
        self.assertEqual(render["crop"]["width"], 30)
        self.assertEqual(render["crop"]["height"], 40)

    @mcp_integration_test
    async def test_08_extract_dicom_range_tool_with_crop_window_and_step(self) -> None:
        data = await self._call_tool(
            "extract_dicom_range",
            {
                "series_id": SERIES_ID,
                "start": RANGE_START,
                "end": RANGE_END,
                "crop": {"x": 5, "y": 6, "width": 50, "height": 60},
                "include_png_base64": False,
                "include_dicom_bytes_base64": False,
                "include_pixels": False,
                "window": "brain",
                "step": 2,
            },
        )

        self.assertEqual(data["series_id"], SERIES_ID)
        self.assertEqual(data["resolved_indices"], [0, 2, 4, 6, 8, 10])
        self.assertEqual(data["extracted_count"], 6)
        self.assertEqual(data["crop"], {"x": 5, "y": 6, "width": 50, "height": 60})
        self.assertEqual(data["step"], 2)
        self.assertEqual(data["window"], {"center": 40.0, "width": 80.0})

        first_instance = data["instances"][0]
        self.assertIn("render", first_instance)
        self.assertEqual(first_instance["render"]["output_size"], {"width": 50, "height": 60})
        self.assertEqual(first_instance["render"]["window"], {"center": 40.0, "width": 80.0})
        self.assertNotIn("png_base64", first_instance)

    @mcp_integration_test
    async def test_09_extract_dicom_range_tool_can_return_png_and_dicom_bytes(self) -> None:
        data = await self._call_tool(
            "extract_dicom_range",
            {
                "series_id": SERIES_ID,
                "start": 0,
                "end": 0,
                "include_png_base64": True,
                "include_dicom_bytes_base64": True,
                "include_pixels": False,
            },
        )

        self.assertEqual(data["extracted_count"], 1)
        instance = data["instances"][0]
        self.assertIn("png_base64", instance)
        self.assertIn("dicom_bytes_base64", instance)
        self._save_png_artifact(instance["png_base64"], "extract_range_idx0.png")
        png_bytes = base64.b64decode(instance["png_base64"])
        dicom_bytes = base64.b64decode(instance["dicom_bytes_base64"])
        self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(png_bytes), 1000)
        self.assertGreater(len(dicom_bytes), 1000)

    @mcp_integration_test
    async def test_10_extract_dicom_ranges_tool_supports_multiple_ranges(self) -> None:
        data = await self._call_tool(
            "extract_dicom_ranges",
            {
                "series_id": SERIES_ID,
                "ranges": [{"start": 0, "end": 4}, {"start": 8, "end": 10}],
                "include_png_base64": False,
                "include_dicom_bytes_base64": False,
                "include_pixels": False,
            },
        )

        self.assertEqual(data["series_id"], SERIES_ID)
        self.assertEqual(data["requested_ranges"], [{"start": 0, "end": 4}, {"start": 8, "end": 10}])
        self.assertEqual(data["resolved_indices"], [0, 1, 2, 3, 4, 8, 9, 10])
        self.assertEqual(data["extracted_count"], 8)
        self.assertEqual([item["index"] for item in data["instances"]], [0, 1, 2, 3, 4, 8, 9, 10])

    @mcp_integration_test
    async def test_11_annotate_slice_and_get_pixel_stats_for_ser00012(self) -> None:
        annotate = await self._call_tool(
            "annotate_dicom_slice",
            {
                "series_id": SERIES_ID,
                "index": 3,
                "shapes": [
                    {"type": "bbox", "x": 10, "y": 20, "width": 30, "height": 40, "color": "red"},
                    {"type": "circle", "cx": 80, "cy": 90, "radius": 20, "color": [0, 255, 0]},
                    {"type": "ellipse", "cx": 150, "cy": 120, "rx": 25, "ry": 15, "color": "#00AAFF"},
                ],
                "include_png_base64": True,
                "include_dicom_bytes_base64": False,
                "include_pixels": False,
            },
        )
        stats_full = await self._call_tool(
            "get_pixel_stats",
            {"series_id": SERIES_ID, "index": 0},
        )
        stats_roi = await self._call_tool(
            "get_pixel_stats",
            {
                "series_id": SERIES_ID,
                "index": 0,
                "roi": {"x": 10, "y": 10, "width": 50, "height": 60},
            },
        )

        self.assertEqual(annotate["series_id"], SERIES_ID)
        self.assertEqual(annotate["resolved_indices"], [3])
        self.assertEqual(annotate["extracted_count"], 1)
        ann_instance = annotate["instances"][0]
        self.assertEqual(ann_instance["render"]["original_size"], {"width": 320, "height": 320})
        self.assertEqual(ann_instance["render"]["output_size"], {"width": 320, "height": 320})
        self.assertEqual(ann_instance["render"]["applied_annotations"], 3)
        self._save_png_artifact(ann_instance["png_base64"], "annotated_idx3.png")
        self.assertEqual(base64.b64decode(ann_instance["png_base64"])[:8], b"\x89PNG\r\n\x1a\n")

        self.assertEqual(stats_full["series_id"], SERIES_ID)
        self.assertEqual(stats_full["index"], 0)
        self.assertIsNone(stats_full["roi"])
        self.assertEqual(stats_full["pixel_count"], 320 * 320)
        self.assertIn("p95", stats_full["percentiles"])

        self.assertEqual(stats_roi["series_id"], SERIES_ID)
        self.assertEqual(stats_roi["index"], 0)
        self.assertEqual(stats_roi["roi"], {"x": 10, "y": 10, "width": 50, "height": 60})
        self.assertEqual(stats_roi["pixel_count"], 50 * 60)
        self.assertIn("p95", stats_roi["percentiles"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
