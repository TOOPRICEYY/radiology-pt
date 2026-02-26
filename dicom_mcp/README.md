# DICOM MCP Server

MCP server for browsing and extracting DICOM series from a configurable root directory.

## Features

- Configurable DICOM root (`--root` or `DICOM_ROOT_DIR`)
- MCP resources for:
  - series listing
  - series details
  - range metadata preview
  - multi-range metadata preview
  - cropped range metadata preview
- MCP tools for:
  - listing series/details
  - extracting one range or multiple ranges
  - optional crop preprocessing
  - optional overlays (bbox / circle / ellipse) on a specific DICOM slice
  - dedicated single-slice annotation helper (`annotate_dicom_slice`)

## Install

```bash
pip install -r dicom_mcp/requirements.txt
```

Or install as a package from repo root (recommended):

```bash
pip install -e .
```

## Run (stdio)

```bash
python -m dicom_mcp.server --root /absolute/path/to/dicom
```

Or via env:

```bash
export DICOM_ROOT_DIR=/absolute/path/to/dicom
python -m dicom_mcp.server
```

## Example MCP Client Config

```json
{
  "mcpServers": {
    "dicom-mcp": {
      "command": "python",
      "args": ["-m", "dicom_mcp.server", "--root", "/absolute/path/to/dicom"]
    }
  }
}
```

## Local MCP Client Example (stdio)

This script launches `dicom_mcp.server` locally over stdio and uses the MCP Python SDK
to call tools/resources through a real MCP client session (useful for local testing).

```bash
python examples/call_dicom_mcp_locally.py --root ./dicom --save-png /tmp/annotated.png
```

## Resource URIs

- `dicom://config`
- `dicom://series`
- `dicom://series/{series_id}`
- `dicom://series/{series_id}/range/{start}/{end}`
- `dicom://series/{series_id}/ranges/{ranges_spec}` (example: `0-4,10-12`)
- `dicom://series/{series_id}/range/{start}/{end}/crop/{x}/{y}/{width}/{height}`

`start` / `end` use zero-based inclusive indices in the sorted series order.

## Annotation Schema (tool input)

Annotations are applied to a specific DICOM using `target_index` (series index) or
`target_instance_number`.

Examples:

```json
[
  {"type":"bbox","target_index":5,"x":30,"y":40,"width":80,"height":120,"color":"red"},
  {"type":"circle","target_index":5,"cx":220,"cy":180,"radius":30,"color":[0,255,0]},
  {"type":"ellipse","target_index":8,"cx":200,"cy":200,"rx":40,"ry":20,"color":"#00aaff"}
]
```

Use `"coordinate_space": "cropped"` if annotation coordinates refer to the cropped image
instead of the source image.
