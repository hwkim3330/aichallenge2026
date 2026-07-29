#!/usr/bin/env python3
"""Extract Plotly motion data embedded in an AI Challenge analytics page."""

import csv
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def load_text(source: str) -> str:
    if source.startswith(("http://", "https://")):
        request = Request(source, headers={"User-Agent": "aichallenge-motion-extractor/1.0"})
        with urlopen(request) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def skip_string(text: str, index: int) -> int:
    index += 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == '"':
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated JSON string")


def skip_json_value(text: str, index: int) -> int:
    if text[index] == '"':
        return skip_string(text, index)
    opening = text[index]
    if opening not in "[{":
        decoder = json.JSONDecoder()
        _, end = decoder.raw_decode(text[index:])
        return index + end
    closing = "]" if opening == "[" else "}"
    depth = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            index = skip_string(text, index)
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError("unterminated JSON value")


def extract_traces(html: str):
    marker = "Plotly.newPlot("
    position = 0
    plot_index = 0
    while True:
        start = html.find(marker, position)
        if start < 0:
            return
        index = start + len(marker)
        while html[index].isspace():
            index += 1
        plot_id, id_end = json.JSONDecoder().raw_decode(html[index:])
        index += id_end
        while html[index].isspace() or html[index] == ",":
            index += 1
        while html[index].isspace():
            index += 1
        traces, index_after_traces = json.JSONDecoder().raw_decode(html[index:])
        for trace_index, trace in enumerate(traces):
            yield plot_index, plot_id, trace_index, trace
        plot_index += 1
        position = index_after_traces + index


def value_at(values, index):
    if isinstance(values, list) and index < len(values):
        return values[index]
    return ""


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} INPUT.html-or-URL OUTPUT.csv")
    rows = []
    for plot_index, plot_id, trace_index, trace in extract_traces(load_text(sys.argv[1])):
        x_values = trace.get("x", [])
        y_values = trace.get("y", [])
        customdata = trace.get("customdata", [])
        length = max(len(x_values), len(y_values), len(customdata))
        for index in range(length):
            rows.append({
                "plot_index": plot_index,
                "plot_id": plot_id,
                "trace_index": trace_index,
                "trace_name": trace.get("name", ""),
                "index": index,
                "x": value_at(x_values, index),
                "y": value_at(y_values, index),
                "customdata": value_at(customdata, index),
                "marker_color": value_at(trace.get("marker", {}).get("color", []), index),
            })
    if not rows:
        raise SystemExit("no Plotly traces found")
    output = Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} samples from {max(row['plot_index'] for row in rows) + 1} plots to {output}")


if __name__ == "__main__":
    main()
