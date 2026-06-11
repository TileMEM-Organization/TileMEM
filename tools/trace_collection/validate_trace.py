#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


VALID_PHASES = {"prefill", "decode"}


def validate_event(obj, line_no):
    required = ["step", "phase", "request_id", "layer", "token_count", "experts"]
    missing = [field for field in required if field not in obj]
    if missing:
        raise ValueError(f"line {line_no}: missing fields: {', '.join(missing)}")
    if obj["phase"] not in VALID_PHASES:
        raise ValueError(f"line {line_no}: invalid phase: {obj['phase']}")
    if int(obj["token_count"]) <= 0:
        raise ValueError(f"line {line_no}: token_count must be positive")
    if not isinstance(obj["experts"], list) or not obj["experts"]:
        raise ValueError(f"line {line_no}: experts must be a non-empty list")
    if "router_scores" in obj and len(obj["router_scores"]) != len(obj["experts"]):
        raise ValueError(f"line {line_no}: router_scores length must match experts")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    events = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        validate_event(obj, line_no)
        events.append(obj)
    if not events:
        raise ValueError(f"trace has no events: {path}")

    summary = {
        "trace_collected": True,
        "trace_path": str(path),
        "events": len(events),
        "tokens": sum(int(event["token_count"]) for event in events),
        "requests": len({int(event["request_id"]) for event in events}),
        "layers": len({int(event["layer"]) for event in events}),
        "phases": sorted({event["phase"] for event in events}),
        "max_top_k": max(len(event["experts"]) for event in events),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"validated {summary['events']} events from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
