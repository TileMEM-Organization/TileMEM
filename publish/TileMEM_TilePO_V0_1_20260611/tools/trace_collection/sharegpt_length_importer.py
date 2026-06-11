#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for idx, line in enumerate(Path(args.input).read_text().splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        prompt_len = int(obj.get("prompt_length", obj.get("prompt_len", 128)))
        output_len = int(obj.get("output_length", obj.get("output_len", 32)))
        rows.append(
            {
                "request_id": idx,
                "arrival_us": idx * 1000,
                "prompt_length": prompt_len,
                "output_length": output_len,
            }
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
