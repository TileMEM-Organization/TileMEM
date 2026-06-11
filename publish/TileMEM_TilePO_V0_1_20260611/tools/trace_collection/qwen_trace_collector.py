#!/usr/bin/env python3
"""Collect Qwen-style MoE routing traces through Transformers.

This script is intended for an external big-memory trace collection machine. It
writes the unified JSONL schema consumed by TileMEM replay:

  step, phase, request_id, layer, token_count, experts, router_scores(optional)

The exact hook points can vary across Qwen/Transformers releases, so this file
keeps the output schema stable and fails clearly if router outputs are not
exposed by the loaded model.
"""

import json
import argparse
from pathlib import Path


def find_router_modules(model):
    for name, module in model.named_modules():
        lowered = name.lower()
        if "gate" in lowered or "router" in lowered:
            yield name, module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["smoke", "stress"], default="smoke")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--provenance-out")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    captured = []
    handles = []

    def make_hook(layer_name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor) or tensor.ndim < 2:
                return
            scores = tensor.detach().float()
            topk = torch.topk(scores.reshape(-1, scores.shape[-1]), k=args.top_k, dim=-1)
            captured.append((layer_name, topk.indices.cpu().tolist(), topk.values.cpu().tolist()))

        return hook

    router_modules = list(find_router_modules(model))
    if not router_modules:
        raise RuntimeError(
            "no router/gate modules found; cannot collect MoE routing trace"
        )
    for name, module in router_modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    max_prompts = args.max_prompts
    if max_prompts is None:
        max_prompts = 2 if args.mode == "smoke" else 128
    prompts = prompts[:max_prompts]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    total_input_tokens = 0
    with out_path.open("w") as out:
        for request_id, prompt in enumerate(prompts):
            captured.clear()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            total_input_tokens += int(inputs.input_ids.numel())
            with torch.inference_mode():
                _ = model(**inputs)
            for layer_idx, (_layer_name, experts, scores) in enumerate(captured):
                out.write(
                    json.dumps(
                        {
                            "step": step,
                            "phase": "prefill",
                            "request_id": request_id,
                            "layer": layer_idx,
                            "token_count": int(inputs.input_ids.numel()),
                            "experts": experts[0],
                            "router_scores": scores[0],
                        }
                    )
                    + "\n"
                )
                step += 1

    for handle in handles:
        handle.remove()
    if step == 0:
        raise RuntimeError("model ran but no routing events were captured")
    if args.provenance_out:
        provenance_path = Path(args.provenance_out)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "prompts": args.prompts,
                    "mode": args.mode,
                    "max_prompts": max_prompts,
                    "max_input_tokens": args.max_input_tokens,
                    "top_k": args.top_k,
                    "router_modules": len(router_modules),
                    "events": step,
                    "input_tokens": total_input_tokens,
                    "trace_path": str(out_path),
                    "torch": getattr(torch, "__version__", "unknown"),
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
