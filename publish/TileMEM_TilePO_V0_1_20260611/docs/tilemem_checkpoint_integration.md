# TileMEM Checkpoint Integration

This page describes the TileMEM checkpoint adapter API for converting a local
Hugging Face-style MoE checkpoint into a TileMEM serving artifact.

The public surface is available through the top-level SDK:

```python
import tilemem as TM

model_spec = TM.model_spec_from_hf_config(checkpoint_dir)
compiled = TM.plan_from_hf_config(checkpoint_dir)
weight_names = TM.checkpoint_weight_names(checkpoint_dir)
matches = TM.match_checkpoint_weights(
    weight_names,
    spec=model_spec,
    family=TM.infer_moe_topology(checkpoint_dir).family,
)
aliases = TM.build_runtime_weight_aliases(
    family=TM.infer_moe_topology(checkpoint_dir).family,
    layers=[0],
    experts=[0],
)
artifact = TM.export_checkpoint_artifact(
    checkpoint_dir,
    out_dir=artifact_dir,
    materialize=False,
)
print(artifact.tile_checkpoint_map_path)
serving = TM.run_serving_backend(
    checkpoint_dir=checkpoint_dir,
    backend="sglang",
    plan_path=artifact.manifest_path,
    execute=False,
)
```

`execute=False` is the default integration posture. It should build and return
the command that would launch the serving backend without starting a process.
Use `execute=True` only when the backend binary, model artifact, runtime
environment, and deployment policy are ready.

## Flow

1. `TM.model_spec_from_hf_config(...)` reads a Hugging Face `config.json` or an
   equivalent config dictionary and maps OLMoE, Qwen MoE, Mixtral, or generic
   MoE metadata into TileMEM's model spec.
2. `TM.plan_from_hf_config(...)` builds a TileMEM plan from the same config.
3. `TM.checkpoint_weight_names(...)` reads the checkpoint index, and
   `TM.match_checkpoint_weights(...)` maps checkpoint tensor names to TileMEM
   layers, experts, projection groups, and fallback paths.
4. `TM.export_checkpoint_artifact(...)` writes a local serving artifact with
   TileMEM metadata, matched checkpoint payload references, and a
   `tile_checkpoint_map.json` file. Each TileMEM tile records its stable key,
   N range, TileMEM payload offset, source checkpoint tensors, and source shard
   files. With `materialize=True`, referenced shard files are copied into the
   artifact.
5. `TM.build_runtime_weight_aliases(...)` provides KT/SGLang internal weight
   name candidates for each TileMEM layer/expert projection group.
6. `TM.build_serving_command(...)` produces the backend command line.
   SGLang commands use standard KT/SGLang flags, while TileMEM passes the plan
   path through the `TILEMEM_PLAN` environment value.
7. `TM.run_serving_backend(..., execute=False)` performs a dry run by default.
   Passing `execute=True` is the explicit opt-in for launching the backend.

## Local Synthetic Fixture

The companion example creates its own temporary fixture, so it does not require
network access or model downloads:

```bash
python examples/tilemem_checkpoint_integration.py
```

The fixture contains:

- a small Hugging Face-style `config.json`;
- a tiny `model.safetensors.index.json` with MoE expert tensor names;
- a local placeholder checkpoint shard;
- an output directory for the exported TileMEM checkpoint artifact.

The CLI path is:

```bash
tools/tilemem_checkpoint_prepare \
  --checkpoint-dir /path/to/hf/checkpoint \
  --out-dir build/checkpoint_artifact \
  --backend sglang \
  --dry-run
```

To opt into execution when the backend integration exists:

```bash
tools/tilemem_checkpoint_prepare \
  --checkpoint-dir /path/to/hf/checkpoint \
  --out-dir build/checkpoint_artifact \
  --backend kt_native \
  --execute
```

Keep BF16 or another validated fallback available when adapting real
checkpoints. TileMEM can own model planning, weight matching metadata, artifact
layout, command construction, and dry-run validation. The integrator remains
responsible for checkpoint provenance, numerical validation, calibration,
serving runtime availability, and production launch policy.
