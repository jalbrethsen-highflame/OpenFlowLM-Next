r"""d3_ref_cpu: vllm-sr/d3 decisions on the CPU with transformers, the NPU path's reference.

The same forward as d3_runtime's PromptEncoder for one unpadded prompt: BF16 Qwen3.5 text
layers (SDPA), every full-attention layer given an all-true key mask (so no causal mask),
the Gated DeltaNet layers causal, positions 0..L-1 in all three rope sections, final norm,
fp32 readout at the last position. Builds only the language model from config.json's
text_config and loads its tensors from the shards (no checkpoint code is run).

    python d3_ref_cpu.py --src ~/models/d3-src --cases cases.json --out cpu.json [--threads 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_npu import Prompter, options  # noqa: E402


def load_language_model(src: Path):
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    tc = json.loads((src / "config.json").read_text())["text_config"]
    cfg = Qwen3_5TextConfig(**tc)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        lm = Qwen3_5TextModel(cfg)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    sd = {}
    for shard in sorted({v for k, v in index.items() if k.startswith("language_model.")}):
        with safe_open(str(src / shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("language_model."):
                    sd[k[len("language_model."):]] = f.get_tensor(k)
    missing, unexpected = lm.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if "rotary_emb" not in m]
    if missing or unexpected:
        raise RuntimeError(f"state dict: missing {missing[:5]}, unexpected {unexpected[:5]}")
    lm.rotary_emb = type(lm.rotary_emb)(config=cfg)          # its buffers were on meta
    return lm.to(torch.bfloat16).eval(), cfg


@torch.inference_mode()
def last_hidden(lm, cfg, ids: list[int]) -> torch.Tensor:
    L = len(ids)
    embeds = lm.embed_tokens(torch.tensor([ids]))
    ramp = torch.arange(L).view(1, 1, -1).expand(4, 1, -1)
    text_positions, mrope = ramp[0], ramp[1:]
    rope = lm.rotary_emb(embeds, mrope)
    keys = torch.ones((1, 1, 1, L), dtype=torch.bool)
    hidden = embeds
    for layer, kind in zip(lm.layers, cfg.layer_types):
        out = layer(hidden, position_embeddings=rope, attention_mask=keys if kind == "full_attention" else None,
                    position_ids=text_positions, past_key_values=None, use_cache=False)
        hidden = out[0] if isinstance(out, tuple) else out
    return lm.norm(hidden)[0, -1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from safetensors.numpy import load_file
    W = torch.from_numpy(load_file(str(a.src / "readout.safetensors"))["weight"].astype(np.float32))
    pr = Prompter(a.src)
    t0 = time.time()
    lm, cfg = load_language_model(a.src)
    print(f"loaded in {time.time() - t0:.0f} s", file=sys.stderr)
    out = []
    for c in json.loads(a.cases.read_text()):
        ids = pr.ids(c["state"], c["question"])
        t1 = time.time()
        h = last_hidden(lm, cfg, ids)
        lg = (h.float() @ W.T).double().numpy()
        keys = options(c["question"])[0]
        z = lg[:len(keys)] / pr.temperature
        p = np.exp(z - z.max()); p /= p.sum()
        out.append({"id": c["id"], "tokens": len(ids), "s": round(time.time() - t1, 1), "keys": keys,
                    "probs": p.tolist(), "logits": lg[:len(keys)].tolist()})
        print(f"{c['id']:>14}  {len(ids):5d} tok  {time.time() - t1:6.0f} s  -> {keys[int(p.argmax())]} ({p.max():.3f})",
              file=sys.stderr)
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
