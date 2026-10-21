r"""Flint's Q4_1 GGUF (RecursiveSI/flint-int4-gguf) -> the open engine's .q4nx container.

The engine's MoE pack ops (expert_stripes, expert_down, std_perm) read q4_1 chunks from a
q4nx container, so the GGUF is rewritten into one, losslessly but for the block scales:

  * Q4_1 tensors -> I8 [rows / 32, cols / 256, 5120] chunks in the file raster (q4_1_pack.py's
    chunk layout: d / m bf16 [kb][r], nibbles [r / 16][k][r % 16]). The 4-bit codes are copied;
    d and m go fp16 -> bf16 (RNE), the narrowing every q4nx model and the NPU kernels apply.
  * F32 tensors stay F32 (norms, q / k norms, sinks, the router, the embedding). The router is
    transposed to [hidden, experts] (moe_router.weight), the layout the host router reads, and
    the layer norms take the HF names (input_layernorm, post_attention_layernorm) the engine
    looks up by suffix.

Tensor names are the GGUF's (blk.N.attn_q.weight, ...); the Flint manifest names them so.
The container is a safetensors file: u64 header length, JSON header, data.

    python utilities/gguf-utils/flint_to_q4nx.py flint-int4-q4_1.gguf OUT_DIR/model.q4nx
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GGUF_F32, GGUF_F16, GGUF_Q4_1 = 0, 1, 3
CH = 5120


def read_gguf(path: Path):
    """(tensor infos {name: (type, dims fastest-first, offset)}, data base, kv dict)."""
    b = np.memmap(path, dtype=np.uint8, mode="r")
    o = 0

    def r(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, b, o)
        o += struct.calcsize(fmt)
        return v[0]

    def s():
        nonlocal o
        n = r("<Q")
        v = bytes(b[o:o + n]).decode("utf8", "replace")
        o += n
        return v

    sizes = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = r("<I"), r("<Q")
            return [val(et) for _ in range(n)]
        return r(sizes[t])

    if bytes(b[:4]) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    o = 4
    r("<I")
    nt, nk = r("<Q"), r("<Q")
    kv = {}
    for _ in range(nk):
        k = s()
        kv[k] = val(r("<I"))
    infos = {}
    for _ in range(nt):
        name = s()
        nd = r("<I")
        dims = [r("<Q") for _ in range(nd)]
        t = r("<I")
        off = r("<Q")
        infos[name] = (t, dims, off)
    align = kv.get("general.alignment", 32)
    base = (o + align - 1) // align * align
    return b, infos, base, kv


def q4_1_chunks(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """GGUF Q4_1 bytes of a row-major [rows, cols] tensor -> uint8 [rows/32, cols/256, 5120]."""
    nb = cols // 32
    blk = raw.reshape(rows, nb, 20)
    d = np.ascontiguousarray(blk[..., 0:2]).view(np.float16).reshape(rows, nb)
    m = np.ascontiguousarray(blk[..., 2:4]).view(np.float16).reshape(rows, nb)
    qs = blk[..., 4:20]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=-1).reshape(rows, cols)      # value j at [.., j]
    R, C = rows // 32, cols // 256
    d16 = d.astype(np.float32).astype(bfloat16).view(np.uint16).reshape(R, 32, C, 8)
    m16 = m.astype(np.float32).astype(bfloat16).view(np.uint16).reshape(R, 32, C, 8)
    out = np.empty((R, C, CH), np.uint8)
    # d / m: index kb * 32 + r
    out[:, :, 0:512] = d16.transpose(0, 2, 3, 1).reshape(R, C, 256).view(np.uint8).reshape(R, C, 512)
    out[:, :, 512:1024] = m16.transpose(0, 2, 3, 1).reshape(R, C, 256).view(np.uint8).reshape(R, C, 512)
    # nibbles: p = (r / 16) * 4096 + k * 16 + r % 16, even p low
    qq = q.reshape(R, 2, 16, C, 256).transpose(0, 3, 1, 4, 2).reshape(R, C, 8192)
    out[:, :, 1024:] = qq[..., 0::2] | (qq[..., 1::2] << 4)
    return out


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    b, infos, base, kv = read_gguf(src)
    if kv.get("general.architecture") != "flint":
        raise SystemExit(f"{src}: architecture {kv.get('general.architecture')!r}, not flint")
    # the header first (every shape is known from the dims), then the tensors one at a time
    header, plan, off = {}, [], 0
    for name, (t, dims, toff) in infos.items():
        nval = int(np.prod(dims))
        if t == GGUF_F32:
            shape = list(reversed(dims))
            # the engine finds a layer's small tensors by suffix (Core::const_tensor): its
            # norms under the HF names
            out = name.replace(".attn_norm.", ".input_layernorm.").replace(".ffn_norm.", ".post_attention_layernorm.")
            if name.endswith("ffn_gate_inp.weight"):          # [E, hid] -> [hid, E]
                out, shape = name.replace("ffn_gate_inp", "moe_router"), shape[::-1]
            nbytes, dtype = nval * 4, "F32"
        elif t == GGUF_Q4_1:
            cols = dims[0]
            rows = nval // cols
            if rows % 32 or cols % 256:
                raise SystemExit(f"{name}: [{rows}, {cols}] is not whole 32 x 256 chunks")
            out, shape, nbytes, dtype = name, [rows // 32, cols // 256, CH], rows // 32 * (cols // 256) * CH, "I8"
        else:
            raise SystemExit(f"{name}: GGUF type {t} is not F32 / Q4_1")
        header[out] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        plan.append((name, out, t, dims, toff, nbytes))
        off += nbytes
    hjson = json.dumps({"__metadata__": {"format": "q4nx", "source": src.name, "arch": "flint"}, **header}).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    tmp = dst.with_suffix(".q4nx.part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for name, out, t, dims, toff, nbytes in plan:
            nval = int(np.prod(dims))
            start = base + toff
            if t == GGUF_F32:
                a = np.frombuffer(b, np.float32, nval, start).reshape(list(reversed(dims)))
                if out.endswith("moe_router.weight"):
                    a = a.T
            else:
                cols = dims[0]
                a = q4_1_chunks(np.frombuffer(b, np.uint8, nval // 32 * 20, start), nval // cols, cols)
            data = np.ascontiguousarray(a).tobytes()
            if len(data) != nbytes:
                raise SystemExit(f"{name}: {len(data)} B, planned {nbytes}")
            f.write(data)
            print(f"{out:40s} {header[out]['dtype']} {header[out]['shape']}", flush=True)
    tmp.rename(dst)
    print(f"wrote {dst} ({off / 1e9:.2f} GB, {len(header)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
