"""Checks that `speech quantize` makes the files the converters made: for each model's F32 file, it runs `speech quantize`
to a type and compares what it wrote, tensor by tensor (each tensor's name, shape, type and bytes) and key by key (each
key's name, type and value, in order), with either a file of that type that a converter wrote, such as a released
file, or, where there is none, what gguf-py, which the converters quantize with, makes of each F32 tensor in the type
the written file gives it: gguf.quants.quantize() for Q8_0 and numpy's float16 for F16. It then compares the two files
byte for byte where there are two, and fails when a tensor or a key differs.

A comparison is <F32.gguf>=<converted.gguf> or <F32.gguf>=<f16|q8_0>; the second form needs numpy and gguf-py, which
each reference environment has. The file written goes to the work folder, which may hold the files compared, under a
name of its own, and is removed after its comparison.

usage: python3 tools/quantize_compare.py <speech> <work dir> <F32.gguf>=<converted.gguf>...
       uv run --project reference/<model> python tools/quantize_compare.py <speech> <work dir> <F32.gguf>=<f16|q8_0>...
"""

import argparse
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
# ggml's type ids of the types speech.cpp writes, with the values and bytes of one block.
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210)}
FILE_TYPES = {1: "f16", 7: "q8_0", 14: "q4_k", 16: "q5_k", 18: "q6_k"}


class Gguf:
    """A GGUF file's metadata: its keys in order with their types and values, and its tensors in order."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.f = f
            if f.read(4) != b"GGUF":
                raise SystemExit(f"{path} is not a GGUF file")
            self.version = self.u32()
            n_tensors, n_kv = self.u64(), self.u64()
            self.keys = []
            for _ in range(n_kv):
                key = self.string()
                kind = self.u32()
                self.keys.append((key, kind, self.value(kind)))
            self.tensors = []
            for _ in range(n_tensors):
                name = self.string()
                axes = self.u32()
                shape = [self.u64() for _ in range(axes)]
                self.tensors.append({"name": name, "shape": shape, "type": self.u32(), "offset": self.u64()})
            alignment = dict((k, v) for k, _, v in self.keys).get("general.alignment", 32)
            self.data = (f.tell() + alignment - 1) // alignment * alignment
            del self.f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", "surrogateescape")

    def value(self, kind):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            element, n = self.u32(), self.u64()
            return (element, [self.value(element) for _ in range(n)])
        fmt = SCALARS[kind]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def key(self, name):
        return next(v for k, _, v in self.keys if k == name)

    def nbytes(self, tensor):
        _, block, size = TYPES[tensor["type"]]
        n = 1
        for axis in tensor["shape"]:
            n *= axis
        return n // block * size

    def read(self, tensor):
        with open(self.path, "rb") as f:
            f.seek(self.data + tensor["offset"])
            return f.read(self.nbytes(tensor))


def type_name(t):
    return TYPES[t][0] if t in TYPES else str(t)


def compare_keys(made, other):
    """The differences of the keys, in order."""
    out = []
    if made.version != other.version:
        out.append(f"GGUF version {made.version} against {other.version}")
    for i in range(max(len(made.keys), len(other.keys))):
        a = made.keys[i] if i < len(made.keys) else None
        b = other.keys[i] if i < len(other.keys) else None
        if a != b:
            out.append(f"key {i}: {a[0] if a else 'none'} = {str(a[2])[:80] if a else ''} against {b[0] if b else 'none'} = {str(b[2])[:80] if b else ''}")
    return out


def expected_by_gguf_py(f32, tensor, made_type):
    """What the converters wrote of the F32 tensor in `made_type`: gguf-py's Q8_0, numpy's float16, or the F32 itself."""
    import numpy as np
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    values = np.frombuffer(f32.read(tensor), dtype="<f4").reshape(list(reversed(tensor["shape"])))
    if made_type == 0:
        return values.tobytes()
    if made_type == 1:
        return values.astype(np.float16).tobytes()
    if made_type == 8:
        return quantize(values, GGMLQuantizationType.Q8_0).tobytes()
    raise SystemExit(f"gguf-py does not write {type_name(made_type)}, the type of {tensor['name']} in the file speech quantize wrote")


def compare(speech, work, f32_path, other):
    released = Gguf(other) if os.path.exists(other) else None
    if released and os.path.samefile(f32_path, other):
        raise SystemExit(f"{other} is the F32 file itself; give the file of another type to compare with")
    kind = FILE_TYPES[released.key("general.file_type")] if released else other.lower()
    name = os.path.basename(f32_path).replace("-F32.gguf", f"-{kind.upper()}.gguf")
    # The file gets a name no other file has, made with it: under the conventional name in the reference's own folder it
    # would be the reference, which speech quantize would replace, the comparison read on both sides and the removal
    # delete.
    fd, out = tempfile.mkstemp(prefix=name.replace(".gguf", "-"), suffix=".gguf", dir=work)
    os.close(fd)
    try:
        subprocess.run([speech, "quantize", f32_path, out, "--type", kind], check=True)
        return compare_written(Gguf(f32_path), released, out, name, kind, other)
    finally:
        os.remove(out)


def compare_written(f32, released, out, name, kind, other):
    """Compares `out`, the file speech quantize wrote of `f32`, with `released`, or with gguf-py's conversion without one."""
    made = Gguf(out)
    failures = []
    if released:
        failures += compare_keys(made, released)
        if [t["name"] for t in made.tensors] != [t["name"] for t in released.tensors]:
            failures.append("the tensors are not the same names in the same order")
    by_name = {t["name"]: t for t in f32.tensors}
    types = {}
    for tensor in made.tensors:
        types[type_name(tensor["type"])] = types.get(type_name(tensor["type"]), 0) + 1
        source = by_name[tensor["name"]]
        if tensor["shape"] != source["shape"]:
            failures.append(f"{tensor['name']}: shape {tensor['shape']} against the F32 file's {source['shape']}")
            continue
        if released:
            theirs = next((t for t in released.tensors if t["name"] == tensor["name"]), None)
            if theirs is None:
                continue
            if theirs["type"] != tensor["type"] or theirs["shape"] != tensor["shape"]:
                failures.append(f"{tensor['name']}: {type_name(tensor['type'])} {tensor['shape']} against {type_name(theirs['type'])} {theirs['shape']}")
                continue
            expected = released.read(theirs)
        else:
            expected = expected_by_gguf_py(f32, source, tensor["type"])
        mine = made.read(tensor)
        if mine != expected:
            differing = sum(1 for a, b in zip(mine, expected) if a != b)
            failures.append(f"{tensor['name']} ({type_name(tensor['type'])}): {differing} of {len(mine)} bytes differ")
    summary = ", ".join(f"{n} {t}" for t, n in sorted(types.items()))
    against = os.path.basename(other) if released else f"gguf-py's {kind.upper()}"
    if released and not failures:
        hashes = []
        for path in (out, other):
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    digest.update(chunk)
            hashes.append(digest.hexdigest())
        identical = "the same bytes" if hashes[0] == hashes[1] else "different bytes"
        print(f"{name} against {against}: {len(made.tensors)} tensors ({summary}) and {len(made.keys)} keys equal, "
              f"{identical} (SHA-256 {hashes[0]})")
        if hashes[0] != hashes[1]:
            failures.append(f"the files differ although their tensors and keys are equal; SHA-256 {hashes[1]}")
    elif not failures:
        print(f"{name} against {against}: {len(made.tensors)} tensors ({summary}) equal")
    for failure in failures:
        print(f"FAIL {name}: {failure}")
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("speech")
    parser.add_argument("work")
    parser.add_argument("comparisons", nargs="+", metavar="F32.gguf=converted.gguf|f16|q8_0")
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    passed = True
    for comparison in args.comparisons:
        f32, _, other = comparison.partition("=")
        if not other:
            raise SystemExit(f"{comparison} is not <F32.gguf>=<converted.gguf> or <F32.gguf>=<f16|q8_0>")
        passed = compare(args.speech, args.work, f32, other) and passed
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
