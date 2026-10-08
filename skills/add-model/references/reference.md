# The reference folder

`reference/<family>/` is the only place Python runs. Its job is to make the official implementation's numbers
reproducible, and to write the GGUF file the C++ reads.

## The environment

- `pyproject.toml` and `uv.lock` pin the official package, PyTorch and the rest. Upgrade a pin only with a reason, and
  rerun the existing dumps to show that the official output did not change, or say how it changed.
- `pins.py` pins each checkpoint by repository, revision, file, size and SHA-256, and checks them after download.
- A model the pinned version cannot run may get a folder of its own rather than moving the family's pin.

## Dumps

`dump.py` runs the official implementation on the CPU in float32 and writes `out/<case>/*.npy` and `meta.json`.

- Run the official entry point once with its stages wrapped, so that what is saved is what it computed; recompute the
  finer stages from those inputs and assert that the recomputation equals the official result.
- Fix every random draw (the noise, a sampler's seed) and save it, so that the C++ can start from the same draw.
- Write a case for each request option and each setting that changes the computation: a voice, a caption, a guidance
  mode, a chunk size, a language given and detected, a long input that the official code splits.
- Keep `meta.json` complete: the checkpoint's revision, the request as given, the lengths, and anything the C++ needs
  to check the dump against.
- Use only audio that may be published: FLEURS, Common Voice, or speech synthesized for the purpose. Dumps are large
  and stay out of Git (`reference/*/out/`).

## Conversion

`convert.py` writes one GGUF file per model, its codec or frontend included.

- The model's identity and languages go in the GGUF specification's `general.*` keys, under the names it gives them;
  no key of speech.cpp's own holds a fact the specification has a key for.
- Every constant the C++ needs is a key, from the checkpoint's config or, where it has none, from the official code;
  the converter says where each comes from.
- Every key is required and has one type; the tensors are exactly the ones the keys call for.
- The file is named from its keys by the GGUF naming convention (`<Name>-<size label>-<type>.gguf`).
- `speech.layout` names the layout's version and `speech.requires` the release that first reads it; the converter keeps
  a table of both.
- Write F32 alone. F16, Q8_0 and lower types are made with `speech quantize`, which follows the family's `layout.cpp`
  table; `tools/quantize_compare.py` shows that a file made again equals the released one byte for byte.

Run the conversion twice and compare the bytes: a converter that is not deterministic cannot be pinned.
