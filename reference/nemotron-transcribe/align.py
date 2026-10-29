"""Runs Nemotron 3.5 Transcribe's official CTC timestamp alignment on the CPU in float32 over the transcripts dump.py
wrote, and saves what each of its stages took and gave, which the C++ port's word times and speakers are checked against.

usage: uv run python align.py <out dir> <case>[:<text>]...
       uv run python align.py <out dir> --text

A case is a folder of dump.py's under <out dir>/<name>/, <name> being the model's; its audio is encoded again and its
generated tokens are aligned, and the outputs go to its ctc/ folder. With :<text>, the transcript is TRANSCRIPTS[<text>]
applied to the generated one and the outputs go to ctc-<text>/, to show how the alignment fails or what it does with a
transcript the model did not write.

The audio runs through the perception module as SALMAutomodel.generate(generate_timestamps=True) runs it: the parallel
expert encoder's windowed inference with return_ctc_timestamp_inputs, which gives the ASR encoder's kept frames in
bfloat16, the speaker probabilities on the encoder's frames and the speaker branch's native 10 ms probabilities. The
transcript is the generated tokens decoded as _decode_timestamp_transcript() decodes them, special tokens kept, then
cleaned by clean_sot_transcript() with the language model tokenizer's special tokens. The aligner is the one
get_ctc_timestamp_aligner() builds for the encoder: the checkpoint's bundled head in float32, the encoder's windows, and
NeMo's default settings, since the encoder's ctc_timestamp_alignment is empty and the tuned parameters config.json gives
under ctc_timestamps.alignment_config are read by neither NeMo nor NVIDIA's vLLM plugin, which reads a top-level
ctc_timestamps.speaker_logprob_weight the checkpoint does not have. prepare_from_inputs() and align_prepared_batch() run
it, as the vLLM plugin does, and its result goes through the plugin's _public_result(). The banded search's forward pass
runs compiled with Numba when it is installed; every pass is also run by the NumPy version and must give the same scores
and pointers.

--text writes <out dir>/<name>/ctc-text.json: for each text of TEXTS, the words parse_sot_words() gives, the units
_timestamp_units() makes of them, their CTC token ids, each speaker stream's target and the transcript's columns, or the
error the aligner raises at the first stage that refuses the text; and <out dir>/<name>/ctc-tokenizer.json: the ids the
CTC tokenizer's SentencePiece processor encodes texts of mixed scripts, spaces, tags, digits and marks from a fixed seed
into, and the id piece_to_id() gives each of their characters.

ctc/input.npy              the CTC head's input, the ASR encoder's kept frames rounded to bfloat16, [F, 1280], of which
                           the first meta.json's valid_frames are the audio's and the rest the frame the preprocessor's
                           extra feature frame gives, which the head's attention masks
ctc/windows/<k>/           the head on window k (meta.json gives the windows and the frames of each that the attention
                           does not mask), W being the window's encoder frames:
  upsampled.npy              the transposed convolution's output, [2W, 1280]
  layers.npy                 the embedding norm's and each transformer layer's output, [L + 1, 2W, 1280]
  output.npy                 the transformer's final norm, [2W, 1280]
  trunk.npy                  the upsampled frames plus the scaled transformer output, [2W, 1280]
ctc/log_probs.npy          the head's log probabilities of every class over each window's core, [2F, 23863]
ctc/speaker_probs.npy      the speaker probabilities on the encoder's frames, as the alignment reads them, [F, S]
ctc/diarization_probs.npy  the speaker branch's native 10 ms probabilities, [T, S]
ctc/diarization_labels.npy the 10 ms labels the alignment reads, the probabilities above 0.5, [max speakers, T], uint8
ctc/columns.npy            the log probabilities of the transcript's classes, blank first, as the search reads them,
                           [2F, C]
ctc/speaker.npy            the speaker probabilities resampled to the CTC frames, [2F, S]
ctc/meta.json              the versions, the windows, the transcript, the record, the mapping, each stream's search,
                           the words before and after refinement, the result and the public result, or the error
"""

import argparse
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

import nemo.collections.speechlm2 as speechlm2
import official
from nemo.collections.speechlm2.parts import ctc_timestamp_utils as ctc
from pins import MODELS, NEMO, NEMO_COMMIT, snapshot

# Texts the alignment's text stages run on without audio.
TEXTS = [
    "",
    "   ",
    "Hello, world! It's 3:45 p.m. — fine.",
    "the the the cat sat on the mat mat",
    "So <spk:0> hello there. <spk:1> Hi! <spk:0> Bye — bye.",
    "<spk:2> first <spk:0> second <spk:2> third",
    "<SPK:1> case of the tag",
    "Well <ev:laughter> that was <sty:stt:calm_reflective> fun <sty:end:calm_reflective>.",
    "米国体操連盟、U.S.A.ジムナスティックスは、選手全員のための安全な環境づくりを約束。",
    "<spk:0> はい、そうです。 <spk:1> いいえ、違います！ <ev:laughter> ははは",
    "今日は2024年の一月二十一日です。",
    "鬱",
    "彼は叱られた。",
    "１２３番",
    "ＡＢＣの会",
    "ｶﾀｶﾅ",
    "你好，世界。今天天气很好。",
    "안녕하세요. 반갑습니다.",
    "안녕하세요 こんにちは",
    "Привет, мир. Как дела?",
    "مرحبا بالعالم",
    "Ünïcödé café naïve façade",
    "Café déjà vu",
    "ﬁne ligatures ① ½",
    "smile 🙂 please",
    "e-mail co-op A/B 50% $5 #1 @home",
    "สวัสดีครับ",
]

# Transcripts in place of the generated one for an audio case, each a function of the generated transcript.
TRANSCRIPTS = {
    # A character the CTC vocabulary cannot spell.
    "oov": lambda text: text + "鬱",
    # More tokens than the CTC frames hold.
    "long": lambda text: " ".join([text] * 60),
    # The speaker tags removed: one stream, aligned as speaker 0.
    "untagged": lambda text: ctc.MultiSpeakerSOTWordTimestampAligner._SPEAKER_TAG_RE.sub(" ", text),
    # Words before the first tag, which have no speaker.
    "pretag": lambda text: "So, well. " + text,
    # A tag the speakers do not hold, given to the words of the second turn of speaker 0.
    "extra-tag": lambda text: extra_tag(text),
}


def extra_tag(text):
    parts = ctc.MultiSpeakerSOTWordTimestampAligner._SPEAKER_TAG_RE.split(text)
    turns = [i for i in range(1, len(parts), 2) if parts[i] == "0"]
    if len(turns) < 2:
        raise SystemExit("extra-tag needs a transcript with two turns of <spk:0>")
    parts[turns[1]] = "7"
    return "".join(f"<spk:{p}>" if i % 2 else p for i, p in enumerate(parts))


parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("cases", nargs="*")
parser.add_argument("--text", action="store_true")
args = parser.parse_args()
for case in args.cases:
    _, _, text_name = case.partition(":")
    if text_name and text_name not in TRANSCRIPTS:
        raise SystemExit(f"{case}: no transcript {text_name}; transcripts: {', '.join(TRANSCRIPTS)}")
if not args.cases and not args.text:
    raise SystemExit("give cases, --text or both")

torch.set_grad_enabled(False)
MODEL = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
PIN = MODELS[MODEL]
FOLDER = snapshot(PIN)
artifact = ctc.load_ctc_timestamp_artifact(os.path.join(FOLDER, official.WEIGHTS), map_location="cpu", dtype=torch.float32)
decoder = artifact.decoder
BLANK = int(decoder.num_classes_with_blank) - 1
special_tokens = tuple(AutoTokenizer.from_pretrained(FOLDER).all_special_tokens)
# The vLLM plugin's module, loaded from its file so that its package, which imports vLLM, is not.
spec = importlib.util.spec_from_file_location(
    "ctc_timestamps", os.path.join(os.path.dirname(speechlm2.__file__), "vllm", "salm", "ctc_timestamps.py"))
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


def plain(value):
    """A JSON value of nested records: NumPy and torch scalars and arrays become numbers and lists, keys strings."""
    if isinstance(value, dict):
        return {("null" if k is None else str(k)): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return plain(value.tolist())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def save(folder, name, array):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, name + ".npy"), np.ascontiguousarray(array))


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=1)


def tokenizer_texts(pieces):
    """Texts of mixed scripts, spaces, tags, digits and marks from a fixed seed, for the CTC tokenizer's cases."""
    rng = random.Random(0)
    chars = sorted({c for p in pieces for c in p if c != "\u2581"})
    pools = [
        chars,
        [chr(c) for c in range(0x20, 0x7F)],
        list(" \t\n\r\u3000\u200b\u00a0\u2028\u2581"),
        [chr(c) for c in range(0xFF01, 0xFF5F)] + [chr(c) for c in range(0xFF61, 0xFFA0)],
        [chr(c) for c in range(0x300, 0x370)] + ["\u3099", "\u309a"],
        [chr(c) for c in range(0x3040, 0x3100)] + [chr(c) for c in range(0x4E00, 0x4E00 + 400)] + ["鬱", "叱", "凿"],
        [chr(c) for c in range(0xAC00, 0xAC00 + 200)] + [chr(c) for c in range(0x1100, 0x1180)],
        [chr(c) for c in range(0x600, 0x700)] + [chr(c) for c in range(0xE00, 0xE80)] + [chr(c) for c in range(0x900, 0x980)],
        ["\U0001F600", "\U0001F642", "\u2460", "\u00bd", "\ufb01", "\u212b", "\u00c5", "\u1e9b\u0323", "\x00", "\x85", "\ufeff"],
        ["<spk:0>", "<spk:1>", "<spk:15>", "<ev:laughter>", "<sty:stt:flat>", "<sty:end:flat>", "<SPK:0>", "<spk:", "<ev:>", "<unk>"],
    ]
    return ["".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 30))) for _ in range(4000)]


def text_stages(aligner, text):
    """What the aligner's text stages give for a cleaned transcript, up to the first that raises."""
    out = {"text": text}
    try:
        out["words"] = aligner.parse_sot_words(text)
        out["units"] = ctc._timestamp_units(out["words"], aligner.timestamp_language or "auto")
        out["tokenized"] = aligner._tokenize_words(out["units"], BLANK)
        streams = []
        for tag, words in aligner._group_words_by_speaker(out["tokenized"]).items():
            labels, state_to_word = aligner._build_ctc_target(words, BLANK)
            streams.append({"speaker_tag": tag, "labels": labels, "state_to_word": state_to_word})
        out["streams"] = streams
        out["columns"] = [BLANK] + sorted({label for s in streams for label in s["labels"]} - {BLANK})
    except Exception as error:  # noqa: BLE001
        out["error"] = {"type": type(error).__name__, "message": str(error)}
    return out


if args.text:
    aligner = ctc.MultiSpeakerSOTWordTimestampAligner(ctc_decoder=decoder, tokenizer=artifact.tokenizer)
    cases = [text_stages(aligner, ctc.MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript(t, special_tokens))
             for t in TEXTS]
    path = os.path.join(args.out_dir, MODEL, "ctc-text.json")
    write_json(path, {"blank": BLANK, "special_tokens": special_tokens, "cases": cases})
    for case in cases:
        print(repr(case["text"][:40]), case.get("error", {}).get("type", "ok"), flush=True)
    print(len(cases), "texts ->", path, flush=True)
    processor = artifact.tokenizer.tokenizer
    texts = tokenizer_texts([processor.id_to_piece(i) for i in range(processor.get_piece_size())])
    path = os.path.join(args.out_dir, MODEL, "ctc-tokenizer.json")
    write_json(path, {"cases": [{"text": t, "ids": processor.encode(t)} for t in texts],
                      "pieces": [{"text": c, "id": processor.piece_to_id(c)} for c in sorted({c for t in texts for c in t})]})
    print(len(texts), "tokenizer texts ->", path, flush=True)

if not args.cases:
    raise SystemExit(0)

model = official.load(FOLDER, with_language_model=False)
pee = model.perception.encoder
if pee.ctc_timestamp_alignment:
    raise SystemExit(f"the encoder's ctc_timestamp_alignment is {pee.ctc_timestamp_alignment}, not NeMo's defaults")
tokenizer = model.tokenizer


class Record:
    """What the wrapped calls of one case took and gave."""

    def __init__(self):
        self.build = None
        self.heads = None
        self.calls = []
        self.searches = []
        self.mappings = []
        self.refined = []


rec = Record()

build_inputs = pee._build_ctc_timestamp_inputs


def wrapped_build(asr_encoded, asr_lengths, speaker_probs, speaker_lengths, diarization_probs, diarization_lengths):
    rec.build = {"diarization_probs": diarization_probs[0].clone(), "diarization_length": int(diarization_lengths[0])}
    return build_inputs(asr_encoded, asr_lengths, speaker_probs, speaker_lengths, diarization_probs, diarization_lengths)


pee._build_ctc_timestamp_inputs = wrapped_build


def hook(store):
    def run(module, inputs_, output):
        if rec.heads is not None:
            store(output)
    return run


decoder.upsample.register_forward_hook(hook(lambda o: rec.heads.append(("upsampled", o.clone()))))
decoder.transformer.embed_norm.register_forward_hook(hook(lambda o: rec.heads.append(("embed", o.clone()))))
for index, layer in enumerate(decoder.transformer.layers):
    layer.register_forward_hook(hook(lambda o, index=index: rec.heads.append((index, o.clone()))))
decoder.transformer.final_norm.register_forward_hook(hook(lambda o: rec.heads.append(("final", o.clone()))))
trunk = decoder.trunk


def wrapped_trunk(states, encoded_lengths=None):
    first = len(rec.heads) if rec.heads is not None else 0
    out = trunk(states, encoded_lengths)
    if rec.heads is not None:
        rec.calls.append({"states": states.clone(), "lengths": encoded_lengths.clone(), "trunk": out.clone(),
                          "stages": rec.heads[first:]})
    return out


decoder.trunk = wrapped_trunk

search = ctc._banded_ctc_search


def wrapped_search(logp, labels, positions, prior, greedy=None):
    path, score, starts = search(logp, labels, positions, prior, greedy)
    rec.searches.append({"labels": labels, "positions": positions, "prior_dtype": str(np.asarray(prior).dtype),
                         "starts": starts, "path": path, "score": float(score)})
    return path, score, starts


ctc._banded_ctc_search = wrapped_search
forward = ctc._ctc_banded_forward


def wrapped_forward(logp, labels, positions, prior, starts, width):
    previous, pointers = forward(logp, labels, positions, prior, starts, width)
    numpy_previous, numpy_pointers = ctc._ctc_banded_forward_numpy(logp, labels, positions, prior, starts, width)
    if not (np.array_equal(previous, numpy_previous) and np.array_equal(pointers, numpy_pointers)):
        raise SystemExit("the compiled and the NumPy forward passes of the banded search differ")
    return previous, pointers


ctc._ctc_banded_forward = wrapped_forward
mapping_of = ctc._one_pass_speaker_mapping


def wrapped_mapping(ctc_, streams, speaker, threshold, greedy):
    mapping = mapping_of(ctc_, streams, speaker, threshold, greedy)
    rec.mappings.append({"greedy": greedy, "threshold": threshold, "mapping": mapping})
    return mapping


ctc._one_pass_speaker_mapping = wrapped_mapping
refine = ctc._refine_word_timestamps


def wrapped_refine(result, max_frames=12):
    before = json.loads(json.dumps(plain(result["speaker_word_timestamps"])))
    out = refine(result, max_frames)
    rec.refined.append({"max_frames": max_frames, "before": before})
    return out


ctc._refine_word_timestamps = wrapped_refine


def windows(frames):
    """The head's windows as _iter_ctc_windows() cuts them: core start and end, context start and end."""
    out = []
    for start in range(0, frames, aligner.online_inference_length or frames):
        end = min(start + (aligner.online_inference_length or frames), frames)
        out.append((start, end, max(start - aligner.chunk_left_context, 0), min(end + aligner.chunk_right_context, frames)))
    return out


aligner = ctc.MultiSpeakerSOTWordTimestampAligner(ctc_decoder=decoder, tokenizer=artifact.tokenizer)
aligner.online_inference_length = int(pee.online_inference_length)
aligner.chunk_left_context = int(pee.chunk_left_context)
aligner.chunk_right_context = int(pee.chunk_right_context)

for case in args.cases:
    case_name, _, text_name = case.partition(":")
    folder = os.path.join(args.out_dir, MODEL, case_name)
    out = os.path.join(folder, "ctc-" + text_name if text_name else "ctc")
    with open(os.path.join(folder, "meta.json")) as f:
        dump_meta = json.load(f)
    rec.__init__()
    started = time.time()
    audio = np.load(os.path.join(folder, "audio.npy"))
    with pee.online_inference():
        _, _, inputs = model.perception(input_signal=torch.from_numpy(audio)[None],
                                        input_signal_length=torch.tensor([len(audio)]),
                                        return_ctc_timestamp_inputs=True)
    # The encoder's frames include the one the preprocessor's extra feature frame gives, which the lengths leave out
    # and the head's attention masks.
    asr = inputs.asr_encoded[0].T.float()
    frames, valid = asr.shape[0], int(inputs.asr_encoded_lengths[0])
    if frames != dump_meta["frames"] or valid != dump_meta["audio_frames"]:
        raise SystemExit(f"{case}: {frames} encoder frames of {valid} valid, the dump {dump_meta['frames']} of "
                         f"{dump_meta['audio_frames']}")
    dumped = torch.from_numpy(np.load(os.path.join(folder, "asr.npy"))).to(torch.bfloat16).float()
    if not torch.equal(dumped, asr):
        print(f"{case}: {int((dumped != asr).sum())} of {asr.numel()} values of the head's input differ from the "
              f"dump's asr.npy rounded to bfloat16", flush=True)

    tokens = dump_meta["tokens"]
    decoded = tokenizer.ids_to_text(tokens, remove_special_tokens=False)
    generated = ctc.MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript(decoded, special_tokens)
    if generated != ctc.MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript(dump_meta["text"], special_tokens):
        raise SystemExit(f"{case}: the decoded tokens and the dump's text differ once cleaned")
    transcript = TRANSCRIPTS[text_name](generated) if text_name else generated
    duration = len(audio) / model.sampling_rate

    rec.heads = []
    log_probs = aligner.ctc_log_probs_from_inputs(inputs)[0].float()
    heads, rec.heads = rec.calls, None
    cut = windows(frames)
    if len(heads) != len(cut):
        raise SystemExit(f"{case}: {len(heads)} calls of the head for {len(cut)} windows")
    window_meta = []
    by_length = {}
    for index, (start, end, cs, ce) in enumerate(cut):
        by_length.setdefault(ce - cs, []).append(index)
    order = [i for length in by_length for i in by_length[length]]
    for call, index in zip(heads, order):
        start, end, cs, ce = cut[index]
        if call["states"].shape[0] != 1 or not torch.equal(call["states"][0].T, asr[cs:ce]):
            raise SystemExit(f"{case}: window {index}'s call is not the input from {cs} to {ce}")
        stages = call["stages"]
        names = [s[0] for s in stages]
        if names != ["upsampled", "embed", *range(len(decoder.transformer.layers)), "final"]:
            raise SystemExit(f"{case}: window {index} ran the head's stages {names}")
        w = os.path.join(out, "windows", str(index))
        save(w, "upsampled", stages[0][1][0].T.numpy())
        save(w, "layers", torch.stack([s[1][0] for s in stages[1:-1]]).numpy())
        save(w, "output", stages[-1][1][0].numpy())
        save(w, "trunk", call["trunk"][0].T.numpy())
        window_meta.append({"start": start, "end": end, "context_start": cs, "context_end": ce,
                            "length": int(call["lengths"][0])})
    save(out, "input", asr.numpy())
    save(out, "log_probs", log_probs.numpy())
    save(out, "speaker_probs", inputs.sortformer_sigmoids[0, :int(inputs.sortformer_lengths[0])].float().numpy())
    native = rec.build["diarization_probs"].float()
    save(out, "diarization_probs", native[:rec.build["diarization_length"]].numpy())
    labels = inputs.diarization_labels[0, :, :int(inputs.diarization_lengths[0])]
    save(out, "diarization_labels", labels.numpy().astype(np.uint8))

    meta = {
        "model": MODEL,
        "nemo_toolkit": NEMO,
        "nemo_commit": NEMO_COMMIT,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "numba": ctc.NUMBA_AVAILABLE,
        "pin": {k: PIN[k] for k in ("repository", "revision")},
        "case": case_name,
        "transcript_change": text_name or None,
        "samples": len(audio),
        "duration": duration,
        "frames": frames,
        "valid_frames": valid,
        "windows": window_meta,
        "blank": BLANK,
        "special_tokens": special_tokens,
        "decoded": decoded,
        "transcript": transcript,
        "diarization_frame_seconds": inputs.diarization_frame_seconds,
        "config": {k: getattr(aligner, k) for k in ("speaker_activity_threshold", "speaker_logprob_weight",
                                                     "maximum_token_len", "max_word_frames", "timestamp_language",
                                                     "epsilon", "online_inference_length", "chunk_left_context",
                                                     "chunk_right_context")},
        "text_stages": text_stages(aligner, transcript),
    }
    try:
        prepared = aligner.prepare_from_inputs(inputs, [transcript], [duration])
    except ValueError as error:
        meta["error"] = {"stage": "prepare", "type": type(error).__name__, "message": str(error)}
        prepared = None
    if prepared is not None:
        record = prepared["records"][0]
        columns = [BLANK] + sorted({label for s in meta["text_stages"]["streams"] for label in s["labels"]} - {BLANK})
        full = log_probs[:record["ctc_length"], columns]
        if not torch.equal(record["ctc_log_probs"], full):
            print(f"{case}: the search's columns differ from the full output's; largest difference "
                  f"{float((record['ctc_log_probs'] - full).abs().max()):.3g}", flush=True)
        save(out, "columns", record["ctc_log_probs"].numpy())
        save(out, "speaker", aligner._resample_speaker_probs(record["speaker_probs"].clamp(0, 1),
                                                             record["ctc_length"]).numpy())
        meta["prepared"] = {
            "config": prepared["config"],
            "columns": columns,
            "ctc_length": record["ctc_length"],
            "ctc_step": record["ctc_step"],
            "speaker_length": record["speaker_length"],
            "sortformer_step": record["sortformer_step"],
            "audio_duration": record["audio_duration"],
            "words": record["words"],
            "streams": [{k: s[k] for k in ("speaker_tag", "labels", "state_to_word")} for s in record["streams"]],
        }
        try:
            result = ctc.align_prepared_batch(prepared)[0]
        except ValueError as error:
            meta["error"] = {"stage": "align", "type": type(error).__name__, "message": str(error)}
            result = None
        meta["mapping"] = rec.mappings
        meta["searches"] = rec.searches
        meta["refined"] = rec.refined
        if result is not None:
            meta["result"] = result
            meta["public"] = plugin._public_result(result)
    meta["seconds"] = round(time.time() - started, 1)
    write_json(os.path.join(out, "meta.json"), meta)
    public = meta.get("public")
    print(f"{case}: {meta['seconds']} s, {frames} frames, "
          + (f"error {meta['error']}" if "error" in meta else
             f"{len(public['words'])} words, mapping {public['speaker_tag_to_diarization_speaker']}, failed streams "
             f"{meta['result']['alignment_diagnostics']['failed_streams']}"), flush=True)
