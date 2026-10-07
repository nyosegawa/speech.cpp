# Qwen3-TTS follows an instruction where the official code does, told by the size label

Supersedes in part docs/adr/0007: Qwen3-TTS 1.7B takes its instruction of how to speak.

Decided 2026-10-07.

## Context

The official `generate_custom_voice()` takes `instruct`, a description in words of how to speak ("怒った口調で"). It
tokenizes it as the turn `<|im_start|>user\n{instruct}<|im_end|>\n` and puts the turn's text embeddings, with no codec
embedding added, before the rows of the prompt. It drops the instruction of a model whose `config.json` gives
`tts_model_size` "0b6", the 0.6B, although the 0.6B's model card shows an example with one. docs/adr/0007 left the
instruction out, since it states no rate a caller could ask for, and the HTTP server refused OpenAI's `instructions` as
a member it did not have. OpenAI's speech API names the same input `instructions`.

Layout 1 has no key that says whether a model takes an instruction, and no `tts_model_size`. The converter writes
`general.size_label` from `tts_model_size` and from nothing else, "0b6" as `0.6B` and "1b7" as `1.7B`, and refuses any
other.

## Decision

- **One option, `instructions`, OpenAI's name**, a string whose neutral value `""` gives none. A model that takes an
  instruction declares the option, and any other refuses a value other than `""` as unsupported, as docs/adr/0014
  refuses what a model cannot follow. The server takes OpenAI's member as the option, as it takes every option by its
  name.
- **A Qwen3-TTS file takes it unless its size label is `0.6B`.** The size label is the official `tts_model_size` under
  the GGUF specification's name, so the reader decides from it what the official code decides from `tts_model_size`,
  and every file of layout 1 keeps working as it is: the 0.6B as before, the 1.7B with the option. The official code,
  which speech.cpp follows, outweighs the 0.6B's model card.
- **The tokens are the official processor's.** The special tokens split the turn and the tokenizer reads
  `user\n{instruction}` as one text, as it reads the text to speak.
- **The instruction takes positions of the talker that the text would otherwise have**, so a text and an instruction
  together beyond the longest text are refused naming `instructions`.

The alternatives were turned down:

- A key of a new layout that says whether a model takes an instruction. Every published file would be converted and
  uploaded again for a fact the size label already holds, and the file would hold it twice.
- Taking the instruction on the 0.6B, or taking it and dropping it as the official code does. The first runs the model
  in a way its official code never does; the second is a silent fallback.
- `instruct`, the official argument's name. OpenAI's clients send `instructions`.

## Consequences

The 1.7B files on Hugging Face take instructions without being converted again, and the 0.6B files refuse them.
`general.size_label`, which only described a Qwen3-TTS model before, now also decides whether it takes an instruction.
`talker-check` compares the instruction's tokens, the prompt and every stage with a dump of the official 1.7B with an
instruction and greedy decoding.
