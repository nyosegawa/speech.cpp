# A Nemotron 3.5 Transcribe request transcribes verbatim with the model card's prompts, greedily until the end of its turn

## Context

Nemotron 3.5 Transcribe is a speech LLM that does what its user turn asks of the audio: transcription in a chosen style,
contextual transcription, speaker-attributed transcription, translation, language identification and answers to
questions, with reasoning for the last. Its official request is a chat completion to NVIDIA's vLLM plugin (vLLM 0.32):

- a system turn, which the model card keeps empty for transcription, and a user turn of a text and one recording;
- `temperature`, `top_p`, `top_k` and `seed`, which the card sets to 0, 1, off and 0 for transcription and to 0.6,
  0.95, off and 0 for reasoning, and `max_tokens`, 16,384 in the card's example, which it says to raise or remove for
  long recordings, since a dense hour needs about 20,000 tokens;
- `chat_template_kwargs.enable_thinking` and the reasoning budgets, off for transcription;
- `skip_special_tokens` and `spaces_between_special_tokens`, which keep the speaker and event tags in the text;
- `mm_processor_kwargs.capture_ctc_timestamps`, which aligns the text to the audio with the checkpoint's CTC head and
  returns word times, speaker segments and the mapping between the two.

The server's own settings, MTP or DFlash2 speculative decoding, do not change what greedy decoding writes. The card
reports its transcriptions after a generation-loop guardrail that collapses a character repeated 21 times or more and a
block of up to 4096 characters repeated 20 times or more. The model takes 65,536 tokens and recordings up to an hour,
at 80 ms an audio token.

## Decision

- **Verbatim transcription.** The user turn is the card's recommended prompt, the one of its English, multilingual and
  long-form results ("Audio transcription. Language: as spoken. Wording: verbatim, ..."), with no system turn and
  reasoning off, rendered as the vLLM plugin renders it: the text, one space and the audio locator.
- **The option `prompt`** is the context terms of the card's contextual transcription: a request with a prompt asks
  "Transcribe with punctuation and capitalization. Context terms: <prompt>." in place of the verbatim prompt.
- **`language` is checked, not steering.** The card asks for the language as spoken in every result it reports, and
  never names one.
- **Greedy decoding until the end of the turn or the context.** No sampling option and no limit of new tokens below
  the 65,536 the context holds; a request whose text fills it reports so. Recordings longer than an hour, the longest
  the card's server takes, are refused.
- **The text keeps the special tokens the model writes**, as the card's request does, and passes through the card's
  guardrail before it is returned.
- **Not offered:** a system turn, an instruction of the caller's own, translation, language identification, answers
  and reasoning, which are other tasks than recognition; and timestamps, whose alignment the port does not compute
  yet. The file holds the CTC head and its settings, so that timestamps need no new file.

The alternatives were turned down:

- Steering the language by writing it into the prompt ("Language: Japanese."). The card evaluates no such prompt.
- 16,384 new tokens, the card's example. It cuts a dense recording of more than about 45 minutes.
- The caller's own instruction through the option `instructions`, which only synthesis takes. It would make a
  recognition's result whatever the instruction asks for, translations and answers included, and the speaker tags it
  can ask for would want a place in the result of their own.

## Consequences

A request writes what vLLM writes for the card's prompt at temperature 0, but for the guardrail, which only collapses a
loop the model fell into. Speaker-attributed transcription, translation and timestamps each need a decision of their own
before a request can ask for them.
