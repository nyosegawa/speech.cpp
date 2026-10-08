# The model card

The card is the `README.md` of a model's repository on Hugging Face. It is the page someone reaches from a search or
from the catalog's link, so it says what the file is, how to use it with speech.cpp, and where it comes from, and leaves
the rest to speech.cpp's documentation. The docs skill's rules for a page hold here too: the current release alone, no
history of earlier files (the repository's history keeps them), no internals of the port, plain English.

## Front matter

```yaml
---
license: <the upstream model's SPDX id>
base_model:
  - <upstream repository>
  - <the codec's or frontend's repository, where one is inside the file>
base_model_relation: quantized
pipeline_tag: text-to-speech | automatic-speech-recognition
language:
  - <each language code>
tags:
  - gguf
  - ggml
  - speech.cpp
  - <the family, such as irodori-tts>
---
```

## Sections, in this order

1. **`# <Upstream name> GGUF for speech.cpp`**, then one paragraph: the file runs only in speech.cpp `<speech.requires>`
   or later, because its layout is speech.cpp's own, so llama.cpp, LM Studio, Ollama and other GGUF readers cannot run
   it. Then two or three sentences on what the file holds and what the model does.
2. **`## Use`**: a link to speech.cpp's install, then two to four commands by the catalog's name (`speech tts <name> …`,
   `speech asr <name> …`, `speech serve <name> --open`), a sentence saying that the name fetches this file and that a
   path to the downloaded file works as well, and links to the family's page, the command line and the server in
   speech.cpp's docs. No table of options: the docs page holds it, and `speech info` prints it.
3. **`## Files`**: a table of each file with what it holds, its size in GB and in bytes, and its SHA-256, the `.json`
   beside it, and the license files.
4. **What a user of this model needs and the docs page would not show from here**, only where there is something: the
   voices of a synthesis model and what a voice file binds to, or the languages of a recognizer.
5. **`## Accuracy`**: the headline of the stage checks against the official implementation, with the device, and one
   end-to-end number where there is one (a transcription's CER, a WER).
6. **`## Speed`**: the headline table with its conditions (the prompts, the device, the type, one request at a time).
7. **`## Source`**: the converter and the pinned commit of the official code its environment uses, each upstream
   repository at the revision converted, and `speech quantize --type <type>` where the type was made from F32. Then a
   sentence naming the sibling models and their repositories.
8. **`## License`**: each upstream license with its file, and the upstream card's ethical restrictions word for word
   where it has them.

## Making the cards

Write the cards of one family's repositories from one script that fills the same text with each model's values, so
that siblings do not drift apart, and compute every size and SHA-256 from the files being uploaded, never by hand. Read
each card as Hugging Face renders it before the upload.
