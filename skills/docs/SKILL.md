---
name: docs
description: Keep speech.cpp's README.md and docs/ current, short and plain. Covers which page describes what, when a change must update a page, how to write a page, and what to check before committing. Use when a change alters what a user does or sees (a command, an option, an endpoint, a model, a file layout, a number), when writing or rewriting a page, or when asked to update, fix, shorten or check the documentation (ドキュメントを更新、README を直す、docs に書く、説明を短く). Not for decision records (docs/adr/, which the change's ADR covers) or code comments.
---

# The documentation

README.md is the shortest path from install to speech, about 200 lines. `docs/` has one page per topic and
`docs/development/` the pages for people who change speech.cpp. `docs/README.md` is the index. A page a user cannot
find from the README or the index does not exist for them.

## Which page

| A change to | Updates |
|---|---|
| a subcommand or its flags | docs/cli.md; README's table if the subcommand is new |
| a request option, a voice, a model's limits | the family's page in docs/models/; docs/models.md for voices and languages |
| the catalog (a model, a name, a type) | docs/models.md and README's model table |
| `speech serve`, its page or the HTTP API | docs/server.md |
| the worker protocol | docs/worker.md |
| `speech.h` | docs/c-api.md |
| a GGUF layout, a key, a converter | docs/gguf.md, in the same commit as the reader (AGENTS.md) |
| building, CI, release archives | docs/build.md, docs/install.md |
| the installers | docs/install.md and README's Install |
| a stage check, a measurement | docs/development/checks.md; the family page's short table if a headline number moves |
| the release steps | docs/development/releasing.md |

The pull request that changes the behaviour updates its pages. Nothing checks the docs against the code by test, and
none may be added: a test of wording fails on a deliberate rewording and points to no defect. Docs stay current
because the change's author reads the pages above and edits them.

## Writing

- English, plain and concrete: short sentences, common words, commands a reader can copy and run.
- Each page opens with one sentence saying what it is for.
- A small table or a code block beats a paragraph. A paragraph that only introduces the next one is cut.
- Keep pages short. A fact may appear in the README and a page both; brevity matters more than one place per fact.
- Numbers carry their conditions: the model, the type, the device, and the date only where it matters.
- Measured detail, check tables and the evidence for bounds go to docs/development/checks.md; the reasons for a choice
  go to an ADR. A user's page links to them instead of repeating them.
- No marketing words ("blazing", "seamless", "robust", "powerful") and no filler ("Let's dive in", "In this page we
  will").

## Before committing

1. Every relative link resolves, anchors included. Check them with a short script you run and do not commit.
2. Tables render: every row has as many cells as its header (`gh api /markdown` renders GitHub's way).
3. README.md stays near 200 lines; if a change pushes it past, move detail to a page.
4. Commands in the README's quick start run as written against the current catalog.
