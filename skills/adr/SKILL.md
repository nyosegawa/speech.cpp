---
name: adr
description: Write, rewrite, consult or clean up speech.cpp's decision records in docs/adr, one short English file per decision, named by what it decides. Use when a choice between designs is settled (「AとBどっちにする?」→「Bで」「それでいこう」), when an approach is turned down for a lasting reason (「それはやめよう」「やらない」), when a change would contradict a record, when asked to record a decision (ADRにして、記録して、決めたことを残して), and when asked to tidy the records (ADRの整理、ADRを最新にする、まとめる、消す). Not for the reason behind one piece of code (a comment), a rule a check can enforce (the check), a procedure (a skill) or a rule for every task (AGENTS.md).
---

# Decision records

A record keeps what the code cannot show: why speech.cpp does something one way and not another, and which designs
were turned down. An agent lists `docs/adr/` and reads a record only when its file name covers the behavior it is about
to change, so a record is short, found by its name, and true today. The file names are the index; there is no other.

## 1. Whether it is a record

Write one only when all three hold:

- **Hard to reverse.** Undoing it would change the C API, the worker protocol, the GGUF layout, files users hold, or
  what several families do.
- **Surprising without the reason.** A capable agent reading the code would be tempted to "fix" it.
- **A real choice.** Other designs existed and were weighed.

A decision not to do something counts. Anything else goes elsewhere:

| What it is | Where it goes |
|---|---|
| The reason for one piece of code, a quirk of ggml or of the official code, a measured bound | a comment beside that code |
| A rule a check or a smoke script can test | that check or script |
| A procedure for a kind of task | a skill |
| A rule every task follows | AGENTS.md |

When a decision settled in the conversation passes the test, say so and write the record in the same pull request as
the code: 「これは ADR の条件に当てはまるので docs/adr に書きます」.

## 2. Writing one

- File: `docs/adr/<what-it-decides>.md`, lowercase English words joined by hyphens, naming the behavior as a task
  would look for it (`speech-serve-holds-a-model-of-each-task-and-replaces-one-only-after-its-requests-end.md`), never a
  topic (`server.md`). No number: two branches cannot collide on a name the way they do on a number, and if they do,
  the second renames its own.
- The heading is the decision as one sentence. Then `## Context` (what makes a decision necessary, as things are now),
  `## Decision`, the line `The alternatives were turned down:` with each design compared and the lasting reason it lost,
  and `## Consequences` when something follows for programs, users or the code. No date and no status line.
- Measured values carry their condition: the model, the type, the device, and the date where it matters.
- Write only what the code cannot show. No file lists, no implementation steps, no restating of the API.
- Do not invent a reason. A design only mentioned in a conversation is not an alternative, and the conversation itself
  is not context.

## 3. When a decision changes

Keep one true record, not a history:

- Rewrite the record to state the new decision, and move the old design to the alternatives with the reason it was
  dropped. Rename the file when its name no longer says the decision.
- Delete a record whose decision no longer applies at all. Git keeps the history.
- No "Superseded by", "was", "at first" or "before 0.x". A fact that matters only to someone upgrading from an old
  release goes, unless the code still acts on it (as a reader that upgrades an older layout does).

## 4. Nothing points to a record

README.md, `docs/`, AGENTS.md, the skills and code comments name no record: records are rewritten, merged and renamed,
and a citation goes stale. A comment that needs the reason says it in a sentence of its own. Records may link each
other by file name, and those links must resolve.

## 5. Reading records

- List `docs/adr/` and read a record only when its name covers the behavior you are about to change.
- If the change contradicts a record, stop and tell the user which record and why it should change. Rewrite it only
  after the user agrees, in the same pull request as the code.

## 6. Cleaning up the records

The user asks for this from time to time. One pull request covers it.

1. **Audit, without editing.** For each record: read it, read the code it describes, and decide keep, update (which
   statements are stale, and what the code does now, with file and line), merge (with which, and why they are one
   behavior) or delete (why nothing of it needs keeping). List the decisions the code clearly embodies that have no
   record. With many records, hand this reading to a subagent with a brief that says it may not edit anything, and
   check its claims against the code before you rely on them.
2. **Show the plan.** A table of record → action → new name → reason, in Japanese, to the user. Deleting or merging a
   record changes what future work reads, so do not rewrite before the user has seen the table, unless they asked for
   the cleanup to go ahead.
3. **Rewrite** in a worktree of its own (pull-request skill): the records as section 2 says, a merged record taking the
   best name, every citation outside `docs/adr/` removed as section 4 says. Do not write a record for a decision whose
   reason the user has not stated; list it for them instead.
4. **Check:** every relative link resolves; `git grep -n 'Superseded'` and `git grep -n 'docs/adr/[0-9]'` find nothing;
   the build passes when comments in code changed.
5. **Report** in the pull request: the table as done, where the code disagreed with the audit, and the decisions left
   without a record for the user.
