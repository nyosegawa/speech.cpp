---
name: pull-request
description: Take a change to speech.cpp from a branch to main. Covers the worktree and branch, the checks a change runs, commit messages, the pull request, one Codex review, CI, the squash merge and cleaning up, and taking back a subagent's branch. Use when starting a change, committing, pushing, opening, reviewing or merging a pull request, or waiting for CI (PRを出す、レビュー、CIを待つ、マージ、片づけ). Not for tagging a release (release) or for checks on a Windows machine (windows-check).
---

# From a branch to main

Every change reaches main through a pull request; nobody commits on main. A pull request is one coherent unit, and
may be large: a model's options, a feature with its page and docs, a set of fixes in one area.

## 1. Start

- Work in a worktree of its own, on a branch from `origin/main`:
  `git worktree add -b <branch> ../speech.cpp-wt/<name> origin/main`. Link the shared `models/` folder and, if the
  change runs dumps, `reference/<family>/out` and `.venv`; never commit the links.
- Read the ADRs whose names cover the behaviour you change before changing it.

## 2. Check and commit

- Build and run the checks the change touches; AGENTS.md's Workflow names them (stage checks, `speech-api-check`, the
  smoke scripts with a model of each family). A defect fix comes with a check that fails before it.
- Commit as each part works. The message is one English sentence in the imperative without a prefix, then what
  changed and why, then the checks run and what they gave ("Checked: …"), as earlier commits write them.
- A decision the code cannot show gets an ADR in the same pull request.

## 3. Open the pull request

`git push -u origin <branch>` and `gh pr create --base main --title "<one sentence>" --body-file <file>`. The body says
what changed and why, any behaviour that changes for existing users, what was checked and on what, and what was not.
A branch whose Windows or Linux code was not compiled locally can open as a draft first, so that CI compiles it.

## 4. One review

Run Codex once on the branch: `codex review --base origin/main -c model='"gpt-6-astra"'`. Read every finding; fix the
real ones where their cause is, each with a check that fails without the fix where one can be written. Do not run the
review again after fixing. Record anything checked outside CI (a Windows run, a page tried in a browser) as a comment on
the pull request.

## 5. CI and merge

- `gh pr checks <n> --watch` until every job ends. A failure is read from its log (`gh run view <run> --log-failed`)
  and fixed on the branch.
- Merge with a squash once CI passes and the review's findings are fixed, when the maintainer has asked you to merge:
  `gh pr merge <n> --squash --delete-branch`. Otherwise leave it for the maintainer.

## 6. Clean up

`git worktree remove --force <path>`, `git branch -D <branch>`, and confirm with `git worktree list`. Remove temporary
branches pushed for CI runs (`git push origin --delete <branch>`).

## A subagent's branch

A subagent commits on its own branch and never pushes or merges. Read its report, then its diff yourself
(`git diff origin/main...<branch>`), and check that the diff is what the report says before you push it. Send fixes back
to the same subagent rather than finishing its work in its worktree. When main moves under it, have it rebase onto
`origin/main`.
