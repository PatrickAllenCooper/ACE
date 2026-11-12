# Offline diagnosis behind the November 12 agenda

November 12, 2026. These files are the session analyses that support [the experimental agenda](../../docs/development/guidance/ace_llm_experimental_agenda_2026-11-12.md). AI agents running in Claude Code workflows produced them in one session. The reader workflow (01) ran on claude-fable-5-2, and the design panel (02) and review (03) ran on claude-opus-5-5. No human has reviewed them. This is post hoc analysis of development data. It is not a scientific run, not a freeze and not the independent review that submission condition 5 requires.

## What the agents did and did not do

- They read saved repository files only, mainly under `results/ace_llm_intervention_scientific_*_20261108/` and `scripts/research/`. Nothing in the repository was modified.
- They ran no experiments, model inference, weight loading, SLURM jobs, SSH sessions or installs. Saved prompt token ids were decoded with the pinned tokenizer files only.
- Only the two literature agents used the network, to read public papers and documentation.
- The results analysis reconstructed the true target means of the 24 development systems from saved predictions and errors. Those systems are already reported, and the agenda retires them from every evaluation role.

## Verification scope

The results-analysis, training-forensics and deployment-forensics topics were each recomputed by a second agent with fresh code, and their `-verify.md` notes record the verdicts. No figure was refuted, and several were narrowed. The other topics (mechanism, training, numerics-protocol, ledger-ops, prior-evidence-core, prior-evidence-portfolio, literature-policy, literature-benchmarks, manuscript) rest on one agent. Figures about earlier studies were read from project documents rather than recomputed.

## Layout

| Folder | Contents |
| --- | --- |
| `w1/` | Reader notes, one `<topic>.md` per topic, plus per-topic folders of analysis scripts and outputs. The three verified topics also have `<topic>-verify.md` and their own script folders. `w1/deployment-forensics-verify/decisions_verifier.csv` holds 16,384-draw target-risk reference scores for all eight actions at all 4,371 decisions of the screen. |
| `w1_structured/` | Plain-text renderings of each reader's structured return, including the verifier verdicts. |
| `w2/` | `brief.md`, the state-of-evidence brief given to the design panel, five design proposals (`<designer>.md`), three judgments (`judge-<lens>.md`) and the panel's calculations (`calc/`). |
| `w2_structured/` | Plain-text renderings of the panel's structured returns. |
| `w3/` | `review.txt`, the independent fact-check and critique of the agenda draft (68 issues, all resolved in the agenda), and `draft_evidence.md`, an early draft of the agenda's evidence section that the agenda supersedes. |
| `workflow-returns/` | Raw JSON returns of the three workflow runs: 01 readers and verifiers, 02 design panel, 03 agenda review. |
| `workflow-scripts/` | The exact workflow scripts, which contain every agent's instructions and rules. |

A first design-panel run returned no designs or judgments and is not preserved here. Run 02 is the one used.

## Known errata in these notes

The agenda corrects all of these. Do not copy them forward from the notes.

1. `w2/brief.md` section 4 and `w1/prior-evidence-portfolio.md` say Qwen2.5-7B-Instruct is cached locally. Only its tokenizer and config files are cached, and the same holds for Qwen2.5-1.5B-Instruct. Model weights are cached only for Qwen3-0.6B and base Qwen2.5-1.5B.
2. `w2/brief.md` says the eight target queries are not distinct, so X = −1 carries double weight. That holds for the fork only. In the chain do(Y = 1) appears twice, and in the collider all eight queries are distinct (`w1/numerics-protocol.md`, section 1).
3. `w2/brief.md` says every number in it was derived by one analyst and re-derived by a second. Only the three topics listed above have second-agent verification.
4. `w3/review.txt` lists further precision corrections, such as the 23.5-hour gap spanning a night and the paired standard deviation varying by contrast.

## Paths and rerunning

The notes and scripts use absolute paths under the original session directory, `/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/`. That directory corresponds to this one, so `.../scratchpad/w1/mechanism.md` is `w1/mechanism.md` here. Scripts read repository files by absolute path and write outputs to the old session location, so adjust paths before rerunning any of them. They were run with `/usr/bin/python3` (Python 3.9 with numpy, pandas and scipy).

## Excluded files

Forty binary cache files (Python pickles and numpy arrays, 116,086,151 bytes in total) were not copied. They are intermediate products that the scripts regenerate. Their SHA-256 hashes are listed in `excluded-binary-caches.sha256.txt`. Never load a pickle from a copy you cannot trust.

## Integrity and Git

`sha256sums.txt` lists every file in this directory except itself. To check the copy, run this from this directory:

```bash
shasum -a 256 -c sha256sums.txt
```

The repository ignores `*.out` and `*.log` files globally. The `.gitignore` in this directory re-includes them because the notes cite those output files by name. As of November 12 the directory is not committed.
