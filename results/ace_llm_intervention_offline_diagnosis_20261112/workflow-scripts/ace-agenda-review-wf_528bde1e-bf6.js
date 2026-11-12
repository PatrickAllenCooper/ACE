export const meta = {
  name: 'ace-agenda-review',
  description: 'Independent fact-check and critical review of the written ACE experimental agenda',
  phases: [
    { title: 'Review', detail: 'fact-check and consistency/compliance review in parallel' },
  ],
}

const REPO = '/Users/pat/code/ACE'
const DOC = REPO + '/docs/development/guidance/ace_llm_experimental_agenda_2026-11-12.md'
const SP = '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad'

const COMMON = `
You are reviewing a research planning document before it goes to a principal investigator. You start with no other context.
The document: ${DOC} . Read it fully first.
Supporting evidence prepared earlier in this session: a verified brief at ${SP}/w2/brief.md, detailed topic notes at ${SP}/w1/<topic>.md (mechanism, training, numerics-protocol, results-analysis, results-analysis-verify, training-forensics, training-forensics-verify, deployment-forensics, deployment-forensics-verify, ledger-ops, prior-evidence-core, prior-evidence-portfolio, literature-policy, literature-benchmarks, manuscript), five panel designs at ${SP}/w2/<designer>.md (policy-repair, advantage-theory, experimentalist-task, evaluation-statistics, paper-strategy) and three panel judgments at ${SP}/w2/judge-<lens>.md (reviewer, operations, integrity).
RULES: Read-only. Do not modify the document or anything under ${REPO}. No experiments, model execution, network, SLURM or installs. You may run small read-only shell or /usr/bin/python3 checks against repository files and saved CSVs to verify a claim. Keep total file reads to roughly twenty.
Report only problems you can substantiate, each with the exact quote from the document, what is wrong, the evidence (file path and line or computation), and a concrete replacement text. Do not report stylistic preferences unless they cause misreading. If you find nothing wrong in a category, say so.
`

const ISSUES = {
  type: 'object',
  properties: {
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['error', 'misleading', 'omission', 'minor'] },
          quote: { type: 'string', description: 'Exact text from the document' },
          problem: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string', description: 'Concrete replacement text or addition' },
        },
        required: ['severity', 'quote', 'problem', 'evidence', 'fix'],
      },
    },
    verified_ok: { type: 'array', items: { type: 'string' }, description: 'Important claims you checked and found correct' },
    overall: { type: 'string' },
  },
  required: ['issues', 'verified_ok', 'overall'],
}

phase('Review')
const [facts, critique] = await parallel([
  () => agent(COMMON + `
YOUR TASK: fact-check. Verify every number, count, ratio, date, file path, line reference, citation detail and factual statement in the document against the brief, the topic notes (including the -verify notes, which supersede the originals where they qualify a figure), and the repository. Pay special attention to: the 'What the completed evidence says' section; the cited code line numbers in 'Evidence and provenance'; venue dates; Causal Chambers pricing and terms; compute and envelope figures; and any figure the verifiers marked as partially confirmed or seed-dependent. Flag numbers that are stated more precisely or more strongly than the evidence supports.`,
    { label: 'review:facts', phase: 'Review', schema: ISSUES }),
  () => agent(COMMON + `
YOUR TASK: critical review of consistency, compliance and completeness.
(1) Internal consistency: experiment IDs, gate names, dates, sample sizes and decision rules that contradict each other across sections; references to things that are not defined.
(2) Compliance with the project's standing rules: no favorable replacement benchmark (class changes must rest on criteria independent of language-model results); no tuning on the 24 development outcomes; freeze before scoring; the language model must contribute to the decision rather than narrate a numerical optimizer; numerical controls receive the same information; frozen LLM, supervised LLM, numerical ranker and target-risk planning kept as controls; no paid APIs without approval; installs, weights and spend listed as asks. Also the prose style rules: no em dashes, no semicolon chains, the word 'intervention' used consistently.
(3) Completeness against the three judges' MUST_INCLUDE and MISSING_FROM_ALL lists in ${SP}/w2/judge-reviewer.md, judge-operations.md and judge-integrity.md: list any required element the document omits or weakens, and say whether the omission matters.
(4) Clarity for the reader (the principal investigator and a coauthor): places where a reader would misunderstand what is being proposed, what it costs, or what they are being asked to decide.`,
    { label: 'review:critique', phase: 'Review', schema: ISSUES }),
])
return { facts, critique }
