export const meta = {
  name: 'ace-agenda-design-panel-v2',
  description: 'Five independent agenda designs from different lenses, then three critical judges score and cross-examine them',
  phases: [
    { title: 'Design', detail: 'five independent agenda proposals, one per lens' },
    { title: 'Judge', detail: 'reviewer, operations and integrity judges cross-examine all proposals' },
  ],
}

const REPO = '/Users/pat/code/ACE'
const SP = '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad'
const BRIEF = SP + '/w2/brief.md'
const NOTES = SP + '/w1'
const G = REPO + '/docs/development/guidance'

const COMMON = `
You are one member of a small expert panel preparing an experimental agenda for a research project. You start with no other context, so read carefully.

THE PROJECT. The repository ${REPO} is a research codebase called ACE (Active Causal Experimentalist). The lead researcher wants to strengthen the eventual paper as much as possible. A verified state-of-evidence brief has been prepared at ${BRIEF} . READ IT FIRST AND FULLY. It is the ground truth for this task and is designed to be sufficient on its own. Supporting notes with file paths into the repository are in ${NOTES}/ as <topic>.md files (mechanism, training, numerics-protocol, results-analysis, results-analysis-verify, training-forensics, training-forensics-verify, deployment-forensics, deployment-forensics-verify, ledger-ops, prior-evidence-core, prior-evidence-portfolio, literature-policy, literature-benchmarks, manuscript). Consult notes or repository files only to check specific facts your design depends on, and keep it to roughly fifteen file reads in total. The earlier agenda at ${G}/ace_llm_intervention_agenda_2026-11-02.md is partly superseded by what was learned since, so do not anchor on it.

RULES FOR THIS TASK. This is a reading and design task. Do not modify anything under ${REPO}. Do not run experiments, training, model inference, SLURM, SSH, downloads or installs. Do not use the network. Scratch space, if you want any, is only ${SP}/w2/ . Do not invent facts about the repository: if you rely on something not in the brief or the notes, check it in the files or mark it as an assumption.

STANDING RULES OF THE PROJECT THAT THE AGENDA MUST RESPECT. A negative result must narrow the claim or redirect the method, never trigger a favorable replacement benchmark, so any change of benchmark must be justified by criteria that do not depend on how the language model performed. No tuning against the 24 development outcomes. Freeze methods, namespaces, endpoints and failure treatment before scoring fresh systems. The language model must contribute to the decision, not narrate a numerical optimizer's choice. Keep strong numerical controls. No paid APIs. Installs, new model weights, remote-lab spend and new compute ceilings need the lead researcher's approval, so list them as explicit asks. Do not propose rerunning completed studies or the old pipelines.
`

const AGENDA = {
  type: 'object',
  properties: {
    thesis: { type: 'string', description: 'The paper this agenda aims at: the central claim if results are positive, and the claim if they are null or negative. 4 to 8 sentences.' },
    experiments: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          name: { type: 'string' },
          question: { type: 'string', description: 'The single question this experiment answers' },
          design: { type: 'string', description: 'Arms, controls, systems or data, sample size, endpoint. Concrete.' },
          decision_rule: { type: 'string', description: 'Predeclared pass, fail or kill criteria, and what a positive, null and negative outcome each license in the paper' },
          prerequisites: { type: 'string', description: 'Approvals, new code, dependencies on other experiments' },
          cost: { type: 'string', description: 'Compute, new code, calendar time, with the basis for the estimate' },
          main_risk: { type: 'string' },
          priority: { type: 'string', enum: ['must', 'should', 'could'] },
        },
        required: ['id', 'name', 'question', 'design', 'decision_rule', 'prerequisites', 'cost', 'main_risk', 'priority'],
      },
    },
    sequence: { type: 'string', description: 'Order, gates between stages, and what can run in parallel' },
    theory_items: { type: 'array', items: { type: 'string' }, description: 'Formal statements worth proving or deriving, each with why it strengthens the paper' },
    asks_for_lead: { type: 'array', items: { type: 'string' }, description: 'Decisions or approvals only the lead researcher can give' },
    what_not_to_do: { type: 'array', items: { type: 'string' } },
    biggest_uncertainties: { type: 'array', items: { type: 'string' } },
  },
  required: ['thesis', 'experiments', 'sequence', 'asks_for_lead', 'what_not_to_do', 'biggest_uncertainties'],
}

const DESIGNERS = [
  {
    label: 'policy-repair',
    lens: `YOUR LENS: machine-learning engineer who has to make a learned intervention policy actually learn. You care about supervision quality, data volume, the state interface, the action interface and decoding rule, the optimization recipe, on-policy versus offline feedback, capacity, and behavioral gates that would have caught the constant-action collapse before deployment. Design the sequence of diagnostics and training experiments that first explains the collapse with direct evidence, then establishes whether any learner (numerical first, then the language model) can recover planner-quality decisions from the public state, and then earns the right to test for an advantage. Each step needs a measurable acceptance test. Be specific about targets (for example listwise score distillation versus hard labels), state sampling, counterfactual sensitivity tests, and what would make you abandon the direct-policy route in favor of a hybrid.`,
  },
  {
    label: 'advantage-theory',
    lens: `YOUR LENS: theorist of Bayesian and causal experimental design who asks where a language model can legitimately beat a numerical planner, and where it provably cannot. You care about a clean decomposition of the value gap (information the planner lacks, model misspecification, myopia, decision error, candidate omission), about benchmark classes with verified headroom, about dose-response designs in which the predicted benefit scales with a controlled quantity, and about numerical controls that receive the same information as the language model. Design experiments that test predictions of the theory in both directions: regimes where no benefit is possible and regimes where a bounded benefit is predicted. State the formal results worth proving, including why adaptivity has little value in the completed linear-Gaussian screen and what changes that. Say which comparators from the literature (for example goal-oriented amortized causal design policies) are required.`,
  },
  {
    label: 'experimentalist-task',
    lens: `YOUR LENS: applied scientist who must deliver the concrete experimentalist task with an observable useful outcome and transfer beyond the synthetic generator. You care about a real or realistic apparatus with executable interventions, legal actuators, measured costs in time or money, a useful decision at the end, and semantic content that a language model could exploit. Evaluate the candidates in the brief (Causal Chambers wind tunnel simulators and remote lab, a compressor model through pyCycle or OpenModelica, the sibling repository's engineering simulators, BoxingGym, Gaussian Bayesian networks) and design the task ladder from simulated families with many independent systems to a frozen-policy run on real hardware. Be concrete about the causal graph, the action menu, the cost model, the objective, the number of systems and episodes, what is developed on simulators and what is frozen before hardware, and the approvals and spend required.`,
  },
  {
    label: 'evaluation-statistics',
    lens: `YOUR LENS: skeptical statistician and reviewer for a statistics-leaning machine-learning venue. You care about endpoints with adequate power, replication within and across systems, common random numbers, censoring, multiplicity, preregistration, non-inferiority versus superiority, circularity of model-based endpoints, leakage between training and evaluation, treatment-fidelity reporting, and the controls that informed reviewers will demand (feedback shuffling, prior ladders, anonymized and misleading metadata, state-blind and state-permuted policies, tuned static schedules, semantic-feature numerical baselines, training-seed replication). Design the evaluation program: the measurement repairs needed before any new policy is scored, sample sizes with their basis, the primary and secondary contrasts, and the minimum evidence set that makes the paper acceptable under a positive result and under a null result.`,
  },
  {
    label: 'paper-strategy',
    lens: `YOUR LENS: research strategist working backward from the strongest achievable paper. You care about the claims-to-evidence map, about which results are robust to either outcome of the central question, about reuse of what is already solid (numerical SCM planning, the delivery study, the forensic diagnosis of the failed recipe), about what to cut, about realistic calendar time given the operational overhead in the brief, and about venue fit. Propose the portfolio: the spine experiments that every outcome branch needs, the branch-specific experiments, kill criteria, and two or three alternative paper framings with the evidence each requires. Include what should be done about the public preprint and README that still advertise an invalidated result, framed as a decision for the lead researcher.`,
  },
]

const JUDGMENT = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          designer: { type: 'string' },
          score: { type: 'number', description: '0 to 10 from your lens' },
          strongest_ideas: { type: 'array', items: { type: 'string' } },
          fatal_or_serious_flaws: { type: 'array', items: { type: 'string' } },
        },
        required: ['designer', 'score', 'strongest_ideas', 'fatal_or_serious_flaws'],
      },
    },
    must_include: { type: 'array', items: { type: 'string' }, description: 'Experiments or design elements the final agenda must contain, with the justification' },
    must_drop_or_change: { type: 'array', items: { type: 'string' }, description: 'Proposed elements that should be dropped or changed, with the justification' },
    missing_from_all: { type: 'array', items: { type: 'string' }, description: 'Important things none of the designers proposed' },
    recommended_spine: { type: 'string', description: 'Recommended overall structure and order for the final agenda, 8 to 15 sentences' },
  },
  required: ['scores', 'must_include', 'must_drop_or_change', 'missing_from_all', 'recommended_spine'],
}

const JUDGES = [
  {
    label: 'reviewer',
    lens: `YOUR LENS: senior area chair for causal experimental design and language-model agents at ICML, UAI and TMLR. Judge each proposal by whether its experiments would produce decisive, publishable evidence, which claims would still be attacked, whether comparators and controls match what the literature now demands, and whether the proposal yields a strong paper under both a positive and a null outcome. The literature notes at ${NOTES}/literature-policy.md and ${NOTES}/literature-benchmarks.md are relevant.`,
  },
  {
    label: 'operations',
    lens: `YOUR LENS: principal investigator and research engineer who must execute this in this repository with these constraints. Judge feasibility against the facts: what the code can and cannot express, the remaining compute envelope and the approvals needed, the heavy operational overhead of the previous cycle, the amount of new harness code each experiment implies, and calendar time against the venue dates. Flag every underestimated cost and hidden dependency, and say which experiments give the most evidence per unit of effort. The notes at ${NOTES}/ledger-ops.md and ${NOTES}/numerics-protocol.md are relevant.`,
  },
  {
    label: 'integrity',
    lens: `YOUR LENS: scientific-integrity auditor. Judge each proposal against the standing rules: no favorable replacement benchmark, no tuning on the 24 development outcomes, benchmark and endpoint choices justified by criteria independent of language-model outcomes, the language model must contribute to the decision, numerical controls must receive the same information, handicaps declared in advance. Look for garden-of-forking-paths risks, circular endpoints (for example regret measured under the planner's own score), leakage between training and evaluation systems, unverifiable claims, and outcomes that could not be published honestly. Also judge whether each proposal handles the public preprint and README issue responsibly.`,
  },
]

phase('Design')
const designs = (await parallel(DESIGNERS.map(d => () =>
  agent(
    COMMON + '\n' + d.lens + `\n\nDELIVER: an experimental agenda from your lens, at most 10 experiments, ordered. Every experiment needs concrete arms, controls, sample sizes with their basis, an endpoint, a predeclared decision rule, prerequisites and a cost estimate grounded in the brief. Say what a positive, null and negative outcome each license in the paper. Save the agenda as a markdown document, including its supporting calculations and design rationale, at ${SP}/w2/${d.label}.md, then return the structured agenda.`,
    { label: 'design:' + d.label, phase: 'Design', schema: AGENDA }
  ).then(out => (out ? { designer: d.label, agenda: out } : null))
))).filter(Boolean)

log('designs returned: ' + designs.map(d => d.designer).join(', '))
const missingDesigns = DESIGNERS.map(d => d.label).filter(l => !designs.find(x => x.designer === l))
if (missingDesigns.length) log('DROPPED designers: ' + missingDesigns.join(', '))
if (designs.length === 0) return { designs, judgments: [], note: 'no designs returned, judges skipped' }

phase('Judge')
const packet = JSON.stringify(designs, null, 1)
const judgments = (await parallel(JUDGES.map(j => () =>
  agent(
    COMMON + '\n' + j.lens + `\n\nBelow are ${designs.length} independent agenda proposals (JSON), each from a designer with a different lens. Each designer's proposal document is also saved at ${SP}/w2/<designer>.md if you need more detail. Cross-examine the proposals. Be hard on weak arguments and on anything that conflicts with the brief or the standing rules. Then say what the final agenda must include, what must be dropped or changed, what all of them missed, and the overall spine you recommend. Save your assessment as a markdown document at ${SP}/w2/judge-${j.label}.md, then return the structured judgment.\n\nPROPOSALS:\n` + packet,
    { label: 'judge:' + j.label, phase: 'Judge', schema: JUDGMENT }
  ).then(out => (out ? { judge: j.label, judgment: out } : null))
))).filter(Boolean)

const missingJudges = JUDGES.map(j => j.label).filter(l => !judgments.find(x => x.judge === l))
if (missingJudges.length) log('DROPPED judges: ' + missingJudges.join(', '))
return { designs, judgments }
