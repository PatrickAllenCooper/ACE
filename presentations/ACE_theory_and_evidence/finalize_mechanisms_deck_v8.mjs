import fs from 'node:fs/promises';

import {resolvePresentationFont,finalizePresentation} from '/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations/container_tools/artifact_tool_utils.mjs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const ROOT='/Users/pat/code/ACE',OUT=ROOT+'/presentations/ACE_theory_and_evidence',DIR=ROOT+'/.codex-artifacts/ace-mechanism-deck-v8/build';
const SKILL='/Users/pat/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const INPUT=OUT+'/ACE_mechanisms_and_evidence_v7.pptx';
const FONT='Helvetica Neue';
console.log(JSON.stringify(await finalizePresentation({explicitTotalSlideCount:12,workspaceDir:ROOT,candidatePath:DIR+'/draft-preserved.pptx',finalPath:OUT+'/ACE_mechanisms_and_evidence_v8.pptx',pythonExecutable:'/Users/pat/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...([4,7,8,9,12].flatMap(n=>['--require-native-table-slide',String(n)]))],requiredNativeTableOwnerSlides:[4,7,8,9,12],requiredNativeChartOwnerSlides:[11,12],fontPolicy:{basis:'reference',families:[FONT],referencePath:INPUT,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(INPUT)).digest('hex')},verifyArtifactToolImport:true,receiptPath:DIR+'/validation-v8.json'})));
